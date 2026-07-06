"""The Mergatroid voice pipeline: one bot per WebRTC connection.

Transport → Silero VAD + SmartTurn v3 (turn end) → faster-whisper (CPU) → context aggregation →
qwen-voice (27B via the LiteLLM proxy — NEVER Ollama direct; the proxy owns routing + concurrency
caps) → Kokoro (CPU) → transport. Read-only Paperclip tools; turns persisted to the Conference Room
thread. There is no ElevenLabs-style stall timeout anywhere in this pipeline — fillers are UX only.
"""

import asyncio
import functools
import itertools
import os

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

import paperclip
from prompt import build_system_message
from tools import PAPERCLIP_TOOLS, TOOL_HANDLERS

# Single source for the model knobs — server.py imports these for /health, and the warm-up loads
# exactly what the calls use.
WHISPER_MODEL = os.environ.get("VOICE_WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.environ.get("VOICE_WHISPER_COMPUTE", "int8")
KOKORO_VOICE = os.environ.get("VOICE_KOKORO_VOICE", "af_heart")

FILLER_PHRASES = ["Let me check that.", "One second while I look.", "Let me pull that up."]
TOOL_FILLERS = itertools.cycle(FILLER_PHRASES)


def strip_fillers(text: str) -> str:
    """Remove tool-call filler phrases from an assistant transcript before persisting it.

    The assistant aggregator faithfully includes everything that was spoken — fillers too — but the
    durable Conference Room thread should only carry the substantive reply (same rule as the
    ElevenLabs shim's stream-only fillers). Exact-phrase matches only; we own the list.
    """
    for phrase in FILLER_PHRASES:
        text = text.replace(phrase, "")
    return " ".join(text.split())


@functools.cache
def get_whisper_model():
    """The CTranslate2 whisper model, loaded once per process and shared across calls.

    Pipecat service instances are single-pipeline, so each call builds a fresh service shell — but
    the heavyweight model load (~1s + hundreds of MB) must not be paid per call.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)


class SharedModelWhisperSTT(WhisperSTTService):
    """WhisperSTTService whose _load reuses the process-wide model instead of constructing one."""

    def _load(self):
        self._model = get_whisper_model()


def make_stt() -> WhisperSTTService:
    return SharedModelWhisperSTT(model=WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)


def make_tts() -> KokoroTTSService:
    return KokoroTTSService(voice_id=KOKORO_VOICE)


def warm() -> None:
    """Blocking model warm-up (call via asyncio.to_thread): whisper shared model + one kokoro synth.

    The kokoro ONNX session is rebuilt per call (~1s, off-loop) — warming here just pre-downloads
    its weights and shakes out load errors at startup instead of on the first call.
    """
    get_whisper_model()
    tts = make_tts()
    # Private attr poke into the pinned pipecat-ai==1.4.0 wheel; re-check on version bumps.
    tts._kokoro.create("Warm up.", voice=KOKORO_VOICE, lang="en-us", speed=1.0)


async def run_bot(connection: SmallWebRTCConnection) -> None:
    """Build and run the pipeline for one call. Returns when the call ends."""
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # Model/session construction takes ~1-2s of CPU — keep it off the event loop so signaling
    # (trickle ICE, /health) stays responsive during call setup.
    stt, tts, vad = await asyncio.to_thread(
        lambda: (make_stt(), make_tts(), SileroVADAnalyzer(params=VADParams(stop_secs=0.4)))
    )

    llm = OpenAILLMService(
        base_url=os.environ.get("VOICE_LLM_BASE_URL", ""),
        api_key=os.environ.get("VOICE_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        model=os.environ.get("VOICE_LLM_MODEL", "qwen-voice"),
    )
    for name, handler in TOOL_HANDLERS.items():
        llm.register_function(name, handler)

    digest = await paperclip.fetch_digest()
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_message(digest)}],
        tools=PAPERCLIP_TOOLS,
    )
    # In pipecat 1.4 turn-taking lives in the USER AGGREGATOR, not the transport: Silero VAD detects
    # speech; the default UserTurnStrategies already stop the turn with SmartTurn v3 (CPU, bundled).
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
    )

    rtvi = RTVIProcessor(transport=transport)

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[RTVIObserver(rtvi)],
    )

    # Filler while tool calls run — pure UX, nothing times out on this path.
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await llm.push_frame(TTSSpeakFrame(next(TOOL_FILLERS)))

    # Keep the system-slot status snapshot fresh: refetch at each user turn start (the server caches
    # it ~15s, so this is nearly free) and rewrite messages[0] in place. One stable system slot keeps
    # the prompt prefix stable for Ollama's prompt cache.
    refresh_task: asyncio.Task | None = None

    @aggregators.user().event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        nonlocal refresh_task

        async def refresh():
            fresh = await paperclip.fetch_digest()
            messages = context.messages
            if fresh and messages and messages[0].get("role") == "system":
                messages[0]["content"] = build_system_message(fresh)

        if not refresh_task or refresh_task.done():
            refresh_task = asyncio.create_task(refresh())

    # Persist finalized turns to the durable Conference Room thread. The aggregators hand us exactly
    # what entered the context: the aggregated user transcript, and the interruption-aware assistant
    # text (spoken fillers included — scrubbed before persisting).
    @aggregators.user().event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message):
        paperclip.persist_turn_bg("user", message.content)

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        paperclip.persist_turn_bg("assistant", strip_fillers(message.content))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("voice client disconnected — ending pipeline")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    logger.info("voice call starting (digest {} chars)", len(digest))
    await runner.run(task)
    logger.info("voice call ended")
