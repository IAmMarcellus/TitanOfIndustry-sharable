"""FastAPI front for the Mergatroid voice sidecar.

Exposes:
  POST  /api/offer   — SDP offer/renegotiate (relayed from Paperclip's admin-gated route; the
                       browser never hits this port directly)
  PATCH /api/offer   — trickle ICE candidates
  GET   /health      — liveness + config summary (start-stack gate + Paperclip token preflight)

Single-operator service: ConnectionMode.SINGLE rejects a second concurrent call at signaling time.
Model weights are warmed at startup so the first call doesn't pay the cold load.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    ConnectionMode,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

import bot

_ready = False


async def _warm_up() -> None:
    global _ready
    try:
        await asyncio.to_thread(bot.warm)
        _ready = True
        logger.info("voice models warm (whisper={}, kokoro voice={})", bot.WHISPER_MODEL, bot.KOKORO_VOICE)
    except Exception as err:
        # Stay up but unready — /health reports it and the token preflight will surface it.
        logger.error("voice model warm-up failed: {}", err)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_task = asyncio.create_task(_warm_up())
    yield
    warm_task.cancel()
    await handler.close()


app = FastAPI(lifespan=lifespan)

handler = SmallWebRTCRequestHandler(connection_mode=ConnectionMode.SINGLE)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "ready": _ready,
        "stt": bot.WHISPER_MODEL,
        "tts": bot.KOKORO_VOICE,
        "llm": os.environ.get("VOICE_LLM_MODEL", "qwen-voice"),
    }


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    webrtc_request = SmallWebRTCRequest.from_dict(request)

    async def on_connection(connection: SmallWebRTCConnection) -> None:
        background_tasks.add_task(bot.run_bot, connection)

    answer = await handler.handle_web_request(
        request=webrtc_request, webrtc_connection_callback=on_connection
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await handler.handle_patch_request(request)
    return {"status": "success"}
