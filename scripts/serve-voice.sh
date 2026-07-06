#!/usr/bin/env bash
# Self-hosted Mergatroid voice sidecar (Pipecat): SmallWebRTC transport + Silero/SmartTurn turn-taking
# + faster-whisper STT (CPU) + qwen-voice brain (27B via the LiteLLM proxy — NEVER Ollama direct) +
# Kokoro TTS (CPU). Paperclip relays browser signaling to the configured sidecar endpoint when
# VOICE_PROVIDER=pipecat; see voice/ for the service and docs/pipecat-voice.md (vendor/paperclip).
#
# CPU-only by design — the 3090 stays fully owned by the 27B. Whisper threads are capped so STT
# doesn't fight Kokoro (or the rest of the box) for cores.
set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${VOICE_WHISPER_THREADS:-4}"

exec uv run --project voice python -m uvicorn server:app \
  --app-dir voice \
  --host "${VOICE_PIPECAT_HOST:-127.0.0.1}" \
  --port "${VOICE_PIPECAT_PORT:?set VOICE_PIPECAT_PORT in .env}"
