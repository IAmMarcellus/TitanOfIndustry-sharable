#!/usr/bin/env bash
# CPU embedding server for TitanOfIndustry shared memory — bge-base (768-dim), OpenAI-compatible.
#   Served model id: bge-base   (matches EMBEDDING_MODEL=openai/bge-base)
#
# Runs ENTIRELY ON CPU (HuggingFace Text Embeddings Inference, CPU image) so it never competes with
# the Qwen chat model for the 3090's VRAM. The dream pass runs offline, so CPU latency is a non-issue;
# this also upgrades live recall to vector mode at zero VRAM cost. Memory falls back to keyword search
# whenever this is down, so it stays OPTIONAL.
#
# First-time bring-up:
#   make embed-cpu                       # start this server
#   make memory-embed-backfill           # embed pre-existing memories
#   then set EMBEDDING_MODEL=openai/bge-base in .env  (keep EMBEDDING_DIM=768 — matches the index)
#
# Not on Docker? Serve bge-base on CPU any other way that exposes OpenAI /v1/embeddings
# (TEI binary, or a llama.cpp / sentence-transformers shim) and point EMBEDDING_BASE_URL at it.
set -euo pipefail

MODEL="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
PORT="${EMBED_PORT:?set EMBED_PORT in .env}"
CONTAINER_PORT="${EMBED_CONTAINER_PORT:?set EMBED_CONTAINER_PORT in .env}"
IMAGE="${EMBED_TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:cpu-1.7}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker, or serve bge-base on CPU another way exposing OpenAI" >&2
  echo "/v1/embeddings via the endpoint configured in .env." >&2
  exit 1
fi

mkdir -p "$HF_CACHE"
echo "Serving $MODEL on CPU at the endpoint configured in .env ..."
# Map the local port to the container port configured in .env. The OpenAI route accepts any model field.
exec docker run --rm --name titanofindustry-embed-cpu \
  -p "$PORT:$CONTAINER_PORT" \
  -v "$HF_CACHE:/data" \
  "$IMAGE" \
  --model-id "$MODEL"
