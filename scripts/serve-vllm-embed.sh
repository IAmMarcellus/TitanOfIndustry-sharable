#!/usr/bin/env bash
# Serve a small embedding model via vLLM for TitanOfIndustry shared memory.
#   Served model id: bge-base   (matches EMBEDDING_MODEL=openai/bge-base)
#
# Small (~0.4 GB) but shares the 3090 with the Qwen chat server — keep EMBED_GPU_UTIL modest and
# trim VLLM_GPU_UTIL in serve-vllm.sh if VRAM is tight. Memory falls back to keyword (full-text)
# search when this server is down, so it is OPTIONAL.
set -euo pipefail

MODEL="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
PORT="${EMBED_PORT:?set EMBED_PORT in .env}"
GPU_UTIL="${EMBED_GPU_UTIL:-0.10}"

exec vllm serve "$MODEL" \
  --served-model-name bge-base \
  --task embed \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT"
