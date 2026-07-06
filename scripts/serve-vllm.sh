#!/usr/bin/env bash
# Start the local vLLM CHAT server for TitanOfIndustry (Qwen codex AWQ on the RTX 3090), served as
# 'qwen-codex'.
#
# Default: the repo's parameterized engine script (scripts/serve-vllm-engine.sh text) — CUDA graphs ON
# (no --enforce-eager), fp8 KV cache, 32K/4-seq. That script is the single source of truth for the text
# flags. The chat model uses ~93% of the 3090, so there's no room for a second embed vLLM (memory runs
# keyword-only unless you free VRAM).
#
# Set VLLM_USE_PERSONAL=1 to exec your hardcoded ~/.local/bin/vllm-serve.sh instead (e.g. to A/B the
# eager-vs-graphs configs).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${VLLM_USE_PERSONAL:-0}" = "1" ]; then
  LAUNCHER="${VLLM_LAUNCHER:-$HOME/.local/bin/vllm-serve.sh}"
  if [ -x "$LAUNCHER" ]; then
    exec "$LAUNCHER"
  fi
  cat >&2 <<EOF
error: VLLM_USE_PERSONAL=1 but launcher not found/executable at: $LAUNCHER
Unset VLLM_USE_PERSONAL to use the repo engine script, or fix the launcher path.
EOF
  exit 1
fi

exec bash "$HERE/serve-vllm-engine.sh" text
