#!/usr/bin/env bash
# Parameterized vLLM launcher for the DUAL-model (C3a/C3b) setup: shared flags once, a per-mode block
# for the deltas. Driven by the model-manager, which overrides util/seqs per mode via VLLM_* env vars.
#
#   usage: serve-vllm-engine.sh text|vision
#
# This is ALSO the standalone text path: scripts/serve-vllm.sh execs `… text` by default, so `make vllm`
# lands here (set VLLM_USE_PERSONAL=1 to use ~/.local/bin/vllm-serve.sh instead). Keep the text flags in
# sync with that personal launcher if you retune there.
#
# Text runs with CUDA graphs ON (no --enforce-eager) for faster decode. This matters most for this
# hybrid model: qwen-codex is Qwen3.5 (Mamba/linear-attention on 48 of 64 layers + full attention on
# the other 16), and the linear-attention layers are slow step-by-step in eager mode. Vision keeps
# --enforce-eager (small model, not the bottleneck, avoids any graph-capture risk in its tight slice).
set -euo pipefail

MODE="${1:-text}"

# Ensure the vLLM venv is on PATH so a bare `vllm` resolves (matches the personal launcher's interpreter).
[ -d "$HOME/.venvs/vllm/bin" ] && export PATH="$HOME/.venvs/vllm/bin:$PATH"

# Flags common to both engines. fp8 KV cache is REQUIRED to fit 32K context: the ~19.5GB 4-bit weights
# leave only ~3GB on a 24GB card, and only 16/64 layers carry a length-scaling KV cache — fp16 KV
# (~4GB for one 32K sequence) would not fit. Do not drop fp8 without lowering --max-model-len.
common=(
  --kv-cache-dtype fp8
  --host 127.0.0.1
  --api-key "${VLLM_API_KEY:?set VLLM_API_KEY in .env}"
)

case "$MODE" in
  text)
    # CUDA graphs are ON by default (fast decode for the hybrid Mamba layers — see header). But
    # torch.compile is known-buggy UPSTREAM for Qwen3.5 hybrid (GDN/Mamba) models: profile_run
    # crashes with `AttributeError: 'NoneType' ... .size` in qwen3_next.py forward — same class as
    # vllm-project/vllm#19554 (and #41862, Qwen3.5 GDN/Mamba compile issues). This is NOT a version
    # mismatch: vLLM 0.23.0 targets cu13 and torch 2.11.0+cu130 is the intended pairing. Until the
    # Qwen3.5 compile path stabilizes upstream, run eager. (For this model the Mamba/GDN/linear-attn
    # ops are in `splitting_ops` — excluded from graph capture anyway — so eager's loss is partial.)
    text_extra=()
    [ "${VLLM_TEXT_ENFORCE_EAGER:-0}" = "1" ] && text_extra+=(--enforce-eager)
    exec vllm serve "${VLLM_TEXT_MODEL:-/home/marcellus/models/qwen-codex-awq}" \
      --served-model-name qwen-codex \
      --gpu-memory-utilization "${VLLM_TEXT_GPU_UTIL:-0.93}" \
      --max-num-seqs "${VLLM_TEXT_MAX_SEQS:-4}" \
      --max-model-len "${VLLM_TEXT_MAX_MODEL_LEN:-32768}" \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --port "${VLLM_TEXT_PORT:?set VLLM_TEXT_PORT in .env}" \
      "${text_extra[@]}" \
      "${common[@]}"
    ;;
  vision)
    # Sleep mode lets the manager free vision's VRAM/KV when idle (POST /sleep) and restore it
    # (POST /wake_up); the dev endpoints require VLLM_SERVER_DEV_MODE=1.
    export VLLM_SERVER_DEV_MODE=1
    exec vllm serve "${VLLM_VISION_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}" \
      --served-model-name "${VLLM_VISION_SERVED_NAME:-qwen-vl}" \
      --enable-sleep-mode \
      --enforce-eager \
      --gpu-memory-utilization "${VLLM_VISION_GPU_UTIL:-0.30}" \
      --max-num-seqs "${VLLM_VISION_MAX_SEQS:-2}" \
      --max-model-len "${VLLM_VISION_MAX_MODEL_LEN:-16384}" \
      --limit-mm-per-prompt '{"image":2,"video":0}' \
      --port "${VLLM_VISION_PORT:?set VLLM_VISION_PORT in .env}" \
      "${common[@]}"
    ;;
  *)
    echo "usage: $0 text|vision" >&2
    exit 2
    ;;
esac
