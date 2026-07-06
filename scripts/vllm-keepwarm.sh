#!/usr/bin/env bash
# Keep the vLLM text engine resident in VRAM (WSL2 anti-paging).
#
# On WSL2/WDDM the Windows GPU memory manager pages an idle engine's weights out to host RAM; the next
# request then faults ~19.5GB back across PCIe (observed: cold ~4.4s vs warm ~2s for 8 tokens, and VRAM
# collapsing 22958MiB -> 3629MiB while idle). A tiny periodic 1-token completion keeps the model hot so
# interactive sessions never pay that cold-start. Negligible load (one short request per interval).
#
# Run in the background, e.g.:  make vllm-keepwarm &   (or via the Makefile target).
# Stop it with:  pkill -f vllm-keepwarm.sh
set -uo pipefail

BASE="${VLLM_TEXT_BASE_URL:?set VLLM_TEXT_BASE_URL in .env}"
KEY="${VLLM_API_KEY:?set VLLM_API_KEY in .env}"
MODEL="${VLLM_TEXT_SERVED_NAME:-qwen-codex}"
INTERVAL="${KEEPWARM_INTERVAL_S:-45}"

echo "[keepwarm] pinging ${BASE} (model=${MODEL}) every ${INTERVAL}s to keep weights resident"
while true; do
  curl -sf -o /dev/null --max-time 30 \
    -H "Authorization: Bearer ${KEY}" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" \
    "${BASE}/chat/completions" || true
  sleep "${INTERVAL}"
done
