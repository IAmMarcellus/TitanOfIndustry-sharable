#!/usr/bin/env bash
# Health-check the Ollama-served Qwen model used by TitanOfIndustry (OpenAI-compatible API).
# This pings the endpoint configured in the private local environment.
set -euo pipefail

BASE="${OPENAI_BASE_URL:?set OPENAI_BASE_URL in .env}"
MODEL="${OLLAMA_MODEL:-huihui_ai/Qwen3.6-abliterated:27b}"

echo "endpoint: $BASE"
echo "model:    $MODEL"
echo "--- /v1/models ---"
curl -sf --max-time 15 "$BASE/models" | jq -r '.data[].id'
echo "--- chat ping ---"
curl -sf --max-time 90 "$BASE/chat/completions" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "$MODEL" \
        '{model:$m,messages:[{role:"user",content:"Reply with exactly: pong"}],stream:false,max_tokens:2048}')" \
  | jq -r '.choices[0].message.content'
