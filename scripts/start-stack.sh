#!/usr/bin/env bash
# Bring up the whole TitanOfIndustry stack with one command, in dependency order.
#
# Each long-lived service runs in its own tmux window (so you keep live logs and can
# Ctrl-C one without killing the rest); the script waits for each layer to be healthy
# before starting the next. Backends first (neo4j -> model-stack -> proxy), then the
# agents (opencode -> opensage -> paperclip). Mirrors the Makefile targets + start order.
#
#   scripts/start-stack.sh                # bring the stack up (Ollama backend — matches live config)
#   scripts/start-stack.sh --with-vllm      # use the WSL vLLM dual-model engines instead of Ollama
#   scripts/start-stack.sh --with-opencode    # also start the OpenCode headless server
#   scripts/start-stack.sh --no-memory-mcp    # skip the shared-memory MCP server
#   scripts/start-stack.sh --with-voice       # also start the Pipecat voice sidecar
#   scripts/start-stack.sh --resume last     # resume the most recent OpenSage thread
#   scripts/start-stack.sh --resume <id>     # resume a specific OpenSage thread
#   scripts/start-stack.sh --attach          # attach to the tmux session when ready
#   scripts/start-stack.sh --down            # tear the whole stack down
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="titanofindustry"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

need_env() {
  local name="$1" value="${!1:-}"
  if [ -z "$value" ]; then
    echo "Missing required local setting: $name (set it in .env)" >&2
    exit 1
  fi
  printf '%s' "$value"
}

WITH_OPENCODE=0
WITH_VLLM=0
WITH_MEMORY_MCP=1
WITH_VOICE=0
ATTACH=0
RESUME=""
DOWN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --with-opencode) WITH_OPENCODE=1 ;;
    --with-vllm)     WITH_VLLM=1 ;;
    --with-memory-mcp) WITH_MEMORY_MCP=1 ;;
    --no-memory-mcp)   WITH_MEMORY_MCP=0 ;;
    --with-voice)    WITH_VOICE=1 ;;
    --attach)        ATTACH=1 ;;
    --resume)        RESUME="${2:-}"; shift ;;
    --down)          DOWN=1 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# ---- teardown -------------------------------------------------------------
if [ "$DOWN" = "1" ]; then
  echo "Stopping TitanOfIndustry stack..."
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "  killed tmux session '$SESSION'" || echo "  no tmux session '$SESSION'"
  ( cd "$REPO_ROOT" && make neo4j-stop ) || true
  echo "Done. (GPU engines spawned by model-stack exit with its window.)"
  exit 0
fi

command -v tmux >/dev/null || { echo "tmux is required (sudo apt install tmux)" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A '$SESSION' tmux session already exists." >&2
  echo "  attach:   tmux attach -t $SESSION" >&2
  echo "  tear down: scripts/start-stack.sh --down" >&2
  exit 1
fi

# ---- health helpers -------------------------------------------------------
# Dump the last lines of a window's pane so a stalled service explains itself.
dump_window() {
  local win="$1"
  echo "    --- last output of '$win' (tmux window) ---" >&2
  tmux capture-pane -p -t "${SESSION}:${win}" 2>/dev/null | tail -n 30 | sed 's/^/    | /' >&2 || true
}

wait_http() {  # url name timeout_s [window]
  local url="$1" name="$2" timeout="${3:-120}" win="${4:-}" waited=0
  printf '  waiting for %s (%s) ' "$name" "$url"
  until curl -fsS --max-time 4 "$url" >/dev/null 2>&1; do
    sleep 2; waited=$((waited + 2)); printf '.'
    if [ "$waited" -ge "$timeout" ]; then
      printf ' TIMEOUT after %ss\n' "$timeout"
      [ -n "$win" ] && dump_window "$win"
      echo "  (session left running so you can inspect: tmux attach -t $SESSION)" >&2
      exit 1
    fi
  done
  printf ' up (%ss)\n' "$waited"
}

wait_port() {  # port name timeout_s [window]
  local port="$1" name="$2" timeout="${3:-120}" win="${4:-}" waited=0
  printf '  waiting for %s ' "$name"
  until (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; do
    sleep 2; waited=$((waited + 2)); printf '.'
    if [ "$waited" -ge "$timeout" ]; then
      printf ' TIMEOUT after %ss\n' "$timeout"
      [ -n "$win" ] && dump_window "$win"
      echo "  (session left running so you can inspect: tmux attach -t $SESSION)" >&2
      exit 1
    fi
  done
  exec 3>&- 3<&- 2>/dev/null || true
  printf ' up (%ss)\n' "$waited"
}

# Launch `make <target>` in a new tmux window; keep the window alive after exit so
# a crashed service shows its error instead of vanishing.
run_window() {  # window_name  make_args...
  local win="$1"; shift
  local cmd="cd '$REPO_ROOT' && make $* ; rc=\$?; echo; echo \"[$win exited rc=\$rc]\"; exec \$SHELL"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n "$win" "$cmd"
  else
    tmux new-session -d -s "$SESSION" -n "$win" "$cmd"
  fi
}

echo "==> Bringing up TitanOfIndustry ($SESSION) — dual-model path"

# 1) Backends ---------------------------------------------------------------
echo "==> neo4j (shared memory)"
( cd "$REPO_ROOT" && make neo4j )
wait_port "$(need_env NEO4J_BOLT_PORT)" "neo4j bolt" 90

# Shared-memory MCP (on by default; --no-memory-mcp to skip): only needs Neo4j, so start it right after.
# Lets non-OpenSage agents (Claude Code / OpenCode / Paperclip-driven) reach the same brain over MCP.
if [ "$WITH_MEMORY_MCP" = "1" ]; then
  echo "==> memory-mcp (shared-memory MCP for non-OpenSage agents)"
  run_window memory-mcp memory-mcp
  # generous gate: a cold `uv run --with mcp ...` resolves deps before uvicorn binds on first run
  wait_port "$(need_env MEMORY_MCP_PORT)" "memory-mcp" 120 memory-mcp
fi

# Model backend. DEFAULT = Ollama: the proxy routes qwen-codex/qwen-vl to OLLAMA_BASE_URL (Ollama on
# the Windows host), so the WSL vLLM engines are NOT needed and would only contend for the 3090.
# Pass --with-vllm to run the WSL vLLM dual-model stack instead (and repoint the proxy at it).
if [ "$WITH_VLLM" = "1" ]; then
  echo "==> model-stack (WSL vLLM text+vision engines; loads qwen-codex into VRAM)"
  # The manager only serves /healthz AFTER the text engine is healthy (its lifespan awaits engine
  # startup, up to MODEL_MANAGER_ENGINE_START_TIMEOUT_S=300), so allow a generous cold-load window.
  # /health is unauthenticated; /v1/models would 401 under --api-key and never pass.
  run_window model-stack model-stack
  wait_http "${VLLM_TEXT_HEALTH_URL:-$(need_env VLLM_TEXT_BASE_URL | sed 's#/v1$##')/health}" "qwen-codex text engine" 360 model-stack
  wait_http "${MODEL_MANAGER_HEALTH_URL:-$(need_env MODEL_MANAGER_URL | sed 's#/$##')/healthz}" "model-manager" 60 model-stack
else
  echo "==> model backend: Ollama (proxy -> OLLAMA_BASE_URL); skipping WSL vLLM. Use --with-vllm to switch."
  _ollama="$(need_env OLLAMA_BASE_URL)"
  if curl -fsS --max-time 4 "${_ollama%/v1}/api/tags" >/dev/null 2>&1; then
    echo "  Ollama reachable at ${_ollama%/v1}"
  else
    echo "  WARNING: Ollama not reachable at ${_ollama%/v1} — start it on Windows, or rerun with --with-vllm." >&2
  fi
fi

echo "==> proxy (LiteLLM gateway)"
run_window proxy proxy
wait_port "$(need_env PROXY_PORT)" "litellm proxy" 90 proxy

# 2) Agents -----------------------------------------------------------------
if [ "$WITH_OPENCODE" = "1" ]; then
  echo "==> opencode (headless executor)"
  run_window opencode opencode
  wait_port "$(need_env OPENCODE_PORT)" "opencode" 60 opencode
fi

echo "==> opensage (orchestrator)"
if [ -n "$RESUME" ]; then run_window opensage opensage "RESUME=$RESUME"; else run_window opensage opensage; fi
wait_http "${OPENSAGE_BASE_URL:-http://localhost:$(need_env OPENSAGE_PORT)}/list-apps" "opensage" 150 opensage

echo "==> paperclip (governance plane)"
run_window paperclip paperclip
wait_port "$(need_env PAPERCLIP_PORT)" "paperclip" 300 paperclip

# Pipecat voice sidecar (opt-in): needs the proxy (qwen-voice brain) + paperclip (tools/digest/persist),
# so it starts last. First-ever start downloads whisper + kokoro weights — generous timeout.
if [ "$WITH_VOICE" = "1" ]; then
  echo "==> voice (Pipecat: whisper-cpu -> qwen-voice -> kokoro-cpu)"
  run_window voice voice
  wait_http "${VOICE_PIPECAT_URL:-http://localhost:$(need_env VOICE_PIPECAT_PORT)}/health" "pipecat voice" 240 voice
fi

# ---- ready ----------------------------------------------------------------
cat <<EOF

==> TitanOfIndustry is UP.
    Local service URLs are determined by your private .env file.

    Live logs : tmux attach -t $SESSION   (switch windows: Ctrl-b n / Ctrl-b <number>)
    Tear down : scripts/start-stack.sh --down
EOF

if [ "$ATTACH" = "1" ]; then exec tmux attach -t "$SESSION"; fi
