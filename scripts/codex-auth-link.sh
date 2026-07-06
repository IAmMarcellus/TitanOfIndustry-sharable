#!/usr/bin/env bash
# codex_local auth auto-linker (loop).
#
# Symlinks the shared codex `auth.json` into every codex_local agent's isolated
# CODEX_HOME so new agents — INCLUDING ones a CEO hires autonomously — never 401
# on first run, and clears any 401 they got stuck on. Runs one sweep, then waits
# for a filesystem change under the instance's companies/ tree (inotify) or a
# poll interval, and repeats. Started as a tmux window by start-stack.sh.
#
#   scripts/codex-auth-link.sh           # watch loop (default)
#   scripts/codex-auth-link.sh --once     # single sweep, then exit
#
# Config (env):
#   CODEX_SHARED_AUTH_JSON   shared auth.json   (default: $HOME/.codex/auth.json)
#   PAPERCLIP_HOME           paperclip home     (default: $HOME/.paperclip)
#   PAPERCLIP_INSTANCE_ID    instance id        (default: default)
#   PAPERCLIP_BASE_URL       paperclip API base URL
#   CODEX_AUTH_LINK_POLL_SEC poll/timeout secs  (default: 60)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
SWEEP="$REPO_ROOT/scripts/codex-auth-link.py"
POLL="${CODEX_AUTH_LINK_POLL_SEC:-60}"
PAPERCLIP_HOME="${PAPERCLIP_HOME:-$HOME/.paperclip}"
INSTANCE="${PAPERCLIP_INSTANCE_ID:-default}"
WATCH="$PAPERCLIP_HOME/instances/$INSTANCE/companies"

sweep() { "$PY" "$SWEEP" || true; }

if [ "${1:-}" = "--once" ]; then sweep; exit 0; fi

echo "[codex-auth-link] watching $WATCH (poll ${POLL}s); shared auth: ${CODEX_SHARED_AUTH_JSON:-$HOME/.codex/auth.json}"
mkdir -p "$WATCH" 2>/dev/null || true

have_inotify=0
command -v inotifywait >/dev/null 2>&1 && have_inotify=1
[ "$have_inotify" = 1 ] || \
  echo "[codex-auth-link] inotifywait not found — polling every ${POLL}s (apt install inotify-tools for instant linking)"

while true; do
  sweep
  if [ "$have_inotify" = 1 ]; then
    # returns on the first create/move/attrib under the tree, or after POLL seconds
    inotifywait -r -q -e create -e moved_to -e attrib --timeout "$POLL" "$WATCH" >/dev/null 2>&1 || true
  else
    sleep "$POLL"
  fi
done
