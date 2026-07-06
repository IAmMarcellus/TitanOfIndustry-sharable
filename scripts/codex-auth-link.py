#!/usr/bin/env python3
"""One sweep of the codex_local auth auto-linker.

Paperclip gives every `codex_local` agent an *isolated* CODEX_HOME
(`.../instances/<id>/companies/<cid>/agents/<aid>/codex-home`) and blanks
OPENAI_API_KEY, so the agent authenticates from a `codex-home/auth.json`.
A freshly provisioned home has none -> the agent 401s on first run. Sharing a
whole CODEX_HOME is forbidden by Paperclip (assertCodexLocalHomeIsNotShared),
so we share only the *credential file*: symlink `auth.json` -> the shared login.

This sweep makes that idempotent and authoritative: it pre-seeds the symlink for
every codex_local agent (incl. ones a CEO hired autonomously) and clears any 401
those agents are stuck on. `scripts/codex-auth-link.sh` runs it on a loop.

Stdlib only (no uv / third-party deps). Config via env:
  CODEX_SHARED_AUTH_JSON  shared codex auth.json   (default: $HOME/.codex/auth.json)
  PAPERCLIP_HOME          paperclip home dir        (default: $HOME/.paperclip)
  PAPERCLIP_INSTANCE_ID   instance id               (default: default)
  PAPERCLIP_BASE_URL      paperclip API base
"""
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
SHARED_AUTH = Path(os.environ.get("CODEX_SHARED_AUTH_JSON") or HOME / ".codex" / "auth.json")
PAPERCLIP_HOME = Path(os.environ.get("PAPERCLIP_HOME") or HOME / ".paperclip")
INSTANCE = os.environ.get("PAPERCLIP_INSTANCE_ID") or "default"
BASE = (os.environ.get("PAPERCLIP_BASE_URL") or "").rstrip("/")
INSTANCE_ROOT = PAPERCLIP_HOME / "instances" / INSTANCE
AUTH_ERR_MARKERS = ("401", "Unauthorized", "Missing bearer")


def log(msg: str) -> None:
    print(f"[codex-auth-link {datetime.now():%H:%M:%S}] {msg}", flush=True)


def ensure_link(home: Path) -> bool:
    """Ensure <home>/auth.json -> SHARED_AUTH. Return True if it created/fixed it."""
    auth = home / "auth.json"
    try:
        if (
            auth.is_symlink()
            and os.path.realpath(auth) == os.path.realpath(SHARED_AUTH)
            and os.access(auth, os.R_OK)
        ):
            return False
        home.mkdir(parents=True, exist_ok=True)
        if auth.is_symlink() or auth.exists():
            try:
                auth.unlink()
            except OSError:
                pass
        auth.symlink_to(SHARED_AUTH)
        return True
    except OSError as e:
        log(f"WARN could not link {auth}: {e}")
        return False


def api_get(path: str):
    with urllib.request.urlopen(f"{BASE}/api{path}", timeout=8) as r:
        return json.load(r)


def api_post(path: str) -> int:
    req = urllib.request.Request(
        f"{BASE}/api{path}", data=b"{}", method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        return r.status


def main() -> int:
    if not SHARED_AUTH.exists():
        log(f"FATAL shared auth not found: {SHARED_AUTH} (run `codex login` once, or set CODEX_SHARED_AUTH_JSON)")
        return 0  # exit 0 so the loop keeps trying — the file may appear later

    # home path -> agent record (None if discovered only on the filesystem)
    homes: dict[str, dict | None] = {}

    # Pass 1 (API, authoritative): every codex_local agent's declared CODEX_HOME.
    # Pre-seeds the symlink *before* codex's first run creates the home, and lets us clear 401s.
    api_ok = False
    try:
        for c in api_get("/companies"):
            for a in api_get(f"/companies/{c['id']}/agents"):
                if a.get("adapterType") != "codex_local":
                    continue
                env = (a.get("adapterConfig") or {}).get("env") or {}
                ch = env.get("CODEX_HOME")
                chv = ch.get("value") if isinstance(ch, dict) else ch
                if chv:
                    homes[str(Path(chv))] = a
        api_ok = True
    except (urllib.error.URLError, OSError, ValueError) as e:
        log(f"API unavailable ({e}); filesystem-only pass")

    # Pass 2 (filesystem fallback): codex-homes already on disk, so we still link when the API is down.
    if INSTANCE_ROOT.exists():
        for home in INSTANCE_ROOT.glob("companies/*/agents/*/codex-home"):
            homes.setdefault(str(home), None)

    linked, cleared = [], []
    for hp, a in homes.items():
        if ensure_link(Path(hp)):
            linked.append(hp)
        if a:
            er = a.get("errorReason") or ""
            if (a.get("status") == "error" or er) and any(m in er for m in AUTH_ERR_MARKERS):
                try:
                    api_post(f"/agents/{a['id']}/clear-error")
                    cleared.append(a.get("name") or a["id"])
                except urllib.error.HTTPError as e:
                    # 409 = agent busy/mid-transition; benign, the next sweep retries when idle
                    if e.code != 409:
                        log(f"WARN clear-error failed for {a.get('name')}: {e}")
                except (urllib.error.URLError, OSError) as e:
                    log(f"WARN clear-error failed for {a.get('name')}: {e}")

    if linked or cleared:
        detail = f" [{', '.join(cleared)}]" if cleared else ""
        log(f"linked {len(linked)} home(s); cleared {len(cleared)} 401 error(s){detail}"
            f"  (scanned {len(homes)}{'' if api_ok else ', API down'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
