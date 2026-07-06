"""Single Python source for the ADK session-state keys the Paperclip opensage adapter seeds each turn.

These key names are a cross-runtime contract with
``vendor/paperclip/server/src/adapters/opensage/execute.ts`` (the ``paperclipStateDelta`` object) — there
is no shared constant across the TS↔Python boundary, so a rename on either side silently breaks delivery.
Keeping the Python side in one stdlib-only module (no httpx / google.adk pulled in) means a rename touches
exactly one place here. Read with ``state_get`` (defensive: ``""`` for missing / non-string / no state).

The keys are NOT ``temp:``-prefixed: ADK's ``extract_state_delta`` drops ``temp:`` keys before they reach
a tool, so these land in session state and are readable. Used by ``memory.py`` (tenant scope);
``paperclip_tool.py`` may adopt these to retire its local copies.
"""

from __future__ import annotations

from typing import Any

COMPANY_ID_KEY = "paperclip_company_id"
AGENT_ID_KEY = "paperclip_agent_id"


def state_get(tool_context: Any, key: str) -> str:
    """Read a string from ADK session state defensively (``""`` if missing / not a string / no state)."""
    state = getattr(tool_context, "state", None)
    if state is None:
        return ""
    try:
        value = state.get(key)
    except Exception:
        return ""
    return value if isinstance(value, str) else ""
