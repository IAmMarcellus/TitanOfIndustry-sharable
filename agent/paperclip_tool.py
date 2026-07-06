"""OpenSage -> Paperclip control-plane tools (coordinate over the governance API).

The OpenSage orchestrator runs as a Paperclip ``opensage``-adapter agent. On every turn the
adapter injects a short-lived, per-run Paperclip API token + run id into the ADK session state
under ``temp:`` keys (so they live for the turn but are stripped from the persisted session
snapshot). These tools read those creds from ``tool_context.state`` and call the Paperclip HTTP
API, so the agent can post comments, update issue status, checkout/release, and delegate work —
with correct per-run attribution (``X-Paperclip-Run-Id``).

Process boundary by design: plain host functions over HTTP (like ``opencode_tool``/``memory``).
See ../CLAUDE.md. When no token is present (e.g. the JWT secret is unset on the Paperclip server,
or this agent isn't on the ``opensage`` adapter), every tool returns a graceful
``{"success": False, "error": ...}`` — the agent should skip control-plane actions and keep coding.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from google.adk.tools.tool_context import ToolContext

# State keys seeded per-run by the opensage adapter. These MUST stay in sync with the keys written
# in vendor/paperclip/server/src/adapters/opensage/execute.ts (the `paperclipStateDelta` object) —
# it's a cross-runtime (TS↔Python) contract with no shared constant, so a rename on either side
# silently breaks credential delivery.
# Plain (session-scoped) keys — NOT `temp:`-prefixed: ADK's extract_state_delta drops temp: keys
# from the session state entirely (sessions/_session_util.py), so a temp: key passed via the
# run_async state_delta never reaches tool_context.state. Plain keys land in session state and ARE
# readable here. The token is short-lived (~1h) and the adapter overwrites it every turn, so its
# presence in the persisted session snapshot is bounded.
_TOKEN_KEY = "paperclip_api_token"
_RUN_ID_KEY = "paperclip_run_id"
_BASE_URL_KEY = "paperclip_base_url"
_AGENT_ID_KEY = "paperclip_agent_id"
_COMPANY_ID_KEY = "paperclip_company_id"

_DEFAULT_BASE_URL = os.environ.get("PAPERCLIP_BASE_URL", "")
_TIMEOUT_S = float(os.environ.get("PAPERCLIP_API_TIMEOUT_S", "30"))

_NO_CREDS = {
    "success": False,
    "error": (
        "no paperclip credentials this run (JWT secret unset on the Paperclip server, or this "
        "agent is not on the opensage adapter) — skip control-plane actions and continue your work"
    ),
}


def _state_get(tool_context: ToolContext, key: str) -> str:
    """Read a string value from session state defensively (returns "" if missing/non-string)."""
    state = getattr(tool_context, "state", None)
    if state is None:
        return ""
    value = state.get(key)  # ADK State / dict .get() doesn't raise; isinstance handles odd values
    return value if isinstance(value, str) else ""


def _creds(tool_context: ToolContext) -> dict[str, str] | None:
    """Resolve the per-run Paperclip creds from session state; ``None`` when no token was injected."""
    token = _state_get(tool_context, _TOKEN_KEY)
    if not token:
        return None
    return {
        "token": token,
        "run_id": _state_get(tool_context, _RUN_ID_KEY),
        "base_url": _state_get(tool_context, _BASE_URL_KEY) or _DEFAULT_BASE_URL,
        "agent_id": _state_get(tool_context, _AGENT_ID_KEY),
        "company_id": _state_get(tool_context, _COMPANY_ID_KEY),
    }


def _compact(**fields: str) -> dict[str, Any]:
    """Build a request body from only the non-empty fields (drops empty-string / falsy values)."""
    return {key: value for key, value in fields.items() if value}


async def _request(
    tool_context: ToolContext,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    creds: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call the Paperclip API with the per-run JWT + run-id headers; normalize the result.

    ``creds`` lets a caller pass creds it already resolved (skips a second state read).
    Returns ``{success, status_code, data}`` (data is parsed JSON or raw text), or
    ``{success: False, error}`` when creds are missing or the request itself fails.
    """
    creds = creds or _creds(tool_context)
    if creds is None:
        return dict(_NO_CREDS)
    headers = {"authorization": f"Bearer {creds['token']}", "accept": "application/json"}
    if creds["run_id"]:
        headers["x-paperclip-run-id"] = creds["run_id"]
    url = creds["base_url"].rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.request(method, url, json=body, params=params, headers=headers)
    except Exception as exc:
        return {"success": False, "error": f"request failed: {exc}", "url": url}
    try:
        data: Any = resp.json()
    except Exception:
        data = resp.text
    return {"success": resp.status_code < 400, "status_code": resp.status_code, "data": data}


async def paperclip_list_assignments(tool_context: ToolContext) -> dict[str, Any]:
    """List this agent's current Paperclip assignments (the compact heartbeat inbox).

    Call this to see which issues/tasks are assigned to you before deciding what to work on.

    Returns:
        dict with ``success``/``status_code``/``data`` (the inbox-lite payload), or
        ``success=False`` with an ``error`` (no creds / API error).
    """
    return await _request(tool_context, "GET", "/api/agents/me/inbox-lite")


async def paperclip_get_issue(issue_id: str, tool_context: ToolContext) -> dict[str, Any]:
    """Fetch a Paperclip issue/task (status, description, assignee, blockers, ancestors).

    Args:
        issue_id: The issue id to read.
    """
    return await _request(tool_context, "GET", f"/api/issues/{issue_id}")


async def paperclip_post_comment(
    issue_id: str, body: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Post a comment on a Paperclip issue (progress notes, questions, decisions).

    Args:
        issue_id: The issue to comment on.
        body: The comment text (markdown; real newlines are preserved).
    """
    return await _request(
        tool_context, "POST", f"/api/issues/{issue_id}/comments", body={"body": body}
    )


async def paperclip_update_issue(
    issue_id: str,
    tool_context: ToolContext,
    status: str = "",
    priority: str = "",
    assignee_agent_id: str = "",
    comment: str = "",
) -> dict[str, Any]:
    """Update a Paperclip issue: set status/priority/assignee and optionally attach a comment.

    You MUST ``paperclip_checkout`` an issue before changing the status of work you own (Paperclip
    ties the mutation to your current run id).

    Args:
        issue_id: The issue to update.
        status: New status — backlog|todo|in_progress|in_review|done|blocked|cancelled. Empty = unchanged.
        priority: New priority — critical|high|medium|low. Empty = unchanged.
        assignee_agent_id: Reassign to this agent id (delegation/handoff). Empty = unchanged.
        comment: Optional comment to post with the update.
    """
    payload = _compact(
        status=status, priority=priority, assigneeAgentId=assignee_agent_id, comment=comment
    )
    if not payload:
        return {
            "success": False,
            "error": "nothing to update (set status, priority, assignee_agent_id, and/or comment)",
        }
    return await _request(tool_context, "PATCH", f"/api/issues/{issue_id}", body=payload)


async def paperclip_checkout(
    issue_id: str, tool_context: ToolContext, expected_statuses: str = ""
) -> dict[str, Any]:
    """Check out a Paperclip issue (claim it / move it to in_progress) before working on it.

    Required before mutating an in_progress issue. A 409 means another agent owns it — stop and
    pick a different task; never retry a 409.

    Args:
        issue_id: The issue to check out.
        expected_statuses: Optional comma-separated statuses the issue must currently be in,
            e.g. "todo,blocked,in_review". Empty uses a sensible default set.
    """
    creds = _creds(tool_context)
    if creds is None:
        return dict(_NO_CREDS)
    statuses = [s.strip() for s in expected_statuses.split(",") if s.strip()] or [
        "todo",
        "backlog",
        "blocked",
        "in_review",
    ]
    return await _request(
        tool_context,
        "POST",
        f"/api/issues/{issue_id}/checkout",
        body={"agentId": creds["agent_id"], "expectedStatuses": statuses},
        creds=creds,
    )


async def paperclip_release(issue_id: str, tool_context: ToolContext) -> dict[str, Any]:
    """Release a previously checked-out issue (give up the lock without finishing it).

    Args:
        issue_id: The issue to release.
    """
    return await _request(tool_context, "POST", f"/api/issues/{issue_id}/release")


async def paperclip_create_subtask(
    title: str,
    description: str,
    tool_context: ToolContext,
    parent_id: str = "",
    goal_id: str = "",
    assignee_agent_id: str = "",
    priority: str = "",
) -> dict[str, Any]:
    """Create a child issue to delegate work to another agent (or future you).

    Prefer setting ``parent_id`` (and ``goal_id`` when known) so the subtask links into the tree.

    Args:
        title: Short imperative title.
        description: What needs to be done (markdown).
        parent_id: The parent issue id this subtask belongs under (recommended).
        goal_id: The goal id to attach (recommended when known).
        assignee_agent_id: Agent id to assign the subtask to (delegation). Empty leaves it unassigned.
        priority: critical|high|medium|low. Empty uses the company default.
    """
    creds = _creds(tool_context)
    if creds is None:
        return dict(_NO_CREDS)
    payload = _compact(
        title=title,
        description=description,
        parentId=parent_id,
        goalId=goal_id,
        assigneeAgentId=assignee_agent_id,
        priority=priority,
    )
    return await _request(
        tool_context,
        "POST",
        f"/api/companies/{creds['company_id']}/issues",
        body=payload,
        creds=creds,
    )
