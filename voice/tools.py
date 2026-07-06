"""Read-only Paperclip tools for the voice brain (v1: no write actions).

Every handler slims its response hard before returning it to the model: tool results land in the
next prompt, and prompt size is the dominant latency lever for the 27B on Ollama (a fat JSON dump
can add multi-second prefill). Keep fields few, lists capped, and strings clipped.
"""

import os
from typing import Any

import httpx
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

import paperclip

MAX_LIST_ITEMS = 15
MAX_TEXT = 200

# Optional pre-rendered report endpoint. Private hosting details are intentionally redacted.
PAPER_REPORT_URL = os.environ.get("VOICE_PAPER_REPORT_URL", "")
PAPER_REPORT_MAX = 4000


def _clip(value: Any, max_len: int = MAX_TEXT) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[: max_len - 1].rstrip() + "…"
    return value


def _pick(obj: Any, fields: list[str]) -> dict:
    if not isinstance(obj, dict):
        return {}
    return {f: _clip(obj[f]) for f in fields if f in obj and obj[f] is not None}


def _pick_list(items: Any, fields: list[str]) -> list[dict]:
    if not isinstance(items, list):
        return []
    slimmed = [_pick(i, fields) for i in items[:MAX_LIST_ITEMS]]
    if len(items) > MAX_LIST_ITEMS:
        slimmed.append({"note": f"+{len(items) - MAX_LIST_ITEMS} more not shown"})
    return slimmed


def _unwrap(data: Any, *keys: str) -> Any:
    """Paperclip wraps most list endpoints ({companies: [...]}, {issues: [...]}); unwrap if so."""
    if isinstance(data, dict):
        for k in keys:
            if k in data:
                return data[k]
    return data


def _result(tool: str, data: Any, slimmed: Any, fallback: str) -> Any:
    """Slimmed result or fallback — warns when a NON-EMPTY 200 came back but slimming found nothing,
    so a silently-changed response envelope is observable instead of the tools just going blind.
    (An empty list/dict is a legitimate "none found", not a shape mismatch.)"""
    if slimmed:
        return slimmed
    if data:
        logger.warning(f"{tool}: unexpected response shape — got data but nothing survived slimming")
    return fallback


COMPANY_FIELDS = ["id", "name", "issuePrefix", "status", "description"]
AGENT_FIELDS = ["id", "name", "role", "status", "title", "reportsTo"]
ISSUE_FIELDS = ["id", "identifier", "title", "status", "priority", "assigneeAgentId", "updatedAt"]


async def _list_companies(params: FunctionCallParams):
    data = await paperclip.get_json("/companies")
    slim = _pick_list(_unwrap(data, "companies"), COMPANY_FIELDS)
    await params.result_callback(_result("list_companies", data, slim, "unavailable"))


async def _get_company(params: FunctionCallParams):
    data = await paperclip.get_json(f"/companies/{params.arguments.get('company_id', '')}")
    slim = _pick(_unwrap(data, "company"), COMPANY_FIELDS + ["createdAt"])
    await params.result_callback(_result("get_company", data, slim, "not found"))


async def _company_dashboard(params: FunctionCallParams):
    data = await paperclip.get_json(f"/companies/{params.arguments.get('company_id', '')}/dashboard")
    # The dashboard summary is already an aggregate — pass its top-level sections through, clipped.
    slim = (
        {k: data[k] for k in ("tasks", "agents", "costs", "pendingApprovals", "budgets") if k in data}
        if isinstance(data, dict)
        else None
    )
    await params.result_callback(_result("company_dashboard", data, slim or data, "unavailable"))


async def _list_agents(params: FunctionCallParams):
    data = await paperclip.get_json(f"/companies/{params.arguments.get('company_id', '')}/agents")
    slim = _pick_list(_unwrap(data, "agents"), AGENT_FIELDS)
    await params.result_callback(
        _result("list_agents", data, slim, "none found — check company_id is the uuid from list_companies")
    )


async def _get_agent(params: FunctionCallParams):
    data = await paperclip.get_json(f"/agents/{params.arguments.get('agent_id', '')}")
    slim = _pick(_unwrap(data, "agent"), AGENT_FIELDS + ["heartbeatEnabled", "lastRunAt"])
    await params.result_callback(_result("get_agent", data, slim, "not found"))


async def _list_issues(params: FunctionCallParams):
    query: dict = {}
    if params.arguments.get("status"):
        query["status"] = params.arguments["status"]
    if params.arguments.get("search"):
        query["q"] = params.arguments["search"]
    data = await paperclip.get_json(f"/companies/{params.arguments.get('company_id', '')}/issues", params=query)
    slim = _pick_list(_unwrap(data, "issues"), ISSUE_FIELDS)
    await params.result_callback(_result("list_issues", data, slim, "none found"))


async def _get_issue(params: FunctionCallParams):
    data = await paperclip.get_json(f"/issues/{params.arguments.get('issue_id', '')}")
    issue = _pick(_unwrap(data, "issue"), ISSUE_FIELDS + ["description"])
    if isinstance(data, dict) and isinstance(data.get("comments"), list):
        issue["recent_comments"] = _pick_list(data["comments"][-5:], ["authorAgentId", "body", "createdAt"])
    await params.result_callback(_result("get_issue", data, issue, "not found"))


async def _costs_summary(params: FunctionCallParams):
    data = await paperclip.get_json(f"/companies/{params.arguments.get('company_id', '')}/costs/summary")
    await params.result_callback(data if isinstance(data, dict) else "unavailable")


# ── Write tools ────────────────────────────────────────────────────────────────────────────────────
# Voice writes are confirmation-gated IN THE PROMPT (the model must say what it's about to do and
# hear a yes first) because ASR mishears — but the handlers still validate hard and, unlike the read
# tools, report failures explicitly so the operator hears "that didn't land" instead of silence.
# Approvals are deliberately NOT exposed: they gate agent actions and spending, and a misheard "yes"
# on a live trading company is not a voice-sized risk. Use the dashboard for approvals.


async def _create_issue(params: FunctionCallParams):
    company_id = str(params.arguments.get("company_id", "")).strip()
    title = str(params.arguments.get("title", "")).strip()
    if not company_id or not title:
        await params.result_callback("rejected — company_id and a title are both required")
        return
    body: dict = {"title": title}
    if params.arguments.get("description"):
        body["description"] = str(params.arguments["description"])
    if params.arguments.get("priority") in ("critical", "high", "medium", "low"):
        body["priority"] = params.arguments["priority"]
    data, err = await paperclip.send_json("POST", f"/companies/{company_id}/issues", body)
    if err:
        await params.result_callback(f"write failed — {err}")
        return
    slim = _pick(_unwrap(data, "issue"), ["id", "identifier", "title", "status", "priority"])
    await params.result_callback({"created": slim or True})


async def _add_comment(params: FunctionCallParams):
    issue_id = str(params.arguments.get("issue_id", "")).strip()
    comment = str(params.arguments.get("body", "")).strip()
    if not issue_id or not comment:
        await params.result_callback("rejected — issue_id and a comment body are both required")
        return
    data, err = await paperclip.send_json("POST", f"/issues/{issue_id}/comments", {"body": comment})
    if err:
        await params.result_callback(f"write failed — {err}")
        return
    await params.result_callback({"commented": True})


async def _update_issue_status(params: FunctionCallParams):
    issue_id = str(params.arguments.get("issue_id", "")).strip()
    status = str(params.arguments.get("status", "")).strip()
    if not issue_id or status not in ("backlog", "todo", "in_progress", "blocked", "done", "cancelled"):
        await params.result_callback("rejected — issue_id and a valid status are required")
        return
    body: dict = {"status": status}
    if params.arguments.get("comment"):
        body["comment"] = str(params.arguments["comment"])
    data, err = await paperclip.send_json("PATCH", f"/issues/{issue_id}", body)
    if err:
        await params.result_callback(f"write failed — {err}")
        return
    slim = _pick(_unwrap(data, "issue"), ["id", "identifier", "title", "status"])
    await params.result_callback({"updated": slim or True})


async def _paper_trading_report(params: FunctionCallParams):
    if not PAPER_REPORT_URL:
        await params.result_callback("unavailable — the paper-track report feed is not configured")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            resp = await client.get(PAPER_REPORT_URL)
            resp.raise_for_status()
            text = resp.text.strip()
    except Exception as exc:
        logger.warning(f"paper_trading_report: fetch failed: {exc}")
        await params.result_callback("unavailable — the paper-track report feed is not reachable right now")
        return
    await params.result_callback(_clip(text, PAPER_REPORT_MAX) if text else "unavailable")


_COMPANY_ID_PROP = {
    "company_id": {
        "type": "string",
        "description": "Company id — the uuid from list_companies. NOT the name or issue prefix like BET.",
    }
}

# Single source of truth: schema + handler side by side, so a tool can't exist in one table and
# not the other. TOOL_HANDLERS / PAPERCLIP_TOOLS are derived below.
_TOOLS: list[tuple[FunctionSchema, Any]] = [
    (
        FunctionSchema("list_companies", "List every company with ids and statuses.", {}, []),
        _list_companies,
    ),
    (
        FunctionSchema("get_company", "Get one company's details.", dict(_COMPANY_ID_PROP), ["company_id"]),
        _get_company,
    ),
    (
        FunctionSchema(
            "company_dashboard",
            "Get one company's dashboard rollup: task counts, agent states, spend, approvals.",
            dict(_COMPANY_ID_PROP),
            ["company_id"],
        ),
        _company_dashboard,
    ),
    (
        FunctionSchema("list_agents", "List a company's agents.", dict(_COMPANY_ID_PROP), ["company_id"]),
        _list_agents,
    ),
    (
        FunctionSchema(
            "get_agent",
            "Get one agent's details.",
            {"agent_id": {"type": "string", "description": "Agent id"}},
            ["agent_id"],
        ),
        _get_agent,
    ),
    (
        FunctionSchema(
            "list_issues",
            "List a company's issues, optionally filtered.",
            {
                **_COMPANY_ID_PROP,
                "status": {
                    "type": "string",
                    "enum": ["backlog", "todo", "in_progress", "blocked", "done", "cancelled"],
                    "description": "Only issues in this status",
                },
                "search": {"type": "string", "description": "Text search over titles"},
            },
            ["company_id"],
        ),
        _list_issues,
    ),
    (
        FunctionSchema(
            "get_issue",
            "Get one issue's full details including recent comments.",
            {"issue_id": {"type": "string", "description": "Issue id (the uuid from list_issues, not the CEL-12 identifier)"}},
            ["issue_id"],
        ),
        _get_issue,
    ),
    (
        FunctionSchema(
            "costs_summary", "Get a company's cost/spend summary.", dict(_COMPANY_ID_PROP), ["company_id"]
        ),
        _costs_summary,
    ),
    (
        FunctionSchema(
            "create_issue",
            "Create a new issue/task in a company. ONLY call after the operator has confirmed the "
            "exact title aloud.",
            {
                **_COMPANY_ID_PROP,
                "title": {"type": "string", "description": "Issue title, short and imperative"},
                "description": {"type": "string", "description": "Optional details/context"},
                "priority": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                    "description": "Optional priority",
                },
            },
            ["company_id", "title"],
        ),
        _create_issue,
    ),
    (
        FunctionSchema(
            "add_comment",
            "Add a comment to an issue thread. ONLY call after the operator has confirmed the "
            "message aloud.",
            {
                "issue_id": {"type": "string", "description": "Issue id (uuid from list_issues)"},
                "body": {"type": "string", "description": "The comment text"},
            },
            ["issue_id", "body"],
        ),
        _add_comment,
    ),
    (
        FunctionSchema(
            "update_issue_status",
            "Change an issue's status (reprioritize/close/reopen). ONLY call after the operator has "
            "confirmed aloud.",
            {
                "issue_id": {"type": "string", "description": "Issue id (uuid from list_issues)"},
                "status": {
                    "type": "string",
                    "enum": ["backlog", "todo", "in_progress", "blocked", "done", "cancelled"],
                    "description": "New status",
                },
                "comment": {"type": "string", "description": "Optional note explaining the change"},
            },
            ["issue_id", "status"],
        ),
        _update_issue_status,
    ),
    (
        FunctionSchema(
            "paper_trading_report",
            "Get <REDACTED_COMPANY>'s live forward paper-trading report from the <REDACTED_HOST> machine: every dry-run "
            "track (bml3 cash baseline, momentum-neutral, trend-following, rotation) with per-sleeve "
            "Sharpe/CAGR/drawdown and blend performance. Use for ANY question about paper trading, "
            "trading tracks, or strategy performance.",
            {},
            [],
        ),
        _paper_trading_report,
    ),
]

TOOL_HANDLERS = {schema.name: handler for schema, handler in _TOOLS}
PAPERCLIP_TOOLS = ToolsSchema(standard_tools=[schema for schema, _ in _TOOLS])
