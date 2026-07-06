"""MCP server exposing TitanOfIndustry shared memory (``remember``/``recall``) to NON-OpenSage agents.

OpenSage reaches the shared Neo4j "brain" by importing ``agent.memory`` in-process (auto-scoping to its
Paperclip company/agent). Every other agent — Claude Code (incl. Paperclip-driven runtimes) and the
OpenCode executor — reaches it through this thin MCP wrapper instead. Purely additive.

**Per-token company binding (tenant isolation):** each bearer token maps to a company via
``MEMORY_MCP_TOKENS`` (JSON ``{token: company}``); that token's reads/writes are scoped to its company
(∪ the shared ``global`` tier) and cannot see other companies. Unknown tokens are rejected (401). The
resolved company is carried per-request in a contextvar (set by a pure-ASGI middleware, which — unlike
``BaseHTTPMiddleware`` — runs in-context so the value reaches the tool handler).

Transports (``MEMORY_MCP_TRANSPORT``):
  * ``http`` (default) — streamable-HTTP at ``http://<host>:<port>/mcp``; one shared server, many tokens.
  * ``stdio`` — spawned per-agent locally; binds to ``MEMORY_MCP_COMPANY`` (default ``global``); no token.

Run: ``make memory-mcp``. Config in ``.env`` (see ``.env.example``). See ../CLAUDE.md.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import time
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from .memory import GLOBAL
from .memory import close as _close
from .memory import recall_for as _recall_for
from .memory import remember_for as _remember_for

_HOST = os.environ.get("MEMORY_MCP_HOST", "127.0.0.1")
_PORT = int(os.environ["MEMORY_MCP_PORT"])
_TRANSPORT = os.environ.get("MEMORY_MCP_TRANSPORT", "http").strip().lower()

# Single-tenant fallback / stdio default company.
_DEFAULT_COMPANY = os.environ.get("MEMORY_MCP_COMPANY", GLOBAL).strip() or GLOBAL

# token -> company map. Falls back to the single MEMORY_MCP_TOKEN bound to MEMORY_MCP_COMPANY.
try:
    _TOKENS: dict[str, str] = json.loads(os.environ.get("MEMORY_MCP_TOKENS", "").strip() or "{}")
    if not isinstance(_TOKENS, dict):
        _TOKENS = {}
except Exception:
    _TOKENS = {}
_SINGLE_TOKEN = os.environ.get("MEMORY_MCP_TOKEN", "").strip()
if _SINGLE_TOKEN and not _TOKENS:
    _TOKENS = {_SINGLE_TOKEN: _DEFAULT_COMPANY}

# Live resolution: tokens not in the static map are resolved token->company by calling Paperclip
# (GET /api/internal/memory-mcp/resolve), cached for the TTL. The static _TOKENS map (the global tier
# + offline overrides) is always checked first, with no network. This is how per-company tokens that
# Paperclip auto-mints at company-create work without editing .env / restarting memory-mcp.
_PAPERCLIP_BASE_URL = os.environ.get("PAPERCLIP_BASE_URL", "").rstrip("/")
_RESOLVE_TTL_S = float(os.environ.get("MEMORY_MCP_RESOLVE_TTL_S", "60"))
_RESOLVE_URL = f"{_PAPERCLIP_BASE_URL}/api/internal/memory-mcp/resolve"
_INTERNAL_TOKEN = os.environ.get("PAPERCLIP_INTERNAL_TOKEN", "").strip()
_RESOLVE_ENABLED = _RESOLVE_TTL_S > 0
# Auth gates whenever either path can resolve a token (static map non-empty OR live resolution on).
_AUTH_REQUIRED = bool(_TOKENS) or _RESOLVE_ENABLED

# token -> (company, monotonic_expiry) — successful live resolutions only.
_resolve_cache: dict[str, tuple[str, float]] = {}

# Per-request company, set by ScopedAuth (HTTP) or left at the default (stdio).
_company_var: contextvars.ContextVar[str] = contextvars.ContextVar("memory_mcp_company", default=_DEFAULT_COMPANY)

mcp = FastMCP("titanofindustry-memory", host=_HOST, port=_PORT, json_response=True, stateless_http=True)


@mcp.tool()
async def remember(text: str, tags: str = "") -> dict[str, Any]:
    """Store a durable fact/decision in shared memory so future sessions and agents can reuse it.

    Call this AFTER a change is verified or a useful fact is established. Keep it concise and
    self-contained (one fact per call); add comma-separated tags to aid retrieval. The fact is scoped
    to the company bound to your bearer token (visible to that company plus the shared global tier).

    Args:
        text: The fact/decision to remember (concise, self-contained).
        tags: Optional comma-separated tags, e.g. "build,ci,deps".

    Returns:
        dict with ``success`` (bool), ``embedded`` (bool — False means keyword-only), and the
        ``company`` it was scoped to; or ``success=False`` with an ``error`` (Neo4j down, write failed).
    """
    return await _remember_for(_company_var.get(), text, tags)


@mcp.tool()
async def recall(query: str, k: int = 5) -> dict[str, Any]:
    """Search shared memory for facts/decisions relevant to a query (call BEFORE planning a task).

    Returns memories for your token's company plus the shared global tier. Uses semantic vector search
    when the embedding server is available, else keyword (full-text).

    Args:
        query: What to look for (natural language or keywords).
        k: Max results to return (default 5).

    Returns:
        dict with ``success`` (bool), ``mode`` ("vector" | "keyword"), ``results`` — a list of
        ``{text, tags, topic, importance, company, score}`` — and ``related`` (same-topic, high-importance
        memories; ``[]`` until ``make memory-graph`` has run). Or ``success=False`` with an ``error``.
    """
    return await _recall_for(_company_var.get(), query, k)


def _bearer_from_scope(scope: dict[str, Any]) -> str:
    for key, value in scope.get("headers", []):
        if key == b"authorization":
            val = value.decode("latin-1")
            return val[7:].strip() if val.lower().startswith("bearer ") else ""
    return ""


async def _send_401(send: Any) -> None:
    body = b'{"error":"unauthorized"}'
    await send({"type": "http.response.start", "status": 401,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def _resolve_company(token: str) -> str | None:
    """Resolve a bearer token -> company: static map first (offline/global), then a TTL cache, then a
    live lookup against Paperclip. ``None`` on miss / Paperclip error (fail-closed — never default-scope)."""
    if not token:
        return None
    mapped = _TOKENS.get(token)  # (a) static map: global tier + offline overrides, no network
    if mapped:
        return mapped
    if not _RESOLVE_ENABLED:
        return None
    now = time.monotonic()
    cached = _resolve_cache.get(token)  # (b) TTL cache of prior live resolutions
    if cached and cached[1] > now:
        return cached[0]
    for stale in [t for t, (_c, exp) in _resolve_cache.items() if exp <= now]:
        del _resolve_cache[stale]  # bound the cache to live tokens (rotations would otherwise leak keys)
    headers = {"Authorization": f"Bearer {token}"}  # (c) live lookup against Paperclip
    if _INTERNAL_TOKEN:
        headers["X-Internal-Token"] = _INTERNAL_TOKEN
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_RESOLVE_URL, headers=headers)
        if resp.status_code == 200:
            company = resp.json().get("company_id")
            if company:
                _resolve_cache[token] = (company, now + _RESOLVE_TTL_S)
                return company
    except Exception:
        pass  # Paperclip down / network error -> miss (401); never fall back to the default company
    return None


class ScopedAuth:
    """Pure-ASGI middleware: resolve the bearer token -> company into ``_company_var`` (401 on unknown
    token when auth is configured). Pure ASGI (not BaseHTTPMiddleware) so the contextvar set here
    propagates to the downstream tool handler."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            if _AUTH_REQUIRED:
                company = await _resolve_company(_bearer_from_scope(scope))
                if company is None:
                    await _send_401(send)
                    return
                _company_var.set(company)
            else:
                _company_var.set(_DEFAULT_COMPANY)
        await self.app(scope, receive, send)


def _http_app() -> Any:
    """Starlette app: ScopedAuth (token->company + 401) wrapping the streamable-HTTP mount; the session
    manager runs once in the parent lifespan (Starlette doesn't run a mounted sub-app's own lifespan),
    and the Neo4j driver is closed on shutdown."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.routing import Mount

    @contextlib.asynccontextmanager
    async def lifespan(_app: Any) -> Any:
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                await _close()

    return Starlette(
        routes=[Mount("/", app=mcp.streamable_http_app())],
        middleware=[Middleware(ScopedAuth)],
        lifespan=lifespan,
    )


def main() -> None:
    if _TRANSPORT == "stdio":
        try:
            mcp.run(transport="stdio")  # binds to MEMORY_MCP_COMPANY (contextvar default); no network surface
        finally:
            import asyncio

            with contextlib.suppress(Exception):
                asyncio.run(_close())
    else:
        import uvicorn

        uvicorn.run(_http_app(), host=_HOST, port=_PORT)


if __name__ == "__main__":
    main()
