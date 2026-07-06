"""Shared, persistent memory for the TitanOfIndustry agent (DIY: Neo4j Community + local embeddings).

Two ADK function tools — ``remember`` (store) and ``recall`` (retrieve) — over a single ``:Memory``
node type in a Neo4j Community database. Embeddings come from a local vLLM embedding server via
LiteLLM; when embeddings are unavailable, both write and read fall back to Neo4j full-text (keyword)
search.

**Tenant scoping (company + agent):** every memory carries a ``company`` and ``agent``. ``remember``
auto-scopes to the Paperclip company/agent from the ADK session state (``paperclip_company_id`` /
``paperclip_agent_id``, seeded by the opensage adapter — see ``paperclip_tool.py``); with no context
it falls back to the shared ``global`` tier. ``recall`` returns the caller's company **∪ ``global``**
(stack-wide facts stay shared; company-specific stays isolated). Dedup is per-(company) via a single
``scope_key`` (Neo4j Community has no composite unique constraints). Out-of-process callers (memory_mcp)
pass the scope explicitly via ``remember_for`` / ``recall_for``.

Process boundary by design: plain host functions (like ``opencode_tool``). See ../CLAUDE.md.
We deliberately bypass OpenSage's built-in memory (needs Neo4j Enterprise + a 3072-dim embedding).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import litellm
from google.adk.tools.tool_context import ToolContext
from neo4j import AsyncGraphDatabase

from .adk_state import AGENT_ID_KEY, COMPANY_ID_KEY, state_get

_NEO4J_URI = os.environ.get("NEO4J_URI", "")
_NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")
_EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
_EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

_VECTOR_INDEX = "memory_embedding_index"
_FULLTEXT_INDEX = "memory_text_index"

# The shared tier every company can also read; also the fallback for writes with no company context.
GLOBAL = "global"

_driver: Any = None  # lazy singleton
_indexes_ready = False


def _get_driver() -> Any:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    return _driver


async def close() -> None:
    """Close the shared Neo4j driver. A clean-shutdown hook for out-of-process servers (memory_mcp);
    OpenSage never calls this (it shares the singleton for the process lifetime)."""
    global _driver, _indexes_ready
    if _driver is not None:
        await _driver.close()
        _driver = None
        _indexes_ready = False


async def _ensure_indexes() -> None:
    """Idempotently create the vector + full-text indexes and the per-scope uniqueness constraint.

    Index OPTIONS cannot take query parameters, so the (int-validated) dimension is interpolated.
    Dedup is keyed on ``scope_key`` (= ``company:text_hash``) so two companies can hold the same text
    as separate nodes; the old global ``text_hash`` constraint is dropped (Community has no composite
    unique constraint, hence the single combined key).

    Also creates lookup indexes on ``archived`` and ``kind`` so the dream pass
    (``agent/memory_dream.py``) and recall's archived filter stay cheap.
    """
    global _indexes_ready
    if _indexes_ready:
        return
    async with _get_driver().session() as session:
        await session.run(
            f"CREATE VECTOR INDEX {_VECTOR_INDEX} IF NOT EXISTS "
            "FOR (m:Memory) ON (m.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {_EMBEDDING_DIM}, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        await session.run(
            f"CREATE FULLTEXT INDEX {_FULLTEXT_INDEX} IF NOT EXISTS FOR (m:Memory) ON EACH [m.text]"
        )
        await session.run("DROP CONSTRAINT memory_text_hash_unique IF EXISTS")
        await session.run(
            "CREATE CONSTRAINT memory_scope_key_unique IF NOT EXISTS "
            "FOR (m:Memory) REQUIRE m.scope_key IS UNIQUE"
        )
        await session.run(
            "CREATE INDEX memory_archived_idx IF NOT EXISTS FOR (m:Memory) ON (m.archived)"
        )
        await session.run(
            "CREATE INDEX memory_kind_idx IF NOT EXISTS FOR (m:Memory) ON (m.kind)"
        )
    _indexes_ready = True


async def _embed(text: str) -> list[float] | None:
    """Embed text via the local vLLM embedding server; ``None`` if unavailable or disabled.

    When ``EMBEDDING_MODEL`` is empty (keyword-only mode — the default when no embed server fits
    alongside the chat model), skip the call entirely so recall/remember go straight to full-text.
    """
    if not _EMBEDDING_MODEL:
        return None
    try:
        resp = await litellm.aembedding(
            model=_EMBEDDING_MODEL, input=[text], api_base=_EMBEDDING_BASE_URL, api_key="EMPTY"
        )
        return resp.data[0]["embedding"]
    except Exception:
        return None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lucene_safe(text: str) -> str:
    """Reduce a free-text query to plain terms so Neo4j's Lucene full-text parser cannot choke on
    metacharacters (``? : + - ( ) [ ] ^ " ~ * \\`` etc. from a natural-language query)."""
    cleaned = re.sub(r"[^\w\s]", " ", text).strip()
    return cleaned or text


def _scope(tool_context: Any) -> tuple[str, str]:
    """Resolve ``(company, agent)`` from ADK session state; ``(global, "")`` when there is no context."""
    return (state_get(tool_context, COMPANY_ID_KEY) or GLOBAL, state_get(tool_context, AGENT_ID_KEY))


def _scopes(company: str) -> list[str]:
    """Recall visibility: a company sees its own tier ∪ the shared ``global`` tier."""
    return [GLOBAL] if company == GLOBAL else [company, GLOBAL]


# ---- scoped core (explicit company/agent) — used by the ADK tools AND memory_mcp ----
async def _store(text: str, tags: str, company: str, agent: str, kind: str = "episodic") -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    try:
        await _ensure_indexes()
    except Exception as exc:
        return {"success": False, "error": f"neo4j unavailable: {exc}"}
    emb = await _embed(text)
    h = _hash(text)  # hash once; scope_key = "company:text_hash" (format mirrored in migrate_memory_scope.py)
    sk = f"{company}:{h}"
    try:
        async with _get_driver().session() as session:
            # kind is create-only so the dream pass's reflection/semantic upgrades survive a re-remember.
            await session.run(
                "MERGE (m:Memory {scope_key: $sk}) "
                "ON CREATE SET m.text = $text, m.text_hash = $th, m.company = $company, "
                "m.agent = $agent, m.ts = timestamp(), m.source = 'agent', m.kind = $kind "
                "SET m.tags = $tags, m.embedding = $emb",
                sk=sk, th=h, text=text, company=company, agent=agent, tags=tags, emb=emb, kind=kind,
            )
    except Exception as exc:
        return {"success": False, "error": f"write failed: {exc}"}
    return {"success": True, "embedded": emb is not None, "company": company, "scope_key": sk}


async def _search(query: str, k: int, company: str, agent: str | None = None) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "empty query"}
    try:
        await _ensure_indexes()
    except Exception as exc:
        return {"success": False, "error": f"neo4j unavailable: {exc}"}
    emb = await _embed(query)
    scopes = _scopes(company)
    overfetch = max(k * 10, 50)  # vector/fulltext index calls can't pre-filter; over-fetch then filter
    agent_clause = " AND node.agent = $agent" if agent else ""
    # exclude soft-deleted memories (archived by the dream pass); absent property = not archived.
    where = (
        f"WHERE node.company IN $scopes{agent_clause} "
        "AND (node.archived IS NULL OR node.archived = false) "
    )
    # topic/importance are populated by `make memory-graph`; related_keys by `make memory-dream`
    # (Stage 6, opt-in) — all null until then (graceful).
    ret = (
        "RETURN node.text AS text, node.tags AS tags, node.topic AS topic, node.importance AS importance, "
        "node.company AS company, node.scope_key AS scope_key, node.related_keys AS related_keys, score "
    )
    try:
        async with _get_driver().session() as session:
            if emb is not None:
                result = await session.run(
                    f"CALL db.index.vector.queryNodes('{_VECTOR_INDEX}', $of, $vec) YIELD node, score "
                    + where + ret + "ORDER BY score DESC LIMIT $k",
                    of=overfetch, vec=emb, scopes=scopes, agent=agent, k=k,
                )
                mode = "vector"
            else:
                result = await session.run(
                    f"CALL db.index.fulltext.queryNodes('{_FULLTEXT_INDEX}', $q) YIELD node, score "
                    + where + ret + "ORDER BY score DESC LIMIT $k",
                    q=_lucene_safe(query), scopes=scopes, agent=agent, k=k,
                )
                mode = "keyword"
            rows = [dict(r) async for r in result]
            related = await _related(session, rows, scopes)
            await _touch(session, rows)  # best-effort recency/frequency reinforcement (feeds decay)
    except Exception as exc:
        return {"success": False, "error": f"search failed: {exc}"}
    for r in rows:  # scope_key + related_keys are internal — keep the agent-facing output clean
        r.pop("scope_key", None)
        r.pop("related_keys", None)
    return {"success": True, "mode": mode, "results": rows, "related": related}


async def _touch(session: Any, rows: list[dict[str, Any]]) -> None:
    """Best-effort access reinforcement: bump ``last_accessed`` + ``access_count`` on the returned
    memories. NEVER raises — a recall must not fail because this telemetry write did; it only feeds
    the dream pass's Ebbinghaus decay (``agent/memory_dream.py``)."""
    sks = [r.get("scope_key") for r in rows if r.get("scope_key")]
    if not sks:
        return
    try:
        await session.run(
            "UNWIND $sks AS sk MATCH (m:Memory {scope_key: sk}) "
            "SET m.last_accessed = timestamp(), m.access_count = coalesce(m.access_count, 0) + 1",
            sks=sks,
        )
    except Exception:
        pass


async def _related(
    session: Any, rows: list[dict[str, Any]], scopes: list[str], limit: int = 3
) -> list[dict[str, Any]]:
    """Graph expansion of the recall hits (within scope, excluding archived + already-returned rows).

    Prefers the dream pass's precomputed associative neighbors (``related_keys``, HippoRAG-style
    multi-hop proximity from ``make memory-dream`` Stage 6) when present; otherwise falls back to
    same-topic, higher-importance memories from ``make memory-graph``. Returns ``[]`` when neither
    layer has been built, keeping recall graceful.
    """
    seen = [r.get("scope_key") for r in rows]
    # (a) precomputed associative neighbors (dream Stage 6 PPR enrichment), if present
    assoc: list[str] = []
    for r in rows:
        for sk in r.get("related_keys") or []:
            if sk not in seen and sk not in assoc:
                assoc.append(sk)
    if assoc:
        result = await session.run(
            "MATCH (m:Memory) WHERE m.scope_key IN $sks AND m.company IN $scopes "
            "AND (m.archived IS NULL OR m.archived = false) "
            "RETURN m.text AS text, m.tags AS tags, m.topic AS topic, m.importance AS importance, "
            "m.company AS company ORDER BY coalesce(m.importance, 0.0) DESC LIMIT $limit",
            sks=assoc, scopes=scopes, limit=limit,
        )
        hits = [dict(r) async for r in result]
        if hits:
            return hits
    # (b) fallback: same-topic, higher-importance memories
    topics = [r["topic"] for r in rows if r.get("topic") is not None]
    if not topics:
        return []
    result = await session.run(
        "MATCH (m:Memory) WHERE m.topic IN $topics AND m.importance IS NOT NULL "
        "AND m.company IN $scopes AND NOT m.scope_key IN $seen "
        "AND (m.archived IS NULL OR m.archived = false) "
        "RETURN m.text AS text, m.tags AS tags, m.topic AS topic, m.importance AS importance, "
        "m.company AS company ORDER BY m.importance DESC LIMIT $limit",
        topics=topics, scopes=scopes, seen=seen, limit=limit,
    )
    return [dict(r) async for r in result]


# ---- ADK function tools (auto-scope from session state; scope is invisible to the model) ----
async def remember(text: str, tags: str = "", tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Store a durable fact/decision in shared memory so future sessions and agents can reuse it.

    Call this AFTER a change is verified or a useful fact is established. Keep it concise and
    self-contained (one fact per call); add comma-separated tags to aid retrieval.

    Args:
        text: The fact/decision to remember (concise, self-contained).
        tags: Optional comma-separated tags, e.g. "build,ci,deps".

    Returns:
        dict with ``success`` (bool), ``embedded`` (bool — False means keyword-only), and the
        ``company`` it was scoped to; or ``success=False`` with an ``error`` (Neo4j down, write failed).
    """
    company, agent = _scope(tool_context)
    return await _store(text, tags, company, agent)


async def recall(query: str, k: int = 3, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Search shared memory for facts/decisions relevant to a query (call BEFORE planning a task).

    Uses semantic vector search when the embedding server is available, else keyword (full-text).
    Returns memories for your own company plus the shared global tier.

    Args:
        query: What to look for (natural language or keywords).
        k: Max results to return (default 3; the top hits carry most of the value and this payload
            lands in the cloud planner's uncached tail, so keep it tight — raise k when you need breadth).

    Returns:
        dict with ``success`` (bool), ``mode`` ("vector" | "keyword"), ``results`` — a list of
        ``{text, tags, topic, importance, company, score}`` — and ``related`` (same-topic, high-importance
        memories from the graph layer; ``[]`` until ``make memory-graph`` has run). Or
        ``success=False`` with an ``error``.
    """
    company, _agent = _scope(tool_context)
    return await _search(query, k, company, agent=None)


# ---- explicit-scope wrappers for out-of-process callers (memory_mcp per-token binding) ----
async def remember_for(company: str, text: str, tags: str = "", agent: str = "") -> dict[str, Any]:
    """Store under an explicit company (and optional agent). ``company`` falsy → the ``global`` tier."""
    return await _store(text, tags, company or GLOBAL, agent or "")


async def recall_for(company: str, query: str, k: int = 5, agent: str | None = None) -> dict[str, Any]:
    """Recall for an explicit company (∪ ``global``). Pass ``agent`` to narrow to that agent's own writes."""
    return await _search(query, k, company or GLOBAL, agent=agent)
