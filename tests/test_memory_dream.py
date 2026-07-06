"""Unit tests for agent/memory_dream.py (the offline consolidation / "dreaming" pass).

Exercises stage ordering, graceful degradation (embeddings off, LLM unavailable), the forgetting
guards, and the JSON/ranking helpers — all without a live Neo4j, embedder, or model.

Run: uv run --project vendor/opensage-adk --with pytest pytest tests/test_memory_dream.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `import agent.*`

import agent.memory as mem
import agent.memory_dream as md


class _FakeResult:
    def __init__(self, rows=None, single=None):
        self._rows = list(rows or [])
        self._single = single

    def __aiter__(self):
        self._it = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def single(self):
        return self._single


class _FakeSession:
    def __init__(self, calls, responder):
        self._calls = calls
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **params):
        self._calls.append((cypher, params))
        return self._responder(cypher, params)


class _FakeDriver:
    def __init__(self, responder):
        self.calls: list = []
        self._responder = responder

    def session(self):
        return _FakeSession(self.calls, self._responder)


def _install(responder) -> _FakeDriver:
    mem._indexes_ready = True  # skip DDL in unit tests
    fake = _FakeDriver(responder)
    mem._get_driver = lambda: fake
    return fake


# ---- pure helpers --------------------------------------------------------------------------------
def test_parse_json_handles_fences_and_noise():
    assert md._parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert md._parse_json('here is the verdict: {"contradict": true} ok') == {"contradict": True}
    assert md._parse_json("not json at all") is None
    assert md._parse_json(None) is None


def test_rank_keep_prefers_richer():
    a = {"sk": "a", "imp": 0.9, "ac": 1, "ts": 10}
    b = {"sk": "b", "imp": 0.1, "ac": 5, "ts": 99}
    keep, drop = md._rank_keep(a, b)
    assert keep["sk"] == "a" and drop["sk"] == "b"  # importance dominates


# ---- full deterministic pass (embeddings off, LLM never reached) ---------------------------------
def test_dream_deterministic_pass_skips_semantic_stages():
    # Pin the config this test asserts on — module constants are read from the env at import, and
    # litellm's load_dotenv() pulls in the operator's live .env, so don't depend on its values.
    saved = (mem._EMBEDDING_MODEL, md._PPR_ENABLED, md._ENABLE_FORGET)
    mem._EMBEDDING_MODEL = ""  # keyword-only → backfill/dedup/contradict skip
    md._PPR_ENABLED = False    # opt-in associative-enrichment stage off
    md._ENABLE_FORGET = False  # forgetting disabled → dry-run count only

    def responder(cypher, _params):
        if "SET m.decay_score" in cypher:
            return _FakeResult(single={"scored": 5})
        if "RETURN topic, company" in cypher:  # reflect cluster selection
            return _FakeResult(rows=[])
        if "<> $global_tier" in cypher:  # forget dry-run count
            return _FakeResult(single={"n": 3})
        return _FakeResult()

    _install(responder)
    try:
        out = asyncio.run(md.dream())
    finally:
        mem._EMBEDDING_MODEL, md._PPR_ENABLED, md._ENABLE_FORGET = saved

    assert out["success"]
    assert out["backfill"]["skipped"] == "embeddings_off"
    assert out["decay"] == {"scored": 5}
    assert out["dedup"]["skipped"] == "embeddings_off"
    assert out["contradict"]["skipped"] == "embeddings_off"
    assert out["reflect"]["skipped"] == "no_eligible_clusters"
    assert out["forget"] == {"archived": 0, "would_archive": 3, "skipped": "disabled"}
    assert out["ppr"]["skipped"] == "disabled"  # opt-in, default off


def test_dream_backfill_only_short_circuits():
    saved = mem._EMBEDDING_MODEL
    mem._EMBEDDING_MODEL = ""
    _install(lambda c, p: _FakeResult())
    try:
        out = asyncio.run(md.dream(backfill_only=True))
    finally:
        mem._EMBEDDING_MODEL = saved
    assert out["backfill"]["skipped"] == "embeddings_off"
    assert "decay" not in out and "forget" not in out  # nothing past backfill ran


# ---- forgetting: disabled = dry-run; enabled = bounded + guarded ---------------------------------
def test_forget_disabled_reports_would_archive_without_writing():
    md._ENABLE_FORGET = False
    writes = []

    def responder(cypher, _params):
        if "SET m.archived = true" in cypher:
            writes.append(cypher)
        return _FakeResult(single={"n": 9})

    _install(responder)
    out = asyncio.run(md._stage_forget())
    assert out == {"archived": 0, "would_archive": 9, "skipped": "disabled"}
    assert not writes  # disabled must never archive


def test_forget_enabled_is_bounded_and_guarded():
    md._ENABLE_FORGET = True
    seen = {}

    def responder(cypher, _params):
        seen["cypher"] = cypher
        return _FakeResult(single={"n": 7})

    _install(responder)
    try:
        out = asyncio.run(md._stage_forget())
    finally:
        md._ENABLE_FORGET = False  # restore default

    assert out == {"archived": 7}
    c = seen["cypher"]
    assert "SET m.archived = true" in c and "LIMIT $max_per_run" in c  # soft + bounded
    for guard in (
        "m.company <> $global_tier",          # never forget the shared tier
        "coalesce(m.source, 'agent') <> 'user'",  # never forget user-stated facts
        "coalesce(m.access_count, 0) = 0",    # only never-accessed
        "coalesce(m.kind, 'episodic') = 'episodic'",  # never forget reflections
    ):
        assert guard in c, f"missing forgetting guard: {guard}"


# ---- reflection: no-op when the local model is unavailable; writes provenance when it answers -----
def _reflect_responder_with_one_cluster():
    def responder(cypher, _params):
        if "RETURN topic, company" in cypher:
            return _FakeResult(rows=[{"topic": 1, "company": "c1", "n": 3, "imp": 5.0}])
        if "DESC LIMIT $lim" in cypher:  # cluster members
            return _FakeResult(rows=[
                {"sk": "c1:a", "text": "x", "tags": "t1", "ts": 1},
                {"sk": "c1:b", "text": "y", "tags": "t2", "ts": 2},
                {"sk": "c1:c", "text": "z", "tags": "", "ts": 3},
            ])
        if "RETURN count(r) AS c" in cypher:  # idempotency coverage check
            return _FakeResult(single={"c": 0})
        return _FakeResult()

    return responder


def test_reflect_noop_when_llm_unavailable():
    saved = md._ENABLE_LLM
    md._ENABLE_LLM = False  # _llm returns None → nothing synthesized
    _install(_reflect_responder_with_one_cluster())
    try:
        out = asyncio.run(md._stage_reflect())
    finally:
        md._ENABLE_LLM = saved
    assert out["reflections_created"] == 0 and out["llm_calls"] == 1


def test_reflect_creates_reflection_when_llm_answers():
    orig_llm, orig_store = md._llm, mem._store

    async def fake_llm(_system, _user):
        return '[{"insight": "team prefers trunk-based dev", "evidence_indices": [0, 1], "confidence": 0.8}]'

    async def fake_store(text, tags, company, agent, kind="episodic"):
        assert kind == "reflection" and company == "c1"  # stored as a scoped reflection
        return {"success": True, "scope_key": f"{company}:rk", "embedded": False}

    edges = {"n": 0}

    def responder(cypher, _params):
        if "MERGE (r)-[:CONSOLIDATES" in cypher:
            edges["n"] += 1
        base = _reflect_responder_with_one_cluster()
        return base(cypher, _params)

    _install(responder)
    md._llm, mem._store = fake_llm, fake_store
    try:
        out = asyncio.run(md._stage_reflect())
    finally:
        md._llm, mem._store = orig_llm, orig_store

    assert out["reflections_created"] == 1 and out["llm_calls"] == 1
    assert edges["n"] == 1  # reflection linked to its evidence via CONSOLIDATES


# ---- dedup / contradiction skip cleanly with embeddings off --------------------------------------
def test_semantic_stages_skip_without_embeddings():
    saved = mem._EMBEDDING_MODEL
    mem._EMBEDDING_MODEL = ""
    _install(lambda c, p: _FakeResult())
    try:
        assert asyncio.run(md._stage_dedup())["skipped"] == "embeddings_off"
        assert asyncio.run(md._stage_contradict())["skipped"] == "embeddings_off"
    finally:
        mem._EMBEDDING_MODEL = saved
