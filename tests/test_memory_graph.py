"""Unit tests for agent/memory_graph.py (rebuild_graph) and the graph-enriched recall.

Run: uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `import agent.*`

import agent.memory as mem
import agent.memory_graph as mg


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
    mem._indexes_ready = True  # skip index/constraint DDL in these unit tests
    fake = _FakeDriver(responder)
    mem._get_driver = lambda: fake
    return fake


def test_rebuild_graph_runs_gds_pipeline_in_order():
    def responder(cypher, _params):
        if "count(m)" in cypher:
            return _FakeResult(single={"memories": 3, "with_topic": 3})
        if "count(r)" in cypher:
            return _FakeResult(single={"sim_edges": 2})
        return _FakeResult()

    fake = _install(responder)
    out = asyncio.run(mg.rebuild_graph())
    assert out == {"success": True, "memories": 3, "with_topic": 3, "sim_edges": 2}
    seq = " || ".join(c[0] for c in fake.calls)
    for needle in (
        "MERGE (g:Tag",
        "gds.graph.project",
        "gds.nodeSimilarity.write",
        "gds.louvain.write",
        "gds.pageRank.write",
        "gds.graph.drop",
    ):
        assert needle in seq, f"missing GDS step: {needle}"
    assert seq.index("gds.louvain.write") < seq.index("gds.pageRank.write")
    assert seq.index("gds.nodeSimilarity.write") < seq.index("gds.louvain.write")


def test_recall_includes_topic_importance_and_related():
    def responder(cypher, _params):
        if "queryNodes" in cypher:
            return _FakeResult(rows=[
                {"text": "a", "tags": "x", "topic": 1, "importance": 0.9, "scope_key": "global:h1", "score": 0.8},
            ])
        if "WHERE m.topic IN" in cypher:
            return _FakeResult(rows=[{"text": "b", "tags": "x", "topic": 1, "importance": 0.95}])
        return _FakeResult()

    _install(responder)
    mem._EMBEDDING_MODEL = ""  # keyword mode (no embed call)
    out = asyncio.run(mem.recall("where?", k=3))
    assert out["success"] and out["mode"] == "keyword"
    assert out["results"][0]["topic"] == 1 and out["results"][0]["importance"] == 0.9
    assert "scope_key" not in out["results"][0]  # internal, stripped from agent-facing output
    assert out["related"] == [{"text": "b", "tags": "x", "topic": 1, "importance": 0.95}]


def test_recall_graceful_before_graph_built():
    def responder(cypher, _params):
        if "queryNodes" in cypher:
            return _FakeResult(rows=[
                {"text": "a", "tags": "", "topic": None, "importance": None, "text_hash": "h1", "score": 0.5},
            ])
        return _FakeResult()

    fake = _install(responder)
    mem._EMBEDDING_MODEL = ""
    out = asyncio.run(mem.recall("anything"))
    assert out["success"] and out["related"] == []
    assert not any("WHERE m.topic IN" in c[0] for c in fake.calls)  # related query skipped
