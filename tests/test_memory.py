"""Unit tests for agent/memory.py — exercise remember/recall logic without a live Neo4j or embed
server. Run from the repo root in the OpenSage venv:

    uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `import agent.*`

import agent.memory as mem


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def __aiter__(self):
        self._it = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeSession:
    def __init__(self, calls, rows):
        self._calls = calls
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **params):
        self._calls.append((cypher, params))
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows=None):
        self.calls: list = []
        self.rows = rows or []

    def session(self):
        return _FakeSession(self.calls, self.rows)


def _install(rows=None, embedding=None) -> _FakeDriver:
    """Point memory at a fake driver + embedder; return the fake driver for assertions."""
    mem._indexes_ready = False
    fake = _FakeDriver(rows=rows)
    mem._get_driver = lambda: fake

    async def _fake_embed(_text):
        return embedding

    mem._embed = _fake_embed
    return fake


def test_remember_with_embedding():
    fake = _install(embedding=[0.1] * 768)
    out = asyncio.run(mem.remember("hello fact", tags="t1,t2"))
    assert out["success"] is True and out["embedded"] is True
    merge = [c for c in fake.calls if "MERGE" in c[0]]
    assert len(merge) == 1
    params = merge[0][1]
    assert params["text"] == "hello fact"
    assert params["tags"] == "t1,t2"
    assert params["emb"] == [0.1] * 768
    assert params["th"] == mem._hash("hello fact")
    assert params["kind"] == "episodic"  # default kind set on create


def test_remember_keyword_fallback_when_no_embedding():
    fake = _install(embedding=None)
    out = asyncio.run(mem.remember("no-embed fact"))
    assert out["success"] is True and out["embedded"] is False
    assert [c for c in fake.calls if "MERGE" in c[0]][0][1]["emb"] is None


def test_remember_rejects_empty():
    _install(embedding=[0.0] * 768)
    assert asyncio.run(mem.remember("   "))["success"] is False


def test_recall_vector_mode():
    rows = [{"text": "a", "tags": "x", "score": 0.9}]
    fake = _install(rows=rows, embedding=[0.2] * 768)
    out = asyncio.run(mem.recall("query", k=3))
    assert out["success"] is True and out["mode"] == "vector"
    assert out["results"] == rows
    vec = [c for c in fake.calls if "db.index.vector.queryNodes" in c[0]]
    assert vec and vec[0][1]["k"] == 3


def test_recall_keyword_mode_when_no_embedding():
    rows = [{"text": "b", "tags": "y", "score": 1.2}]
    fake = _install(rows=rows, embedding=None)
    out = asyncio.run(mem.recall("query"))
    assert out["mode"] == "keyword"
    assert [c for c in fake.calls if "db.index.fulltext.queryNodes" in c[0]]


def test_recall_excludes_archived():
    rows = [{"text": "a", "tags": "x", "score": 0.5}]
    fake = _install(rows=rows, embedding=[0.1] * 768)
    asyncio.run(mem.recall("q"))
    vec = [c for c in fake.calls if "db.index.vector.queryNodes" in c[0]][0]
    assert "archived" in vec[0]  # soft-deleted memories are filtered out of recall


def test_recall_tracks_access():
    # rows carry scope_key (topic absent → no graph expansion) so the access counter fires.
    rows = [{"text": "a", "tags": "x", "topic": None, "scope_key": "global:h", "score": 0.7}]
    fake = _install(rows=rows, embedding=None)
    out = asyncio.run(mem.recall("q"))
    assert out["success"]
    touch = [c for c in fake.calls if "SET m.last_accessed" in c[0]]
    assert touch and touch[0][1]["sks"] == ["global:h"]
    assert "scope_key" not in out["results"][0]  # internal key stripped from agent-facing output
