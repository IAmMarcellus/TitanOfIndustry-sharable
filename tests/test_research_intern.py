"""Unit tests for agent/research_intern.py (the idle-time research scout / "intern").

Exercises the pure helpers (finding parsing/clamping/capping, dedup overlap, prompt scoping, tag/text
composition), beats config parsing + cooldown selection, the draft guardrails, and the run_unit
handoff — all without a live Neo4j, embedder, model, opencode, or Paperclip.

Run: uv run --project vendor/opensage-adk --with pytest pytest tests/test_research_intern.py -q
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `import agent.*`

import agent.research_intern as ri  # noqa: E402


# ---- finding parsing -----------------------------------------------------------------------------
def test_parse_findings_clamps_caps_and_floors():
    saved = (ri._WRITE_MIN_CONFIDENCE, ri._MAX_FINDINGS_PER_UNIT)
    ri._WRITE_MIN_CONFIDENCE, ri._MAX_FINDINGS_PER_UNIT = 0.3, 2
    try:
        raw = json.dumps(
            [
                {"title": "A", "text": "found a", "kind": "bug", "confidence": 1.7},  # clamp -> 1.0
                {"title": "B", "text": "found b", "kind": "edge_case", "confidence": 0.1},  # below floor -> dropped
                {"title": "C", "text": "found c", "kind": "perf", "confidence": 0.5},
                {"title": "D", "text": "found d", "kind": "perf", "confidence": 0.9},  # over cap (2) -> dropped
                {"title": "", "text": "no title", "confidence": 0.9},  # missing title -> skipped
                "not a dict",
            ]
        )
        out = ri._parse_findings(raw, default_evidence=["http://x"])
    finally:
        ri._WRITE_MIN_CONFIDENCE, ri._MAX_FINDINGS_PER_UNIT = saved
    assert [f.title for f in out] == ["A", "C"]
    assert out[0].confidence == 1.0
    assert out[0].evidence == ("http://x",)  # default evidence applied when none given


def test_parse_findings_non_list_is_empty():
    assert ri._parse_findings('{"title": "x"}', []) == []
    assert ri._parse_findings(None, []) == []
    assert ri._parse_findings("garbage", []) == []


def test_clamp01():
    assert ri._clamp01(2.0) == 1.0
    assert ri._clamp01(-1) == 0.0
    assert ri._clamp01("nan-ish") == 0.5  # default on bad input
    assert ri._clamp01(0.42) == 0.42


def test_overlap_jaccard():
    assert ri._overlap("fix the risk guard", "fix the risk guard") == 1.0
    assert ri._overlap("", "anything") == 0.0
    assert 0.0 < ri._overlap("risk guard missing", "missing risk check") < 1.0


# ---- prompt scoping + tag/text composition -------------------------------------------------------
def test_codebase_prompt_scopes_and_demands_json():
    beat = ri.Beat(name="b", type="codebase", prompt="Review {files} for bugs.", files=("a.py", "b.py"))
    p = ri._codebase_prompt(beat)
    assert "a.py, b.py" in p  # {files} substituted
    assert "READ-ONLY" in p and "JSON array" in p  # guardrail + contract appended


def test_finding_tags_and_text_carry_provenance():
    beat = ri.Beat(name="engine-risk-scan", type="codebase", tags="redacted-company,engine")
    f = ri.Finding(title="t", text="the body", kind="bug", confidence=0.83, evidence=("execution.py:42",))
    tags = ri._finding_tags(beat, f)
    for t in ("intern", "research", "candidate", "draft", "engine-risk-scan", "bug", "redacted-company", "engine"):
        assert t in tags.split(",")
    text = ri._finding_text(beat, f)
    assert "[intern draft" in text and "conf=0.83" in text and "execution.py:42" in text


def test_strip_html_drops_scripts_and_tags():
    html = "<html><script>var x=1</script><p>Hello <b>world</b></p></html>"
    assert ri._strip_html(html) == "Hello world"


# ---- beats config + selection --------------------------------------------------------------------
_BEATS_TOML = """
[[beat]]
name = "code"
type = "codebase"
files = ["x.py"]
min_interval_s = 100

[[beat]]
name = "off"
type = "web"
enabled = false

[[beat]]
name = "bogus"
type = "nonsense"
"""


def test_load_beats_filters_disabled_and_unknown(tmp_path):
    f = tmp_path / "beats.toml"
    f.write_text(_BEATS_TOML)
    saved = ri._BEATS_FILE
    ri._BEATS_FILE = str(f)
    try:
        beats = ri._load_beats()
    finally:
        ri._BEATS_FILE = saved
    assert [b.name for b in beats] == ["code"]  # disabled + unknown-type dropped
    assert beats[0].files == ("x.py",)


def test_load_beats_missing_file_is_empty():
    saved = ri._BEATS_FILE
    ri._BEATS_FILE = "/nonexistent/intern_beats.toml"
    try:
        assert ri._load_beats() == []
    finally:
        ri._BEATS_FILE = saved


def test_select_beat_respects_cooldown_and_weight():
    a = ri.Beat(name="a", type="synthesis", weight=1.0, min_interval_s=1000)
    b = ri.Beat(name="b", type="synthesis", weight=5.0, min_interval_s=0)
    saved = dict(ri._last_run)
    ri._last_run.clear()
    try:
        # fresh: both eligible; b's higher weight wins the tie-break
        assert ri._select_beat([a, b]).name == "b"
        # a just ran and is on a long cooldown → only b is eligible
        ri._last_run["a"] = ri.time.monotonic()
        assert ri._select_beat([a]) is None
        assert ri._select_beat([a, b]).name == "b"
    finally:
        ri._last_run.clear()
        ri._last_run.update(saved)


# ---- draft guardrails ----------------------------------------------------------------------------
def test_file_draft_gated_off_returns_none():
    f = ri.Finding(title="t", text="x", kind="bug", confidence=0.99)
    saved = (ri._ENABLE_DRAFTS, ri._DRAFT_MIN_CONFIDENCE, ri._COMPANY)
    try:
        ri._ENABLE_DRAFTS = False
        assert asyncio.run(ri._file_draft(ri.Beat(name="b", type="codebase"), f)) is None
        ri._ENABLE_DRAFTS = True
        ri._DRAFT_MIN_CONFIDENCE = 0.6
        low = ri.Finding(title="t", text="x", kind="bug", confidence=0.4)
        assert asyncio.run(ri._file_draft(ri.Beat(name="b", type="codebase"), low)) is None  # below threshold
        ri._COMPANY = ri.memory.GLOBAL
        assert asyncio.run(ri._file_draft(ri.Beat(name="b", type="codebase"), f)) is None  # no real company
    finally:
        ri._ENABLE_DRAFTS, ri._DRAFT_MIN_CONFIDENCE, ri._COMPANY = saved


# ---- handoff: write + run_unit -------------------------------------------------------------------
def test_write_finding_scopes_to_company_with_agent_empty():
    captured: dict = {}

    async def fake_remember(company, text, tags="", agent=""):
        captured.update(company=company, text=text, tags=tags, agent=agent)
        return {"success": True}

    beat = ri.Beat(name="synth", type="synthesis", tags="redacted-company")
    f = ri.Finding(title="t", text="body", kind="hypothesis", confidence=0.7)
    saved_company, saved_fn = ri._COMPANY, ri.memory.remember_for
    ri._COMPANY = "CELLBOT"
    ri.memory.remember_for = fake_remember
    try:
        status = asyncio.run(ri._write_finding(beat, f))
    finally:
        ri._COMPANY, ri.memory.remember_for = saved_company, saved_fn
    assert status == "written"
    assert captured["company"] == "CELLBOT"
    assert captured["agent"] == ""  # company-wide visible so desk agents' recall surfaces it
    assert "intern" in captured["tags"] and "synth" in captured["tags"]


def test_run_unit_handoff_counts(monkeypatch):
    beat = ri.Beat(name="synth", type="synthesis")
    findings = [
        ri.Finding(title="keep", text="a", kind="idea", confidence=0.9),
        ri.Finding(title="dup", text="b", kind="idea", confidence=0.9),
    ]

    async def fake_runner(_beat):
        return findings

    async def fake_seen(_beat, f):
        return f.title == "dup"  # second one is a near-dup → skipped

    async def fake_write(_beat, _f):
        return "written"

    async def fake_draft(_beat, _f):
        return "ISSUE-1"

    monkeypatch.setitem(ri._RUNNERS, "synthesis", fake_runner)
    monkeypatch.setattr(ri, "_already_seen", fake_seen)
    monkeypatch.setattr(ri, "_write_finding", fake_write)
    monkeypatch.setattr(ri, "_file_draft", fake_draft)

    out = asyncio.run(ri.run_unit(beat, write=True))
    assert out["findings"] == 2
    assert out["written"] == 1  # the dup was skipped
    assert out["drafted"] == 1
    assert out["skipped_dup"] == 1
    assert out["drafts"] == ["ISSUE-1"]


def test_run_unit_dry_run_writes_nothing(monkeypatch):
    beat = ri.Beat(name="synth", type="synthesis")

    async def fake_runner(_beat):
        return [ri.Finding(title="x", text="y", kind="idea", confidence=0.9)]

    async def boom(*_a, **_k):  # must NOT be called in dry-run
        raise AssertionError("write path hit during dry-run")

    monkeypatch.setitem(ri._RUNNERS, "synthesis", fake_runner)
    monkeypatch.setattr(ri, "_write_finding", boom)
    monkeypatch.setattr(ri, "_file_draft", boom)

    out = asyncio.run(ri.run_unit(beat, write=False))
    assert out["findings"] == 1 and out["written"] == 0 and out["detail"][0]["title"] == "x"
