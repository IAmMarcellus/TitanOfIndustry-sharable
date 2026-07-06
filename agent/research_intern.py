"""The intern: a continuous, idle-time research *scout* for the TitanOfIndustry stack.

A standalone batch process that puts the otherwise-idle 3090 to work. While no real desk agent is
running, it researches configurable "beats" with the LOCAL worker (``qwen-codex`` via the proxy —
NEVER the cloud planner), and hands findings to the shared brain as **draft, low-confidence
candidates** for the cloud desk agents to vet. It mirrors ``memory_dream.py``'s conventions (env
knobs, a bounded local-only ``_llm``, ``_safe`` stage isolation, ``python -m agent.research_intern``)
and reuses the stack's components rather than reimplementing them:

    codebase beats  -> opencode_run (the executor, on local qwen-codex) over SCOPED subdirs
    web beats       -> httpx fetch + ONE local-LLM summarize call (no agent tool-loop)
    synthesis beats -> recall_for + ONE local-LLM call (connect prior memories into ideas)

Handoff is twofold: ``remember_for`` (company-scoped, tagged ``intern,candidate,draft`` so desk
agents' ``recall`` surfaces them and the dream pass decays unactioned noise) and an OPTIONAL Paperclip
backlog draft (``POST /api/companies/:id/issues``, unassigned, ``backlog``/``low``).

Guardrail — a SCOUT, never a decider (see ../CLAUDE.md): local model only; nothing is auto-assigned
or auto-actioned; every unit yields the GPU when the stack is busy; findings are confidence-floored,
deduped, and capped per unit. Multi-step autonomous research belongs to an OpenSage agent, not this
glue — the loop stays one beat -> one analysis -> write.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import litellm

from . import memory  # reuse the shared Neo4j driver + remember_for/recall_for/GLOBAL/close


# ---- config (env; config-over-code — all knobs live in .env, like DREAM_*) ----------------------
def _env(name: str, default: Any, cast: Any) -> Any:
    """Read an env var, returning ``default`` when unset/blank or when ``cast`` rejects it."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return cast(v.strip())
    except (ValueError, TypeError):
        return default


def _envb(name: str, default: bool) -> bool:
    return _env(name, default, lambda s: s.lower() in ("1", "true", "yes", "on"))


def _envf(name: str, default: float) -> float:
    return _env(name, default, float)


def _envi(name: str, default: int) -> int:
    return _env(name, default, int)


def _envopt_int(name: str) -> int | None:
    """Optional int: ``None`` when unset/blank (lets a knob mean 'feature off')."""
    return _env(name, None, int)


# scope + beats
_COMPANY = os.environ.get("INTERN_COMPANY") or memory.GLOBAL  # Paperclip company candidates scope to
_DEFAULT_BEATS_FILE = Path(__file__).resolve().parent / "intern_beats.toml"
_BEATS_FILE = os.environ.get("INTERN_BEATS_FILE") or str(_DEFAULT_BEATS_FILE)
_DEFAULT_REPO = os.environ.get("INTERN_DEFAULT_REPO", "~/Projects/AlgoCryptoTradingBot")

# cadence / idle
_LOOP_INTERVAL_S = _envf("INTERN_LOOP_INTERVAL_S", 900)
_MAX_UNITS_PER_CYCLE = _envi("INTERN_MAX_UNITS_PER_CYCLE", 1)
_UNIT_TIMEOUT_S = _envf("INTERN_UNIT_TIMEOUT_S", 600)
_BUSY_BACKOFF_S = _envf("INTERN_BUSY_BACKOFF_S", 120)
_GPU_BUSY_UTIL = _envopt_int("INTERN_GPU_BUSY_UTIL")  # nvidia-smi %; None = skip the GPU check
_PAUSE_FILE = Path(os.environ.get("INTERN_PAUSE_FILE") or "~/.cache/titanofindustry/intern.pause").expanduser()

# local model (LOCAL worker via the proxy — NEVER the planner)
_ENABLE_LLM = _envb("INTERN_ENABLE_LLM", True)
_LLM_MODEL = os.environ.get("INTERN_LLM_MODEL", "openai/qwen-codex")
_LLM_BASE_URL = os.environ.get("INTERN_LLM_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL", ""
)
_LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_LLM_MAX_TOKENS = _envi("INTERN_LLM_MAX_TOKENS", 1000)
_LLM_TIMEOUT_S = _envf("INTERN_LLM_TIMEOUT_S", 120)

# codebase beats (reuse opencode_run). The executor's own timeout is read from OPENCODE_TIMEOUT_S at
# import time, so we set it well below opencode's 900s default BEFORE importing opencode_tool — a
# scoped research read that runs long is a scoping bug, not work to wait on.
_OPENCODE_TIMEOUT_S = _envi("INTERN_OPENCODE_TIMEOUT_S", 300)
_OPENCODE_MODEL = os.environ.get("INTERN_OPENCODE_MODEL", "local-vllm/qwen-codex")

# web beats (outbound READS of public pages — no proprietary code leaves the box; opt-in)
_ENABLE_WEB = _envb("INTERN_ENABLE_WEB", False)
_WEB_TIMEOUT_S = _envf("INTERN_WEB_TIMEOUT_S", 20)
_WEB_MAX_URLS = _envi("INTERN_WEB_MAX_URLS", 5)
_WEB_MAX_CHARS = _envi("INTERN_WEB_MAX_CHARS", 12000)
_WEB_USER_AGENT = os.environ.get("INTERN_WEB_USER_AGENT", "titanofindustry-intern/1.0")
_SEARCH_URL = os.environ.get("INTERN_SEARCH_URL", "")
_SEARCH_API_KEY = os.environ.get("INTERN_SEARCH_API_KEY", "")

# handoff: memory thresholds + dedup
_WRITE_MIN_CONFIDENCE = _envf("INTERN_WRITE_MIN_CONFIDENCE", 0.30)
_MAX_FINDINGS_PER_UNIT = _envi("INTERN_MAX_FINDINGS_PER_UNIT", 3)
_DEDUP_RECALL_K = _envi("INTERN_DEDUP_RECALL_K", 5)
_DEDUP_OVERLAP = _envf("INTERN_DEDUP_OVERLAP", 0.7)

# handoff: Paperclip drafts (opt-in). Loopback local_trusted needs no token; the knob is for non-trusted setups.
_ENABLE_DRAFTS = _envb("INTERN_ENABLE_DRAFTS", False)
_PAPERCLIP_BASE_URL = os.environ.get("INTERN_PAPERCLIP_BASE_URL") or os.environ.get(
    "PAPERCLIP_BASE_URL", ""
)
_PAPERCLIP_TOKEN = os.environ.get("INTERN_PAPERCLIP_TOKEN", "")
_PAPERCLIP_TIMEOUT_S = _envf("INTERN_PAPERCLIP_TIMEOUT_S", 30)
_DRAFT_STATUS = os.environ.get("INTERN_DRAFT_STATUS", "backlog")
_DRAFT_PRIORITY = os.environ.get("INTERN_DRAFT_PRIORITY", "low")
_DRAFT_MIN_CONFIDENCE = _envf("INTERN_DRAFT_MIN_CONFIDENCE", 0.60)
_DRAFT_PARENT_ISSUE = os.environ.get("INTERN_DRAFT_PARENT_ISSUE", "")

# handoff: attribute drafts to the "Intern" roster agent via a local-agent JWT (optional). When
# INTERN_AGENT_ID + the JWT secret are set, drafts are POSTed AS the agent so they carry
# createdByAgentId; otherwise we fall back to the admin/local_trusted actor (drafts still land,
# just unattributed). Mirrors Paperclip's server/src/agent-auth-jwt.ts.
_AGENT_ID = os.environ.get("INTERN_AGENT_ID", "")
_AGENT_JWT_SECRET = os.environ.get("PAPERCLIP_AGENT_JWT_SECRET", "") or os.environ.get("BETTER_AUTH_SECRET", "")
_AGENT_ADAPTER_TYPE = os.environ.get("INTERN_AGENT_ADAPTER_TYPE", "process")
_AGENT_JWT_TTL_S = _envi("INTERN_AGENT_JWT_TTL_S", 300)

# opencode_tool captures OPENCODE_TIMEOUT_S at import — set the intern's bound first, then import.
os.environ.setdefault("OPENCODE_TIMEOUT_S", str(_OPENCODE_TIMEOUT_S))
from .opencode_tool import opencode_run  # noqa: E402  (must follow the timeout default above)


# ---- data shapes --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Beat:
    """One research target. ``type`` selects the runner; the rest is config-over-code from the TOML."""

    name: str
    type: str  # codebase | web | synthesis
    prompt: str = ""
    enabled: bool = True
    weight: float = 1.0
    tags: str = ""
    min_interval_s: float = 0.0
    dir: str = ""  # codebase: scoped subdir of the trading-bot repo
    files: tuple[str, ...] = ()  # codebase: exact files named into the prompt to bound the read
    urls: tuple[str, ...] = ()  # web
    query: str = ""  # web: optional search query (only used if INTERN_SEARCH_URL is set)
    max_chars: int = 0  # web: truncate fetched text (0 -> INTERN_WEB_MAX_CHARS)
    recall_query: str = ""  # synthesis: what to pull from memory
    recall_k: int = 8  # synthesis


@dataclass(frozen=True)
class Finding:
    """One candidate the scout surfaced — draft/low-confidence by construction."""

    title: str  # short imperative; draft-issue title + dedup key
    text: str  # self-contained candidate body stored to memory
    kind: str  # bug|edge_case|risky_param|market_signal|strategy_idea|...
    confidence: float  # 0..1
    evidence: tuple[str, ...] = ()  # file:line refs (codebase) or URLs (web)


# ---- LLM primitive (the ONLY model call here; local worker, bounded, degradable) ----------------
async def _llm(system: str, user: str) -> str | None:
    """One bounded local-model completion via the proxy. Returns ``None`` on any failure or when
    ``INTERN_ENABLE_LLM=0`` — callers treat ``None`` as 'no findings'. Routes to ``qwen-codex``
    (LOCAL), never the planner."""
    if not _ENABLE_LLM:
        return None
    try:
        resp = await litellm.acompletion(
            model=_LLM_MODEL,
            api_base=_LLM_BASE_URL,
            api_key=_LLM_API_KEY,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=_LLM_MAX_TOKENS,
            timeout=_LLM_TIMEOUT_S,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def _parse_json(raw: str | None) -> Any:
    """Best-effort JSON parse of a model reply: strip ``` fences, then fall back to slicing the
    outermost array/object. Returns ``None`` if nothing parses. (Copied from memory_dream.)"""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1 :] if nl != -1 else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = s.find(opener), s.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except Exception:
                pass
    return None


def _clamp01(v: Any, default: float = 0.5) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _parse_findings(raw: str | None, default_evidence: list[str]) -> list[Finding]:
    """Parse a model reply into Findings: enforce the JSON-array contract, clamp confidence, drop
    anything below the write floor, and cap the count (anti-spam at the source)."""
    arr = _parse_json(raw)
    if not isinstance(arr, list):
        return []
    out: list[Finding] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        text = str(item.get("text") or "").strip()
        if not title or not text:
            continue
        conf = _clamp01(item.get("confidence"))
        if conf < _WRITE_MIN_CONFIDENCE:
            continue
        ev_raw = item.get("evidence")
        evidence = tuple(str(e).strip() for e in ev_raw if str(e).strip()) if isinstance(ev_raw, list) else ()
        if not evidence and default_evidence:
            evidence = tuple(default_evidence)
        out.append(
            Finding(
                title=title[:200],
                text=text,
                kind=str(item.get("kind") or "note").strip()[:40],
                confidence=conf,
                evidence=evidence,
            )
        )
        if len(out) >= _MAX_FINDINGS_PER_UNIT:
            break
    return out


# ---- the findings contract (shared by web/synthesis system prompts) -----------------------------
def _findings_instruction(kinds: str) -> str:
    return (
        f"Output ONLY a JSON array (no prose, no code fences) of 0-{_MAX_FINDINGS_PER_UNIT} objects: "
        '{"title": str (short imperative), "text": str (the finding, self-contained), "kind": one of '
        f"{kinds}, " '"confidence": float 0-1, "evidence": [str]}. '
        "Each finding must be concrete, self-contained, and falsifiable. If nothing solid, output []."
    )


_WEB_SYS = (
    "You are a research scout for a crypto trading bot. From the fetched web pages, extract concrete, "
    "time-stamped market/news/competitor/regulatory signals relevant to such a bot. Ignore generic "
    "price commentary. " + _findings_instruction("market_signal|competitor|regulation|risk")
)
_SYNTH_SYS = (
    "You are a research scout. Connect MULTIPLE of the given prior memories into higher-level, "
    "TESTABLE ideas or hardening hypotheses — not a restatement of any single memory. "
    + _findings_instruction("strategy_idea|hardening|hypothesis")
)
_CODEBASE_CONTRACT = (
    "Constraints: read ONLY {files}. This is a READ-ONLY review — do NOT create, edit, or delete any "
    "file, and do NOT run anything that changes files or state. When done, "
    + _findings_instruction("bug|edge_case|risky_param|perf|strategy_idea")
)


# ---- execution modes ----------------------------------------------------------------------------
def _codebase_prompt(beat: Beat) -> str:
    # .replace (not .format): the appended findings contract contains literal JSON braces.
    files_list = ", ".join(beat.files) if beat.files else "the source files in this directory"
    body = beat.prompt.replace("{files}", files_list)
    return f"{body}\n\n{_CODEBASE_CONTRACT.replace('{files}', files_list)}"


async def _run_codebase_beat(beat: Beat) -> list[Finding]:
    """Scoped read-only review via the executor (local qwen-codex). Scope is enforced by the named
    files in the prompt + the bounded opencode timeout — NOT by ``dir`` (opencode resolves the git
    root, so ``dir`` alone won't wall off the repo; that's the 900s-timeout trap)."""
    work = Path(beat.dir or _DEFAULT_REPO).expanduser()
    if not work.is_dir():
        return []
    res = await opencode_run(task=_codebase_prompt(beat), cwd=str(work), model=_OPENCODE_MODEL)
    if not res.get("success"):
        return []
    return _parse_findings(res.get("stdout", ""), default_evidence=[])


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Minimal tag-strip (no heavy HTML lib): drop script/style, strip tags, collapse whitespace."""
    text = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


async def _search(query: str) -> list[str]:
    """Optional thin search: GET INTERN_SEARCH_URL?q=... and extract URLs from the JSON. Off (──> [])
    unless INTERN_SEARCH_URL is set; defensive about the response shape."""
    if not _SEARCH_URL or not query:
        return []
    headers = {"accept": "application/json"}
    if _SEARCH_API_KEY:
        headers["authorization"] = f"Bearer {_SEARCH_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=_WEB_TIMEOUT_S, headers=headers) as client:
            resp = await client.get(_SEARCH_URL, params={"q": query})
        data = resp.json()
    except Exception:
        return []
    urls: list[str] = []
    items = data.get("results") if isinstance(data, dict) else data
    if isinstance(items, list):
        for it in items:
            url = it.get("url") or it.get("link") if isinstance(it, dict) else (it if isinstance(it, str) else None)
            if isinstance(url, str) and url.startswith("http"):
                urls.append(url)
    return urls


def _web_user(beat: Beat, docs: list[tuple[str, str]]) -> str:
    parts = "\n\n".join(f"[Source: {u}]\n{txt}" for u, txt in docs)
    return f"{beat.prompt}\n\nFetched pages:\n{parts}\n\nReturn the JSON array of findings."


async def _run_web_beat(beat: Beat) -> list[Finding]:
    """Fetch the configured URLs (+ optional search) and summarize in ONE local-LLM call. Fetch-then-
    summarize only — no agent tool-loop, no headless browser. Off by default."""
    if not _ENABLE_WEB:
        return []
    urls = list(beat.urls) + (await _search(beat.query) if beat.query else [])
    docs: list[tuple[str, str]] = []
    cap = beat.max_chars or _WEB_MAX_CHARS
    async with httpx.AsyncClient(
        timeout=_WEB_TIMEOUT_S, headers={"user-agent": _WEB_USER_AGENT}, follow_redirects=True
    ) as client:
        for u in urls[:_WEB_MAX_URLS]:
            try:
                r = await client.get(u)
                docs.append((u, _strip_html(r.text)[:cap]))
            except Exception:
                continue
    if not docs:
        return []
    raw = await _llm(_WEB_SYS, _web_user(beat, docs))
    return _parse_findings(raw, default_evidence=[u for u, _ in docs])


def _synth_user(beat: Beat, rows: list[dict[str, Any]]) -> str:
    lines = "\n".join(f"{i}. {r.get('text', '')}" for i, r in enumerate(rows))
    return f"{beat.prompt}\n\nMemories:\n{lines}\n\nReturn the JSON array of ideas."


async def _run_synthesis_beat(beat: Beat) -> list[Finding]:
    """Pull prior memories and ask the local model to connect them into testable ideas (ONE call)."""
    res = await memory.recall_for(_COMPANY, beat.recall_query, k=beat.recall_k, agent=None)
    rows = res.get("results") or []
    if not rows:
        return []
    raw = await _llm(_SYNTH_SYS, _synth_user(beat, rows))
    return _parse_findings(raw, default_evidence=[])


_RUNNERS = {"codebase": _run_codebase_beat, "web": _run_web_beat, "synthesis": _run_synthesis_beat}


# ---- handoff: memory candidates + Paperclip drafts ----------------------------------------------
def _finding_tags(beat: Beat, finding: Finding) -> str:
    parts = ["intern", "research", "candidate", "draft", beat.name, finding.kind]
    if beat.tags:
        parts.append(beat.tags)  # may itself be comma-separated; joined verbatim
    return ",".join(p for p in parts if p)


def _finding_text(beat: Beat, finding: Finding) -> str:
    text = f"[intern draft · {beat.name} · {finding.kind} · conf={finding.confidence:.2f}] {finding.text}"
    if finding.evidence:
        text += f" (evidence: {', '.join(finding.evidence[:5])})"
    return text


def _overlap(a: str, b: str) -> float:
    """Jaccard token overlap (cheap near-dup signal)."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def _already_seen(beat: Beat, finding: Finding) -> bool:
    """Near-dup gate before writing: a recalled candidate sharing this beat's tag and high title
    overlap means we've effectively surfaced this already. (Exact dups are a free scope_key no-op.)"""
    res = await memory.recall_for(_COMPANY, finding.title, k=_DEDUP_RECALL_K)
    if not res.get("success"):
        return False
    for r in res.get("results", []):
        tags = r.get("tags") or ""
        if beat.name in tags and _overlap(finding.title, r.get("text", "")) >= _DEDUP_OVERLAP:
            return True
    return False


async def _write_finding(beat: Beat, finding: Finding) -> str:
    """Store the candidate at company scope (agent="" → company-wide visible), provenance in
    text+tags. Returns 'written' or 'error'."""
    res = await memory.remember_for(_COMPANY, _finding_text(beat, finding), tags=_finding_tags(beat, finding), agent="")
    return "written" if res.get("success") else "error"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _mint_agent_jwt() -> str | None:
    """Mint a local-agent JWT so a draft is attributed to the Intern roster agent (createdByAgentId).
    Mirrors Paperclip's ``server/src/agent-auth-jwt.ts`` (HS256 with a per-company derived signing
    key). Returns ``None`` when unconfigured (no ``INTERN_AGENT_ID`` / secret / real company) so the
    caller falls back to the admin actor. ``run_id`` is a throwaway UUID — the scout has no heartbeat
    run, and Paperclip's ``logActivity`` coerces a dangling run id to null (see activity-log.ts), so
    no real run is required. Couples to Paperclip's JWT scheme: re-verify on a Paperclip auth bump."""
    if not (_AGENT_ID and _AGENT_JWT_SECRET and _COMPANY and _COMPANY != memory.GLOBAL):
        return None
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = _b64url(
        json.dumps(
            {
                "sub": _AGENT_ID,
                "company_id": _COMPANY,
                "adapter_type": _AGENT_ADAPTER_TYPE,
                "run_id": str(uuid.uuid4()),
                "iat": now,
                "exp": now + _AGENT_JWT_TTL_S,
            },
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{claims}"
    signing_key = hmac.new(_AGENT_JWT_SECRET.encode(), f"jwt:{_COMPANY}".encode(), hashlib.sha256).hexdigest()
    sig = _b64url(hmac.new(signing_key.encode(), signing_input.encode(), hashlib.sha256).digest())
    return f"{signing_input}.{sig}"


def _draft_payload(beat: Beat, finding: Finding) -> dict[str, Any]:
    evidence = "\n".join(f"- {e}" for e in finding.evidence[:10])
    desc = (
        "> ⚠️ **INTERN DRAFT — needs vetting.** Auto-generated by the local research scout "
        "(qwen-codex). Do NOT action without human/manager review.\n\n"
        f"**Beat:** `{beat.name}` · **kind:** `{finding.kind}` · **confidence:** {finding.confidence:.2f}\n\n"
        f"{finding.text}\n"
        + (f"\n**Evidence:**\n{evidence}\n" if finding.evidence else "")
    )
    payload: dict[str, Any] = {
        "title": f"[intern] {finding.title}"[:200],
        "description": desc,
        "status": _DRAFT_STATUS,
        "priority": _DRAFT_PRIORITY,
    }
    if _DRAFT_PARENT_ISSUE:
        payload["parentId"] = _DRAFT_PARENT_ISSUE
    return payload


async def _post_draft(payload: dict[str, Any], headers: dict[str, str]) -> str | None:
    """POST one draft and return the new issue id (or ``None`` on any non-2xx / parse failure)."""
    url = f"{_PAPERCLIP_BASE_URL.rstrip('/')}/api/companies/{_COMPANY}/issues"
    try:
        async with httpx.AsyncClient(timeout=_PAPERCLIP_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload, headers={"accept": "application/json", **headers})
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, dict):
        issue = data.get("issue") if isinstance(data.get("issue"), dict) else data
        ident = issue.get("id")
        return str(ident) if ident else None
    return None


async def _file_draft(beat: Beat, finding: Finding) -> str | None:
    """Create an UNASSIGNED low-priority Paperclip backlog draft for triage. Returns the issue id, or
    ``None`` (drafts disabled / below threshold / no real company / API error). GUARDRAIL: backlog +
    low + no assignee — never handed to an executing agent. Attributed to the Intern roster agent via
    a local-agent JWT when configured; falls back to the admin/local_trusted actor so a missing or
    stale secret never blocks filing (and, with the activity-log run-id fix, never duplicates: an
    agent-auth failure rejects BEFORE the issue is created, so the fallback POST is the only insert)."""
    if not _ENABLE_DRAFTS or finding.confidence < _DRAFT_MIN_CONFIDENCE:
        return None
    if not _COMPANY or _COMPANY == memory.GLOBAL:
        return None  # need a real company to file against
    payload = _draft_payload(beat, finding)
    token = _mint_agent_jwt()
    if token:
        issue_id = await _post_draft(payload, {"authorization": f"Bearer {token}"})
        if issue_id:
            return issue_id  # attributed to the Intern agent
    admin_headers = {"authorization": f"Bearer {_PAPERCLIP_TOKEN}"} if _PAPERCLIP_TOKEN else {}
    return await _post_draft(payload, admin_headers)


# ---- idle gate ----------------------------------------------------------------------------------
async def _paperclip_busy() -> bool:
    """Busy if the <REDACTED_COMPANY> company has any queued/running agent run. Unreachable / no real company ->
    not-busy-via-this-signal (never wedge the loop on a Paperclip outage)."""
    if not _COMPANY or _COMPANY == memory.GLOBAL:
        return False
    url = f"{_PAPERCLIP_BASE_URL.rstrip('/')}/api/companies/{_COMPANY}/live-runs"
    headers = {"accept": "application/json"}
    if _PAPERCLIP_TOKEN:
        headers["authorization"] = f"Bearer {_PAPERCLIP_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=_PAPERCLIP_TIMEOUT_S) as client:
            resp = await client.get(url, params={"minCount": 0, "limit": 1}, headers=headers)
        if resp.status_code >= 400:
            return False
        data = resp.json()
    except Exception:
        return False
    runs = data if isinstance(data, list) else (data.get("runs") if isinstance(data, dict) else None)
    return bool(runs)


async def _gpu_busy() -> bool:
    """Busy if any GPU's utilization >= INTERN_GPU_BUSY_UTIL (catches OTHER companies + Claude Code +
    Windows-side Ollama via the hardware counter). Off unless the threshold is set; nvidia-smi
    missing/erroring -> not busy."""
    if _GPU_BUSY_UTIL is None:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except Exception:
        return False
    utils = [int(x) for x in out.decode("utf-8", "replace").split() if x.strip().isdigit()]
    return any(u >= _GPU_BUSY_UTIL for u in utils)


async def _stack_busy() -> bool:
    return await _paperclip_busy() or await _gpu_busy()


def _paused() -> bool:
    return _PAUSE_FILE.exists()


# ---- beat selection -----------------------------------------------------------------------------
_last_run: dict[str, float] = {}  # beat name -> monotonic time of last run (cooldown state)


def _load_beats() -> list[Beat]:
    """Parse the beats TOML (stdlib ``tomllib``). Returns enabled beats of a known type; any malformed
    entry or a missing file degrades to [] rather than crashing the loop."""
    import tomllib

    path = Path(_BEATS_FILE).expanduser()
    if not path.is_file():
        return []
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return []
    out: list[Beat] = []
    for raw in data.get("beat", []):
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        try:
            beat = Beat(
                name=str(raw["name"]),
                type=str(raw.get("type", "")),
                prompt=str(raw.get("prompt", "")),
                enabled=bool(raw.get("enabled", True)),
                weight=float(raw.get("weight", 1.0)),
                tags=str(raw.get("tags", "")),
                min_interval_s=float(raw.get("min_interval_s", 0)),
                dir=str(raw.get("dir", "")),
                files=tuple(str(x) for x in raw.get("files", [])),
                urls=tuple(str(x) for x in raw.get("urls", [])),
                query=str(raw.get("query", "")),
                max_chars=int(raw.get("max_chars", 0)),
                recall_query=str(raw.get("recall_query", "")),
                recall_k=int(raw.get("recall_k", 8)),
            )
        except (KeyError, ValueError, TypeError):
            continue
        if beat.enabled and beat.type in _RUNNERS:
            out.append(beat)
    return out


def _select_beat(beats: list[Beat]) -> Beat | None:
    """Pick the eligible (past its cooldown) beat that's been idle longest, weighted. ``None`` when all
    beats are still cooling down."""
    now = time.monotonic()
    eligible = [b for b in beats if (now - _last_run.get(b.name, 0.0)) >= b.min_interval_s or b.name not in _last_run]
    if not eligible:
        return None
    return max(eligible, key=lambda b: (now - _last_run.get(b.name, 0.0)) * max(b.weight, 0.0001))


# ---- orchestration ------------------------------------------------------------------------------
async def run_unit(beat: Beat, write: bool = True) -> dict[str, Any]:
    """Run one beat: produce findings, then (when ``write``) dedup → remember → optional draft.
    Returns a JSON-able stats dict. Marks the beat's cooldown from the start of the run."""
    _last_run[beat.name] = time.monotonic()
    t0 = time.monotonic()
    runner = _RUNNERS.get(beat.type)
    if runner is None:
        return {"beat": beat.name, "type": beat.type, "error": "unknown beat type"}
    findings = await runner(beat)
    written = drafted = skipped = 0
    drafts: list[str] = []
    if write:
        for f in findings:
            if await _already_seen(beat, f):
                skipped += 1
                continue
            if await _write_finding(beat, f) != "written":
                continue
            written += 1
            issue_id = await _file_draft(beat, f)
            if issue_id:
                drafted += 1
                drafts.append(issue_id)
    return {
        "beat": beat.name,
        "type": beat.type,
        "findings": len(findings),
        "detail": [{"title": f.title, "kind": f.kind, "confidence": round(f.confidence, 2)} for f in findings],
        "written": written,
        "drafted": drafted,
        "skipped_dup": skipped,
        "drafts": drafts,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }


async def run_cycle(max_units: int = 1, write: bool = True) -> dict[str, Any]:
    """Run up to ``max_units`` beats once (for ``--once`` / the Makefile dry-run). Does NOT consult the
    idle gate — that's the continuous loop's job."""
    units: list[dict[str, Any]] = []
    for _ in range(max(1, max_units)):
        beat = _select_beat(_load_beats())
        if beat is None:
            units.append({"skipped": "no_eligible_beat"})
            break
        units.append(await run_unit(beat, write=write))
    return {"success": True, "company": _COMPANY, "write": write, "units": units}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_stop = asyncio.Event()


async def _sleep_or_stop(secs: float) -> None:
    """Sleep, but wake immediately on SIGTERM/SIGINT so systemd stop drains promptly."""
    try:
        await asyncio.wait_for(_stop.wait(), timeout=secs)
    except asyncio.TimeoutError:
        pass


async def loop() -> None:
    """The continuous-while-idle scout loop. Yields the GPU whenever the stack is busy or paused; one
    bounded unit per tick; clean SIGTERM drain."""
    running = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            running.add_signal_handler(sig, _stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    print(json.dumps({"ts": _now_iso(), "event": "intern_start", "company": _COMPANY}), flush=True)
    while not _stop.is_set():
        if _paused() or await _stack_busy():
            await _sleep_or_stop(_BUSY_BACKOFF_S)
            continue
        beat = _select_beat(_load_beats())
        if beat is None:
            await _sleep_or_stop(_LOOP_INTERVAL_S)
            continue
        try:
            stats = await asyncio.wait_for(run_unit(beat), timeout=_UNIT_TIMEOUT_S)
        except asyncio.TimeoutError:
            stats = {"beat": beat.name, "error": "unit_timeout"}
        except Exception as exc:  # noqa: BLE001 — one bad unit must never kill the loop
            stats = {"beat": beat.name, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({"ts": _now_iso(), **stats}), flush=True)
        await _sleep_or_stop(_LOOP_INTERVAL_S)
    await memory.close()
    print(json.dumps({"ts": _now_iso(), "event": "intern_stop"}), flush=True)


async def _once(write: bool) -> dict[str, Any]:
    """One cycle + driver close in a SINGLE event loop (the Neo4j async driver is loop-bound, so
    closing it from a second asyncio.run would raise 'Event loop is closed')."""
    try:
        return await run_cycle(_MAX_UNITS_PER_CYCLE, write=write)
    finally:
        await memory.close()


if __name__ == "__main__":
    if "--once" in sys.argv:
        _write = "--dry-run" not in sys.argv
        print(json.dumps(asyncio.run(_once(_write)), indent=2))
    else:
        asyncio.run(loop())
