"""Dreaming: offline memory consolidation for TitanOfIndustry (Neo4j Community + free GDS + local LLM).

A batch pass over the shared ``:Memory`` store, run AFTER ``make memory-graph`` via ``make
memory-dream``. It mirrors ``memory_graph.py``'s conventions (reuse ``memory._get_driver()``,
idempotent re-derivation, graceful no-ops, ``python -m agent.memory_dream``) and turns the flat,
ever-growing store into one that gets *cleaner and smarter* over time:

    Stage 0  ensure indexes + backfill missing embeddings        DETERMINISTIC
    Stage 1  Ebbinghaus decay scoring (recency+frequency+import)  DETERMINISTIC (Cypher)
    Stage 2  semantic dedup / merge (near-duplicates)             DETERMINISTIC (embeddings)
    Stage 3  reflection / synthesis per topic cluster             LLM (LOCAL qwen-codex, bounded)
    Stage 4  contradiction detection + temporal versioning        LLM (LOCAL, top-N pairs only)
    Stage 5  forgetting / archival (soft, reversible)             DETERMINISTIC (Cypher; OFF by default)
    Stage 6  offline Personalized-PageRank enrichment (HippoRAG2)  DETERMINISTIC (graph; opt-in)

Design rules (see ../CLAUDE.md "Hard rules"):
- Every LLM stage uses the LOCAL worker (``qwen-codex``) via the LiteLLM proxy — NEVER the cloud
  planner (the token-metered bottleneck). ``DREAM_ENABLE_LLM=0`` gives a pure-deterministic pass.
- Soft-delete only: "forgetting" sets ``m.archived = true`` (excluded from recall) — never DELETE.
  Reversible by clearing the flag. The shared ``global`` tier and user-sourced facts are never touched.
- Multi-tenant safe: all merges/contradictions stay within one ``company``.
- Graceful: any stage no-ops (and reports it in the stats ``skipped`` field) when its inputs aren't
  ready (no embeddings, no topic graph) or its endpoint is down. One failing stage never aborts the
  pass — each runs in its own session and is wrapped so its exception is captured, not propagated.

Research lineage: Letta sleep-time agents (offline consolidation on a cheap/local model), Generative
Agents (reflection → higher-level insights), HippoRAG/HippoRAG2 (graph + Personalized PageRank
associative recall), MemoryBank (Ebbinghaus forgetting curve), and the 2026 "Memory for Autonomous
LLM Agents" survey (episodic→semantic, contradiction + temporal versioning, bounded forgetting).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from typing import Any

import litellm

from . import memory  # reuse the shared Neo4j driver + _embed/_store/_ensure_indexes at runtime

_MS_PER_DAY = 86_400_000


# ---- config (env; config-over-code — all knobs live in .env) ------------------------------------
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


# LLM (LOCAL worker via the proxy — never the planner)
_ENABLE_LLM = _envb("DREAM_ENABLE_LLM", True)
_LLM_MODEL = os.environ.get("DREAM_LLM_MODEL", "openai/qwen-codex")
_LLM_BASE_URL = os.environ.get("DREAM_LLM_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL", ""
)
_LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_LLM_MAX_TOKENS = _envi("DREAM_LLM_MAX_TOKENS", 800)
_LLM_TIMEOUT_S = _envf("DREAM_LLM_TIMEOUT_S", 120)

# Stage 0 — embedding backfill
_BACKFILL_BATCH = _envi("DREAM_BACKFILL_BATCH", 64)

# Stage 1 — Ebbinghaus decay
_HALF_LIFE_DAYS = _envf("DREAM_DECAY_HALF_LIFE_DAYS", 30)
_ACCESS_WEIGHT = _envf("DREAM_ACCESS_WEIGHT", 0.5)
_IMPORTANCE_WEIGHT = _envf("DREAM_IMPORTANCE_WEIGHT", 2.0)

# Stage 2 — semantic dedup
_DEDUP_SIM = _envf("DREAM_DEDUP_SIM", 0.95)
_DEDUP_MAX = _envi("DREAM_DEDUP_MAX", 200)
_SCAN_CAP = _envi("DREAM_SCAN_CAP", 2000)  # max nodes a similarity stage scans per run

# Stage 3 — reflection / synthesis
_REFLECT_TOPIC_IMPORTANCE = _envf("DREAM_REFLECT_TOPIC_IMPORTANCE", 1.0)
_REFLECT_MIN_MEMBERS = _envi("DREAM_REFLECT_MIN_MEMBERS", 3)
_REFLECT_MAX_MEMBERS = _envi("DREAM_REFLECT_MAX_MEMBERS", 20)
_MAX_REFLECTIONS_PER_RUN = _envi("DREAM_MAX_REFLECTIONS_PER_RUN", 5)
_REFLECT_RESYNTH_FRAC = _envf("DREAM_REFLECT_RESYNTH_FRAC", 0.8)

# Stage 4 — contradiction
_CONTRADICT_SIM = _envf("DREAM_CONTRADICT_SIM", 0.85)
_CONTRADICT_MAX_PAIRS = _envi("DREAM_CONTRADICT_MAX_PAIRS", 10)

# Stage 5 — forgetting (soft, reversible, bounded; OFF until trusted)
_ENABLE_FORGET = _envb("DREAM_ENABLE_FORGET", False)
_FORGET_IMPORTANCE_MAX = _envf("DREAM_FORGET_IMPORTANCE_MAX", 0.15)
_FORGET_DECAY_MAX = _envf("DREAM_FORGET_DECAY_MAX", 0.2)
_FORGET_MIN_AGE_DAYS = _envf("DREAM_FORGET_MIN_AGE_DAYS", 14)
_FORGET_MAX_PER_RUN = _envi("DREAM_FORGET_MAX_PER_RUN", 20)

# Stage 6 — offline associative (PPR-style) enrichment (opt-in)
_PPR_ENABLED = _envb("DREAM_PPR_ENABLED", False)
_PPR_LIMIT = _envi("DREAM_PPR_LIMIT", 5)


# ---- LLM primitive (the ONLY model call in this module; local worker, bounded, degradable) -------
async def _llm(system: str, user: str) -> str | None:
    """One bounded local-model completion via the proxy. Returns ``None`` on any failure or when
    ``DREAM_ENABLE_LLM=0`` — every LLM stage treats ``None`` as "skip this item" so the pass degrades
    to deterministic-only without erroring. Routes to ``qwen-codex`` (LOCAL), never the planner."""
    if not _ENABLE_LLM:
        return None
    try:
        resp = await litellm.acompletion(
            model=_LLM_MODEL,
            api_base=_LLM_BASE_URL,
            api_key=_LLM_API_KEY,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=_LLM_MAX_TOKENS,
            timeout=_LLM_TIMEOUT_S,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def _parse_json(raw: str | None) -> Any:
    """Best-effort JSON parse of a model reply: strips ``` fences, then falls back to slicing the
    outermost array/object. Returns ``None`` if nothing parses (→ caller no-ops for that item)."""
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


def _rank_keep(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical (keep) vs duplicate (drop) by (importance, access_count, ts) descending."""
    ka = (a.get("imp", 0.0), a.get("ac", 0), a.get("ts") or 0)
    kb = (b.get("imp", 0.0), b.get("ac", 0), b.get("ts") or 0)
    return (a, b) if ka >= kb else (b, a)


# ---- Stage 0: embedding backfill ----------------------------------------------------------------
async def _backfill_embeddings() -> dict[str, Any]:
    """Embed any ``:Memory`` still missing a vector (e.g. all of them, the first time the CPU embedder
    comes online), so vector recall + the semantic stages see the whole store, not just new writes."""
    if not memory._EMBEDDING_MODEL:
        return {"backfilled": 0, "skipped": "embeddings_off"}
    write = (
        "UNWIND $rows AS row MATCH (m:Memory {scope_key: row.sk}) SET m.embedding = row.emb"
    )
    async with memory._get_driver().session() as session:
        res = await session.run(
            "MATCH (m:Memory) WHERE m.embedding IS NULL "
            "RETURN m.scope_key AS sk, m.text AS text"
        )
        todo = [dict(r) async for r in res]
        done = 0
        # Embed each chunk concurrently (the embed server queues them) + write it in one UNWIND,
        # so a cold backfill of the whole store is chunk-parallel, not one-at-a-time-sequential.
        for i in range(0, len(todo), _BACKFILL_BATCH):
            chunk = todo[i : i + _BACKFILL_BATCH]
            embs = await asyncio.gather(*(memory._embed(r["text"]) for r in chunk))
            rows = [{"sk": r["sk"], "emb": e} for r, e in zip(chunk, embs) if e is not None]
            if rows:
                await session.run(write, rows=rows)
                done += len(rows)
            if len(rows) < len(chunk):  # an embed came back None — endpoint down; stop rather than spin
                return {"backfilled": done, "skipped": "embed_unavailable"}
    return {"backfilled": done}


# ---- Stage 1: Ebbinghaus decay scoring ----------------------------------------------------------
async def _stage_decay() -> dict[str, Any]:
    """R = exp(-days_idle / strength); strength grows with access_count + importance, so frequently
    used / central memories decay slowly and stale-ignored ones approach 0. Pure Cypher (no LLM)."""
    async with memory._get_driver().session() as session:
        res = await session.run(
            "MATCH (m:Memory) WHERE (m.archived IS NULL OR m.archived = false) "
            "WITH m, (timestamp() - coalesce(m.last_accessed, m.ts)) / $ms_per_day AS days_idle, "
            "coalesce(m.access_count, 0) AS ac, coalesce(m.importance, 0.0) AS imp "
            "WITH m, days_idle, ($half_life * (1.0 + $access_w * ac + $importance_w * imp)) AS strength "
            "SET m.decay_score = exp(-days_idle / strength) "
            "RETURN count(m) AS scored",
            ms_per_day=float(_MS_PER_DAY),
            half_life=_HALF_LIFE_DAYS,
            access_w=_ACCESS_WEIGHT,
            importance_w=_IMPORTANCE_WEIGHT,
        )
        rec = await res.single()
    return {"scored": rec["scored"] if rec else 0}


# ---- Stage 2: semantic dedup / merge ------------------------------------------------------------
async def _stage_dedup() -> dict[str, Any]:
    """Archive near-duplicate memories (cosine >= DREAM_DEDUP_SIM, same company), keeping the richest
    as canonical and recording provenance. Needs embeddings — skips cleanly when they're off."""
    if not memory._EMBEDDING_MODEL:
        return {"merged": 0, "skipped": "embeddings_off"}
    archived: set[str] = set()
    merged = 0
    async with memory._get_driver().session() as session:
        res = await session.run(
            "MATCH (m:Memory) WHERE (m.archived IS NULL OR m.archived = false) "
            "AND m.embedding IS NOT NULL AND m.company IS NOT NULL "
            "RETURN m.scope_key AS sk, m.embedding AS emb, m.company AS company, "
            "coalesce(m.importance, 0.0) AS imp, coalesce(m.access_count, 0) AS ac, m.ts AS ts "
            "ORDER BY imp DESC LIMIT $cap",
            cap=_SCAN_CAP,
        )
        cands = [dict(r) async for r in res]
        for c in cands:
            if merged >= _DEDUP_MAX:
                break
            if c["sk"] in archived:
                continue
            nres = await session.run(
                f"CALL db.index.vector.queryNodes('{memory._VECTOR_INDEX}', 6, $vec) "
                "YIELD node, score "
                "WHERE node.scope_key <> $sk AND score >= $sim AND node.company = $company "
                "AND (node.archived IS NULL OR node.archived = false) "
                "RETURN node.scope_key AS sk, coalesce(node.importance, 0.0) AS imp, "
                "coalesce(node.access_count, 0) AS ac, node.ts AS ts",
                vec=c["emb"], sk=c["sk"], sim=_DEDUP_SIM, company=c["company"],
            )
            neighbors = [dict(r) async for r in nres]
            for n in neighbors:
                if merged >= _DEDUP_MAX or n["sk"] in archived:
                    continue
                keep, drop = _rank_keep(c, n)
                if keep["sk"] == drop["sk"] or drop["sk"] in archived:
                    continue
                await session.run(
                    "MATCH (keep:Memory {scope_key: $keep}), (drop:Memory {scope_key: $drop}) "
                    "SET drop.archived = true, drop.archived_reason = 'dedup', "
                    "keep.tags = CASE WHEN coalesce(drop.tags, '') = '' THEN keep.tags "
                    "WHEN coalesce(keep.tags, '') = '' THEN drop.tags "
                    "ELSE keep.tags + ',' + drop.tags END "
                    "MERGE (keep)-[:CONSOLIDATES {kind: 'dedup'}]->(drop)",
                    keep=keep["sk"], drop=drop["sk"],
                )
                archived.add(drop["sk"])
                merged += 1
    return {"merged": merged}


# ---- Stage 3: reflection / synthesis (Generative-Agents style) ----------------------------------
_REFLECT_SYS = (
    "You consolidate an AI engineering agent's episodic memories into durable, generalizable "
    "insights. Given numbered memories about one topic, output ONLY a JSON array (no prose, no code "
    'fences) of 0-5 objects: {"insight": str, "evidence_indices": [int], "confidence": float 0-1}. '
    "Each insight must be a higher-level, durable fact inferable from the memories (a recurring "
    "pattern, preference, convention, or stable conclusion) — NOT a restatement of a single memory "
    "and NOT speculation. evidence_indices list the memory numbers that support it. If nothing "
    "durable can be synthesized, return []."
)


def _reflect_user(members: list[dict[str, Any]]) -> str:
    lines = "\n".join(f"{i}. {m['text']}" for i, m in enumerate(members))
    return f"Memories:\n{lines}\n\nReturn the JSON array of insights."


def _reflect_tags(members: list[dict[str, Any]]) -> str:
    seen: list[str] = []
    for m in members:
        for t in (m.get("tags") or "").split(","):
            t = t.strip()
            if t and t not in seen:
                seen.append(t)
    if "reflection" not in seen:
        seen.append("reflection")
    return ",".join(seen[:12])


async def _stage_reflect() -> dict[str, Any]:
    """Per high-importance topic cluster, ask the LOCAL model for durable insights and store each as a
    new ``kind='reflection'`` memory linked to its evidence. Bounded by DREAM_MAX_REFLECTIONS_PER_RUN
    clusters (the hard token cap) and idempotent (skips clusters already synthesized with nothing
    new). No-ops if the topic graph isn't built or the model is unavailable."""
    created = 0
    llm_calls = 0
    async with memory._get_driver().session() as session:
        cres = await session.run(
            "MATCH (m:Memory) WHERE (m.archived IS NULL OR m.archived = false) "
            "AND m.topic IS NOT NULL AND m.company IS NOT NULL "
            "AND coalesce(m.kind, 'episodic') = 'episodic' "
            "WITH m.topic AS topic, m.company AS company, count(m) AS n, "
            "sum(coalesce(m.importance, 0.0)) AS imp "
            "WHERE n >= $min_members AND imp >= $topic_importance "
            "RETURN topic, company, n, imp ORDER BY imp DESC LIMIT $max_clusters",
            min_members=_REFLECT_MIN_MEMBERS,
            topic_importance=_REFLECT_TOPIC_IMPORTANCE,
            max_clusters=_MAX_REFLECTIONS_PER_RUN,
        )
        clusters = [dict(r) async for r in cres]
        if not clusters:
            return {"reflections_created": 0, "llm_calls": 0, "skipped": "no_eligible_clusters"}

        for cl in clusters:
            mres = await session.run(
                "MATCH (m:Memory) WHERE m.topic = $topic AND m.company = $company "
                "AND (m.archived IS NULL OR m.archived = false) "
                "AND coalesce(m.kind, 'episodic') = 'episodic' "
                "RETURN m.scope_key AS sk, m.text AS text, m.tags AS tags, coalesce(m.ts, 0) AS ts "
                "ORDER BY coalesce(m.importance, 0.0) DESC LIMIT $lim",
                topic=cl["topic"], company=cl["company"], lim=_REFLECT_MAX_MEMBERS,
            )
            members = [dict(r) async for r in mres]
            if len(members) < _REFLECT_MIN_MEMBERS:
                continue
            member_sks = [m["sk"] for m in members]
            newest_ts = max((m["ts"] or 0) for m in members)
            need = int(math.ceil(_REFLECT_RESYNTH_FRAC * len(members)))
            cov = await session.run(
                "MATCH (r:Memory) WHERE r.kind = 'reflection' AND r.company = $company "
                "AND r.consolidated_from IS NOT NULL "
                "WITH r, size([x IN r.consolidated_from WHERE x IN $sks]) AS overlap "
                "WHERE overlap >= $need AND coalesce(r.ts, 0) >= $newest "
                "RETURN count(r) AS c",
                company=cl["company"], sks=member_sks, need=need, newest=newest_ts,
            )
            crec = await cov.single()
            if crec and crec["c"] > 0:
                continue  # already synthesized; nothing new since

            raw = await _llm(_REFLECT_SYS, _reflect_user(members))
            llm_calls += 1
            insights = _parse_json(raw)
            if not isinstance(insights, list):
                continue
            tags = _reflect_tags(members)
            for ins in insights:
                if not isinstance(ins, dict):
                    continue
                text = (ins.get("insight") or "").strip()
                if not text:
                    continue
                idx = ins.get("evidence_indices") or []
                evidence = [
                    members[i]["sk"] for i in idx if isinstance(i, int) and 0 <= i < len(members)
                ] or member_sks
                stored = await memory._store(text, tags, cl["company"], "", kind="reflection")
                if not stored.get("success"):
                    continue
                conf = ins.get("confidence")
                conf = float(conf) if isinstance(conf, (int, float)) else None
                await session.run(
                    "MATCH (r:Memory {scope_key: $rsk}) "
                    "SET r.confidence = $conf, r.consolidated_from = $ev "
                    "WITH r UNWIND $ev AS esk MATCH (e:Memory {scope_key: esk}) "
                    "MERGE (r)-[:CONSOLIDATES {kind: 'reflection'}]->(e)",
                    rsk=stored["scope_key"], conf=conf, ev=evidence,
                )
                created += 1
    return {"reflections_created": created, "llm_calls": llm_calls}


# ---- Stage 4: contradiction detection + temporal versioning -------------------------------------
_CONTRADICT_SYS = (
    "You detect contradictions between two facts an AI engineering agent stored at different times. "
    'Output ONLY a JSON object (no prose, no code fences): {"contradict": bool, "supersedes": "A" | '
    '"B" | null}. They contradict only if both cannot be true at once (not merely different). If they '
    "contradict, set supersedes to the one that should WIN: prefer source 'user' over 'agent'; on a "
    "tie prefer the newer (larger ts). If they do not contradict, supersedes is null."
)


def _contradict_user(a: dict[str, Any], b: dict[str, Any]) -> str:
    return (
        f"A) text={a['text']!r} source={a['source']} kind={a['kind']} ts={a['ts']}\n"
        f"B) text={b['text']!r} source={b['source']} kind={b['kind']} ts={b['ts']}\n\n"
        "Return the JSON verdict."
    )


async def _stage_contradict() -> dict[str, Any]:
    """Find same-company, same-topic, high-similarity pairs (candidates for "same subject, opposite
    claim"), adjudicate only the top-N with the LOCAL model, and soft-archive the superseded one with
    a provenance edge. Never archives a user-sourced memory. Needs embeddings — skips when off."""
    if not memory._EMBEDDING_MODEL:
        return {"resolved": 0, "llm_calls": 0, "skipped": "embeddings_off"}
    pairs: dict[tuple[str, str], float] = {}
    async with memory._get_driver().session() as session:
        nres = await session.run(
            "MATCH (m:Memory) WHERE (m.archived IS NULL OR m.archived = false) "
            "AND m.embedding IS NOT NULL AND m.topic IS NOT NULL AND m.company IS NOT NULL "
            "RETURN m.scope_key AS sk, m.embedding AS emb, m.company AS company, m.topic AS topic "
            "ORDER BY coalesce(m.importance, 0.0) DESC LIMIT $cap",
            cap=_SCAN_CAP,
        )
        nodes = [dict(r) async for r in nres]
        for nd in nodes:
            qres = await session.run(
                f"CALL db.index.vector.queryNodes('{memory._VECTOR_INDEX}', 4, $vec) "
                "YIELD node, score "
                "WHERE node.scope_key <> $sk AND score >= $sim AND score < 0.999 "
                "AND node.company = $company AND node.topic = $topic "
                "AND (node.archived IS NULL OR node.archived = false) "
                "AND NOT EXISTS { (n:Memory {scope_key: $sk})-[:CONSOLIDATES]-(node) } "
                "RETURN node.scope_key AS sk, score",
                vec=nd["emb"], sk=nd["sk"], sim=_CONTRADICT_SIM,
                company=nd["company"], topic=nd["topic"],
            )
            for r in [dict(x) async for x in qres]:
                key = tuple(sorted((nd["sk"], r["sk"])))  # type: ignore[assignment]
                pairs[key] = max(pairs.get(key, 0.0), r["score"])

        top = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:_CONTRADICT_MAX_PAIRS]
        resolved = 0
        llm_calls = 0
        for (a_sk, b_sk), _score in top:
            pres = await session.run(
                "MATCH (m:Memory) WHERE m.scope_key IN [$a, $b] "
                "RETURN m.scope_key AS sk, m.text AS text, coalesce(m.source, 'agent') AS source, "
                "coalesce(m.kind, 'episodic') AS kind, coalesce(m.ts, 0) AS ts",
                a=a_sk, b=b_sk,
            )
            recs: dict[str, dict[str, Any]] = {}
            async for r in pres:
                recs[r["sk"]] = dict(r)
            a, b = recs.get(a_sk), recs.get(b_sk)
            if not a or not b:
                continue
            verdict = _parse_json(await _llm(_CONTRADICT_SYS, _contradict_user(a, b)))
            llm_calls += 1
            if not isinstance(verdict, dict) or not verdict.get("contradict"):
                continue
            sup = str(verdict.get("supersedes") or "").upper()
            if sup == "A":
                winner, loser = a, b
            elif sup == "B":
                winner, loser = b, a
            else:
                continue
            if loser["source"] == "user":  # never auto-forget a user-stated fact
                continue
            await session.run(
                "MATCH (w:Memory {scope_key: $w}), (l:Memory {scope_key: $l}) "
                "SET l.archived = true, l.archived_reason = 'superseded' "
                "MERGE (w)-[:CONSOLIDATES {kind: 'supersedes'}]->(l)",
                w=winner["sk"], l=loser["sk"],
            )
            resolved += 1
    return {"resolved": resolved, "llm_calls": llm_calls}


# ---- Stage 5: forgetting / archival (soft, reversible, bounded) ---------------------------------
_FORGET_MATCH = (
    "MATCH (m:Memory) WHERE (m.archived IS NULL OR m.archived = false) "
    "AND m.company <> $global_tier AND coalesce(m.kind, 'episodic') = 'episodic' "
    "AND coalesce(m.source, 'agent') <> 'user' "
    "AND coalesce(m.importance, 0.0) < $imp_max "
    "AND coalesce(m.decay_score, 1.0) < $decay_max "
    "AND coalesce(m.access_count, 0) = 0 "
    "AND (timestamp() - coalesce(m.ts, timestamp())) > $min_age_ms "
)


def _forget_params() -> dict[str, Any]:
    return dict(
        global_tier=memory.GLOBAL,
        imp_max=_FORGET_IMPORTANCE_MAX,
        decay_max=_FORGET_DECAY_MAX,
        min_age_ms=int(_FORGET_MIN_AGE_DAYS * _MS_PER_DAY),
        max_per_run=_FORGET_MAX_PER_RUN,
    )


async def _stage_forget() -> dict[str, Any]:
    """Soft-archive the lowest-value memories: low importance + decayed + never accessed + past a
    grace period — never the global tier, reflections, or user-sourced facts; bounded per run;
    reversible (clear ``archived``). Disabled by default: reports ``would_archive`` so the operator
    can inspect the candidate set before flipping ``DREAM_ENABLE_FORGET=1``."""
    async with memory._get_driver().session() as session:
        if not _ENABLE_FORGET:
            res = await session.run(_FORGET_MATCH + "RETURN count(m) AS n", **_forget_params())
            rec = await res.single()
            return {"archived": 0, "would_archive": rec["n"] if rec else 0, "skipped": "disabled"}
        res = await session.run(
            _FORGET_MATCH
            + "WITH m ORDER BY coalesce(m.decay_score, 1.0) ASC LIMIT $max_per_run "
            "SET m.archived = true, m.archived_reason = 'decayed' RETURN count(m) AS n",
            **_forget_params(),
        )
        rec = await res.single()
    return {"archived": rec["n"] if rec else 0}


# ---- Stage 6: offline associative (PPR-style) enrichment ----------------------------------------
async def _stage_ppr() -> dict[str, Any]:
    """Precompute, per memory, its top associative neighbors via multi-hop (1-2) weighted SIMILAR_TO
    proximity (HippoRAG's associative-recall spirit, done offline so recall stays a cheap property
    read of ``related_keys``). Opt-in (DREAM_PPR_ENABLED); needs the ``make memory-graph`` edges."""
    if not _PPR_ENABLED:
        return {"enriched": 0, "skipped": "disabled"}
    async with memory._get_driver().session() as session:
        # CALL (m) { ... } = variable-scope subquery (Neo4j 5.23+); the old `CALL { WITH m ... }` form
        # is deprecated on 5.26.
        res = await session.run(
            "MATCH (m:Memory) WHERE (m.archived IS NULL OR m.archived = false) "
            "CALL (m) { "
            "  MATCH (m)-[rels:SIMILAR_TO*1..2]-(n:Memory) "
            "  WHERE n.company = m.company AND n <> m AND (n.archived IS NULL OR n.archived = false) "
            "  WITH n, sum(reduce(s = 1.0, rel IN rels | s * rel.score)) AS prox "
            "  RETURN n.scope_key AS sk ORDER BY prox DESC LIMIT $k "
            "} "
            "WITH m, collect(sk) AS related SET m.related_keys = related "
            "RETURN count(m) AS enriched",
            k=_PPR_LIMIT,
        )
        rec = await res.single()
    return {"enriched": rec["enriched"] if rec else 0}


# ---- orchestrator -------------------------------------------------------------------------------
async def _safe(coro: Any) -> dict[str, Any]:
    """Run a stage; capture (don't propagate) its exception so one bad stage never aborts the pass."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 — stage isolation is the whole point
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dream(backfill_only: bool = False) -> dict[str, Any]:
    """Run the full consolidation pass. Returns per-stage stats (counts + any ``skipped``/``error``).
    Each stage runs in its own session and is isolated via ``_safe``."""
    try:
        await memory._ensure_indexes()
    except Exception as exc:
        return {"success": False, "error": f"neo4j unavailable: {exc}"}

    stats: dict[str, Any] = {"success": True}
    stats["backfill"] = await _safe(_backfill_embeddings())
    if backfill_only:
        return stats
    stats["decay"] = await _safe(_stage_decay())
    stats["dedup"] = await _safe(_stage_dedup())
    stats["reflect"] = await _safe(_stage_reflect())
    stats["contradict"] = await _safe(_stage_contradict())
    stats["forget"] = await _safe(_stage_forget())
    stats["ppr"] = await _safe(_stage_ppr())
    return stats


if __name__ == "__main__":
    _backfill_only = "--backfill-only" in sys.argv
    print(json.dumps(asyncio.run(dream(backfill_only=_backfill_only)), indent=2))
