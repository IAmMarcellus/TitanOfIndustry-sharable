"""One-off migration: assign company/agent/scope_key to pre-scoping :Memory nodes.

Pre-scoping memory was a single global brain (no tenant property). This backfills the scope:
  - nodes tagged with a ``BET-*`` / ``TOI-*`` task id  -> the **bet-arb company** (which owns those
    Paperclip projects; override with ``--company``).
  - everything else (stack / infra facts)             -> the shared ``global`` tier.
Then it swaps the dedup constraint from ``text_hash`` (global) to ``scope_key`` (= ``company:text_hash``).

Idempotent: only nodes with **no** ``company`` are touched, so re-running is a no-op. Run via
``make memory-migrate`` after ``make neo4j`` (``make memory-migrate ARGS='--dry-run'`` to preview).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from . import memory

# The company that owns the bet-arb (BET) + TitanOfIndustry (TOI) Paperclip projects.
_DEFAULT_COMPANY = "<REDACTED_UUID>"

# Categorize an unscoped node: BET-*/TOI-* task memories -> the company; stack facts -> the global tier
# ($global_tier = memory.GLOBAL, passed as a param so the tier name has one source of truth).
_CATEGORIZE = (
    "(CASE WHEN m.tags CONTAINS 'BET-' OR m.tags CONTAINS 'TOI-' THEN $company ELSE $global_tier END)"
)


async def migrate(company: str, dry_run: bool) -> dict[str, Any]:
    async with memory._get_driver().session() as session:
        preview = await (
            await session.run(
                "MATCH (m:Memory) WHERE m.company IS NULL AND m.text_hash IS NOT NULL "
                f"WITH {_CATEGORIZE} AS company RETURN company, count(*) AS n ORDER BY company",
                company=company, global_tier=memory.GLOBAL,
            )
        ).data()
        plan = {r["company"]: r["n"] for r in preview}
        orphans = (
            await (
                await session.run(
                    "MATCH (m:Memory) WHERE m.company IS NULL AND m.text_hash IS NULL RETURN count(*) AS n"
                )
            ).single()
        )["n"]
        if dry_run:
            return {"dry_run": True, "would_assign": plan, "skipped_no_text_hash": orphans}

        # 1. drop the global text_hash uniqueness (scoped duplicates of the same text become allowed)
        await session.run("DROP CONSTRAINT memory_text_hash_unique IF EXISTS")
        # 2. backfill company / agent / scope_key for unscoped nodes (company computed once in WITH)
        await session.run(
            "MATCH (m:Memory) WHERE m.company IS NULL AND m.text_hash IS NOT NULL "
            f"WITH m, {_CATEGORIZE} AS comp "
            "SET m.company = comp, m.agent = coalesce(m.agent, ''), "
            "m.scope_key = comp + ':' + m.text_hash",
            company=company, global_tier=memory.GLOBAL,
        )
        # 3. add the per-scope uniqueness constraint
        await session.run(
            "CREATE CONSTRAINT memory_scope_key_unique IF NOT EXISTS "
            "FOR (m:Memory) REQUIRE m.scope_key IS UNIQUE"
        )
        by_company = await (
            await session.run(
                "MATCH (m:Memory) RETURN m.company AS company, count(*) AS n ORDER BY company"
            )
        ).data()
    return {
        "dry_run": False,
        "assigned": plan,
        "skipped_no_text_hash": orphans,
        "by_company": {r["company"]: r["n"] for r in by_company},
    }


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Backfill tenant scope on :Memory nodes.")
    parser.add_argument("--company", default=_DEFAULT_COMPANY, help="company id for BET-*/TOI-* memories")
    parser.add_argument("--dry-run", action="store_true", help="report categorization without writing")
    args = parser.parse_args()
    try:
        print(json.dumps(await migrate(args.company, args.dry_run), indent=2))
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(_main())
