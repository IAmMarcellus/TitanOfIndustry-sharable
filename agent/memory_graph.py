"""Batch graph-enrichment for TitanOfIndustry memory (free GDS on Neo4j Community).

Builds a relationship layer over the flat ``:Memory`` store so ``recall`` can use topic clusters +
importance:

    tags -> (:Tag) + (:Memory)-[:HAS_TAG]->(:Tag)
         -> GDS nodeSimilarity (shared-tag Jaccard) -> (:Memory)-[:SIMILAR_TO]->(:Memory) {score}
         -> GDS Louvain   -> m.topic        (community / topic cluster)
         -> GDS PageRank  -> m.importance    (centrality within the similarity graph)

This is an **on-demand / batch** job — run via ``make memory-graph`` (or ``python -m
agent.memory_graph``), NOT per write. ``remember``/``recall`` stay fast and work whether or not this
has run. All GDS calls here are free on Neo4j Community (capped at 4 cores / 3 in-memory models / no
cross-restart model persistence — all irrelevant at this scale). Phase B2 (embedding-similarity via
``gds.knn`` + a persistable FastRP/GraphSAGE node-embedding model) is deferred until the embed server
has VRAM; true model persistence would require Neo4j Enterprise.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from . import memory  # call memory._get_driver() at runtime (monkeypatch-friendly)

_TAG_GRAPH = "mem_tag"
_SIM_GRAPH = "mem_sim"


async def _drop(session: Any, name: str) -> None:
    """Drop a GDS in-memory projection if present (failIfMissing=false)."""
    # explicit YIELD avoids the deprecation notice for the proc's deprecated 'schema' field
    await session.run("CALL gds.graph.drop($name, false) YIELD graphName", name=name)


async def rebuild_graph() -> dict[str, Any]:
    """(Re)build the memory graph layer idempotently. Returns node/edge counts."""
    async with memory._get_driver().session() as session:
        # 0. clear the derived layer so each rebuild is fully idempotent (re-derived below)
        await session.run("MATCH (:Memory)-[r:HAS_TAG]->() DELETE r")
        await session.run("MATCH (:Memory)-[r:SIMILAR_TO]->() DELETE r")

        # 1. tags (comma-separated string) -> :Tag nodes + HAS_TAG edges
        await session.run(
            "MATCH (m:Memory) WHERE m.tags IS NOT NULL AND m.tags <> '' "
            "UNWIND [t IN split(m.tags, ',') WHERE trim(t) <> ''] AS raw "
            "WITH m, trim(raw) AS tag "
            "MERGE (g:Tag {name: tag}) "
            "MERGE (m)-[:HAS_TAG]->(g)"
        )

        # 2. shared-tag similarity -> fresh SIMILAR_TO {score} (GDS nodeSimilarity on the bipartite)
        await _drop(session, _TAG_GRAPH)
        await session.run(
            "CALL gds.graph.project($g, ['Memory', 'Tag'], {HAS_TAG: {orientation: 'NATURAL'}})",
            g=_TAG_GRAPH,
        )
        await session.run(
            "CALL gds.nodeSimilarity.write($g, {writeRelationshipType: 'SIMILAR_TO', "
            "writeProperty: 'score', similarityCutoff: 0.1}) YIELD relationshipsWritten",
            g=_TAG_GRAPH,
        )
        await _drop(session, _TAG_GRAPH)

        # Tenant scoping: drop SIMILAR_TO edges that cross companies so the similarity graph is
        # partitioned by company. Louvain/PageRank below then run globally but can't form communities
        # that span companies (no cross-company edges) — and Louvain's ids stay globally unique, so a
        # company's `topic` never collides with another company's or `global`'s. (Null-company nodes,
        # i.e. pre-migration, compare as null and are left intact — run `make memory-migrate` first.)
        await session.run(
            "MATCH (a:Memory)-[r:SIMILAR_TO]->(b:Memory) WHERE a.company <> b.company DELETE r"
        )

        # 3. Louvain (topic) + PageRank (importance) over the SIMILAR_TO graph
        await _drop(session, _SIM_GRAPH)
        await session.run(
            "CALL gds.graph.project($g, 'Memory', "
            "{SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})",
            g=_SIM_GRAPH,
        )
        await session.run(
            "CALL gds.louvain.write($g, {writeProperty: 'topic', "
            "relationshipWeightProperty: 'score'}) YIELD communityCount",
            g=_SIM_GRAPH,
        )
        await session.run(
            "CALL gds.pageRank.write($g, {writeProperty: 'importance', "
            "relationshipWeightProperty: 'score'}) YIELD ranIterations",
            g=_SIM_GRAPH,
        )
        await _drop(session, _SIM_GRAPH)

        # drop tags no longer referenced by any memory (keeps the :Tag set clean across rebuilds)
        await session.run("MATCH (t:Tag) WHERE NOT ()-[:HAS_TAG]->(t) DELETE t")

        # 4. counts
        rec = await (await session.run(
            "MATCH (m:Memory) RETURN count(m) AS memories, count(m.topic) AS with_topic"
        )).single()
        edges = await (await session.run(
            "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS sim_edges"
        )).single()

    return {
        "success": True,
        "memories": rec["memories"] if rec else 0,
        "with_topic": rec["with_topic"] if rec else 0,
        "sim_edges": edges["sim_edges"] if edges else 0,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(rebuild_graph()), indent=2))
