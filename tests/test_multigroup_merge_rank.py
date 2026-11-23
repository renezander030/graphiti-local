"""A search over several FalkorDB groups ranks the merged results by score.

graphiti-core runs a FalkorDB search once per group and concatenates the per-group
results (https://github.com/getzep/graphiti/issues/2014): the first group's weak
matches come before the second group's strong ones, and the candidate list grows to
groups x limit. These tests drive the real graphiti-core decorator with a fake driver.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchResults

from kg_mcp import queries
from kg_mcp.config import Settings
from kg_mcp.reranker import PassthroughReranker

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Per group, best first, as graphiti-core returns one partition: (name, score).
GROUPS = {
    "g1": [("weak one", 0.40), ("weak two", 0.30), ("weak three", 0.20)],
    "g2": [("strong", 0.98)],
}


def _edge(name: str, group: str) -> EntityEdge:
    return EntityEdge(
        uuid=f"edge-{name}",
        group_id=group,
        source_node_uuid="a",
        target_node_uuid="b",
        name="RELATES_TO",
        fact=name,
        created_at=NOW,
    )


def _node(name: str, group: str) -> EntityNode:
    return EntityNode(uuid=f"node-{name}", name=name, group_id=group, created_at=NOW)


class FakeDriver:
    provider = GraphProvider.FALKORDB

    def __init__(self, database: str = "default_db"):
        self._database = database

    def clone(self, database: str) -> FakeDriver:
        return FakeDriver(database)


class FakeGraph:
    def __init__(self):
        self.clients = SimpleNamespace(driver=FakeDriver())
        self.cross_encoder = PassthroughReranker()

    @handle_multiple_group_ids
    async def search(
        self,
        query,
        center_node_uuid=None,
        group_ids=None,
        num_results=10,
        search_filter=None,
        driver=None,
    ):
        group = driver._database
        return [_edge(name, group) for name, _ in GROUPS[group]][:num_results]

    @handle_multiple_group_ids
    async def search_(
        self,
        query,
        config=None,
        group_ids=None,
        center_node_uuid=None,
        bfs_origin_node_uuids=None,
        search_filter=None,
        driver=None,
    ):
        group = driver._database
        hits = GROUPS[group][: config.limit]
        return SearchResults(
            edges=[_edge(name, group) for name, _ in hits],
            edge_reranker_scores=[score for _, score in hits],
            nodes=[_node(name, group) for name, _ in hits],
            node_reranker_scores=[score for _, score in hits],
        )


def _settings() -> Settings:
    settings = Settings()
    assert settings.database.provider == "falkordb"
    return settings


def test_fact_search_over_groups_ranks_by_score_and_keeps_the_limit():
    for keyword in (False, True):
        ranked, retrieval = asyncio.run(
            queries.search_facts(FakeGraph(), _settings(), "q", ["g1", "g2"], 1, keyword=keyword)
        )
        assert [edge.fact for edge, _ in ranked] == ["strong"], keyword
        assert retrieval["candidates_seen"] <= retrieval["candidate_limit"], keyword


def test_node_search_over_groups_ranks_by_score_and_keeps_the_limit():
    for keyword in (False, True):
        nodes, retrieval = asyncio.run(
            queries.search_nodes(FakeGraph(), _settings(), "q", ["g1", "g2"], 1, keyword=keyword)
        )
        assert [node.name for node in nodes] == ["strong"], keyword
        assert retrieval["candidates_seen"] <= retrieval["candidate_limit"], keyword


def test_merge_keeps_one_copy_of_a_record_at_its_best_score():
    edge = _edge("same", "g1")
    merged = SearchResults.merge(
        [
            SearchResults(edges=[edge], edge_reranker_scores=[0.2]),
            SearchResults(edges=[_edge("other", "g2"), edge], edge_reranker_scores=[0.5, 0.9]),
        ]
    )
    ranked = queries.rank_merged(merged, 5)
    assert [e.fact for e in ranked.edges] == ["same", "other"]
    assert ranked.edge_reranker_scores == [0.9, 0.5]
