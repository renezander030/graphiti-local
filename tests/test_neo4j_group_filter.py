"""Neo4j fulltext searches over several groups match every group, not only the last.

graphiti-core builds ``group_id:"a" OR group_id:"b" AND (q)``; Lucene binds AND
tighter, so that reads ``a OR (b AND q)`` and filters down to group ``b`` alone.
"""

import asyncio

import pytest

from kg_mcp.config import Settings
from kg_mcp.runtime import build_graphiti


def _sent_queries(monkeypatch, groups, search):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from graphiti_core.search import search_utils
    from graphiti_core.search.search_filters import SearchFilters

    settings = Settings.model_validate(
        {"graph": {"groups": ["work", "home"]}, "database": {"provider": "neo4j"}}
    )
    graph = build_graphiti(settings, read_only=True)
    sent = []

    async def execute_query(cypher, **kwargs):
        sent.append(kwargs["query"])
        return [], None, None

    graph.driver.client.execute_query = execute_query

    async def run():
        try:
            await getattr(search_utils, search)(graph.driver, "alice", SearchFilters(), groups)
        finally:
            await graph.driver.client.close()

    asyncio.run(run())
    return sent


@pytest.mark.parametrize(
    "search", ["edge_fulltext_search", "node_fulltext_search", "episode_fulltext_search"]
)
def test_several_groups_are_one_parenthesized_alternative(monkeypatch, search):
    assert _sent_queries(monkeypatch, ["work", "home"], search) == [
        '(group_id:"work" OR group_id:"home") AND (alice)'
    ]


def test_one_group_is_sent_unchanged(monkeypatch):
    assert _sent_queries(monkeypatch, ["work"], "edge_fulltext_search") == [
        'group_id:"work" AND (alice)'
    ]
