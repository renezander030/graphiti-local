"""Nested attribute values reach Neo4j and FalkorDB as JSON text, not as maps.

Neo4j and FalkorDB accept only primitive properties and arrays of them, so a map or a
list of maps among an entity's or edge's attributes aborts the whole episode. These
tests capture what graphiti-core hands the driver after ``build_graphiti`` ran; no
graph server is needed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from kg_mcp.config import Settings

NESTED = {
    "source": {"title": "Spec", "description": "second draft"},
    "authors": [{"name": "Ada"}, {"name": "Grace"}],
    "pages": 3,
    "tags": ["draft", "spec"],
}


def _settings(provider: str) -> Settings:
    return Settings.model_validate(
        {
            "database": {"provider": provider},
            "llm": {"model": "model", "api_key": "unused"},
            "embedder": {"model": "embedder", "api_key": "unused"},
        }
    )


def _primitive(value) -> bool:
    if isinstance(value, (list, tuple)):
        return not any(isinstance(item, (dict, list, tuple)) for item in value)
    return not isinstance(value, dict)


class CapturingDriver:
    graph_operations_interface = None

    def __init__(self, provider):
        self.provider = provider
        self.calls: list[dict] = []

    async def execute_query(self, query, **params):
        self.calls.append(params)
        return [], None, None


class CapturingTx:
    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, query, **params):
        self.calls.append(params)


class Embedder:
    async def create(self, input_data):
        return [0.5, 0.5]


def _node(attributes=NESTED):
    from graphiti_core.nodes import EntityNode

    return EntityNode(name="Spec", group_id="g", labels=["Document"], attributes=dict(attributes))


def _edge():
    from graphiti_core.edges import EntityEdge

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return EntityEdge(
        source_node_uuid="a",
        target_node_uuid="b",
        name="CITES",
        fact="Spec cites Notes",
        group_id="g",
        created_at=now,
        attributes={"citation": {"page": 4, "at": now}, "count": 1},
    )


@pytest.fixture(params=["NEO4J", "FALKORDB"])
def server(request):
    """A built graph (a Neo4j one opens no connection), then the server it targets."""
    from graphiti_core.driver.driver import GraphProvider

    from kg_mcp.runtime import build_graphiti

    graph = build_graphiti(_settings("neo4j"), read_only=False)
    asyncio.run(graph.close())
    return GraphProvider[request.param]


def test_entity_node_save_sends_only_primitive_properties(server):
    driver = CapturingDriver(server)
    node = _node()
    asyncio.run(node.save(driver))

    properties = driver.calls[0]["entity_data"]
    assert all(_primitive(value) for value in properties.values()), properties
    assert json.loads(properties["source"]) == NESTED["source"]
    assert json.loads(properties["authors"]) == NESTED["authors"]
    assert properties["pages"] == 3 and properties["tags"] == ["draft", "spec"]
    assert node.attributes == NESTED, "the node in memory keeps its structured value"


def test_entity_edge_save_sends_only_primitive_properties(server):
    driver = CapturingDriver(server)
    asyncio.run(_edge().save(driver))

    properties = driver.calls[0]["edge_data"]
    assert all(_primitive(value) for value in properties.values()), properties
    assert json.loads(properties["citation"]) == {"page": 4, "at": "2026-01-01T00:00:00+00:00"}
    assert properties["count"] == 1


def test_bulk_write_sends_only_primitive_properties(server):
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

    driver = CapturingDriver(server)
    tx = CapturingTx()
    node = _node()
    asyncio.run(add_nodes_and_edges_bulk_tx(tx, [], [], [node], [_edge()], Embedder(), driver))

    nodes = next(call["nodes"] for call in tx.calls if "nodes" in call)
    edges = next(call["entity_edges"] for call in tx.calls if "entity_edges" in call)
    for row in nodes + edges:
        assert all(_primitive(value) for value in row.values()), row
    assert json.loads(nodes[0]["source"]) == NESTED["source"]
    assert node.name_embedding == [0.5, 0.5], "an embedding filled during the write is kept"


def test_kuzu_keeps_its_own_json_attribute_column(server):
    del server
    from graphiti_core.driver.driver import GraphProvider

    driver = CapturingDriver(GraphProvider.KUZU)
    asyncio.run(_node().save(driver))

    assert json.loads(driver.calls[0]["attributes"]) == NESTED
