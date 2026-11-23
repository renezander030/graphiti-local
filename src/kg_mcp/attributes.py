"""Nested attribute values stored as JSON text on Neo4j and FalkorDB.

graphiti-core spreads entity and edge attributes into graph properties on Neo4j and
FalkorDB. Both servers accept only primitive values and arrays of them, so one
extracted attribute holding an object or a list of objects aborts the whole episode.
The Kuzu (Ladybug) path already stores the attribute map as one JSON string and is
left as it is. Here every nested value is written as its JSON text instead; flat
values keep their native type. Reads return that text as a string.
"""

from __future__ import annotations

import json
from typing import Any

_INSTALLED = False


def _servers() -> set[Any]:
    from graphiti_core.driver.driver import GraphProvider

    return {GraphProvider.NEO4J, GraphProvider.FALKORDB}


def is_nested(value: Any) -> bool:
    """True for a value a property cannot hold: a map, or a list holding maps or lists."""
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, (dict, list, tuple)) for item in value)
    return False


def flat_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """The attribute map with each nested value replaced by its JSON text."""
    from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

    return {
        key: json.dumps(convert_datetimes_to_strings(value), ensure_ascii=False)
        if is_nested(value)
        else value
        for key, value in (attributes or {}).items()
    }


def _flat(item: Any) -> Any:
    """``item`` itself when nothing is nested, else a copy with flat attributes."""
    if not any(is_nested(value) for value in (item.attributes or {}).values()):
        return item
    return item.model_copy(update={"attributes": flat_attributes(item.attributes)})


def install_nested_attribute_serialiser() -> None:
    """Wrap graphiti-core's three entity write paths once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode
    from graphiti_core.utils import bulk_utils

    node_save = EntityNode.save
    edge_save = EntityEdge.save
    bulk_tx = bulk_utils.add_nodes_and_edges_bulk_tx

    async def save_node(self: Any, driver: Any) -> Any:
        target = _flat(self) if driver.provider in _servers() else self
        return await node_save(target, driver)

    async def save_edge(self: Any, driver: Any) -> Any:
        target = _flat(self) if driver.provider in _servers() else self
        return await edge_save(target, driver)

    async def save_bulk(
        tx: Any,
        episodic_nodes: list[Any],
        episodic_edges: list[Any],
        entity_nodes: list[Any],
        entity_edges: list[Any],
        embedder: Any,
        driver: Any,
    ) -> Any:
        if driver.provider not in _servers():
            return await bulk_tx(
                tx, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder, driver
            )
        nodes = [_flat(node) for node in entity_nodes]
        edges = [_flat(edge) for edge in entity_edges]
        try:
            return await bulk_tx(tx, episodic_nodes, episodic_edges, nodes, edges, embedder, driver)
        finally:
            # The bulk write fills missing embeddings on what it was given; keep them.
            for original, copy in zip(entity_nodes, nodes, strict=True):
                if copy is not original:
                    original.name_embedding = copy.name_embedding
            for original, copy in zip(entity_edges, edges, strict=True):
                if copy is not original:
                    original.fact_embedding = copy.fact_embedding

    EntityNode.save = save_node  # type: ignore[method-assign]
    EntityEdge.save = save_edge  # type: ignore[method-assign]
    bulk_utils.add_nodes_and_edges_bulk_tx = save_bulk
    _INSTALLED = True
