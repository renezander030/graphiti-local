"""Graph reads shared by the ``kg`` CLI and the MCP server.

Each function takes an open graph and returns graphiti objects with their retrieval
stats. The CLI and the server only decide the groups a call may read and shape the
output, so a search fix lands once for both.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kg_mcp.runtime import bounded


def group_scopes(graph: Any, settings: Any, groups: list[str]) -> list[tuple[Any, list[str]]]:
    """Where these groups' records live, as the (driver, group ids) pairs a read visits.

    FalkorDB keeps each group in its own database. A Ladybug file is single-graph and
    ingestion writes ``group_id=None``, so its empty group is swept too.
    """
    if settings.database.provider == "falkordb":
        return [(graph.driver.clone(database=group), [group]) for group in groups]
    if settings.database.provider == "ladybug":
        return [(graph.driver, groups), (graph.driver, [""])]
    return [(graph.driver, groups)]


def graph_drivers(graph: Any, settings: Any, groups: list[str] | None = None) -> list[Any]:
    """Each distinct driver the groups' records live behind, for lookups by uuid."""
    scopes = group_scopes(graph, settings, groups or settings.graph.groups)
    return list({id(driver): driver for driver, _ in scopes}.values())


async def collect(graph: Any, settings: Any, groups: list[str], fetch: Any) -> list[Any]:
    """What ``fetch(driver, group_ids)`` returns across the groups' scopes, once per uuid."""
    from graphiti_core.errors import GroupsEdgesNotFoundError

    found: dict[str, Any] = {}
    for driver, ids in group_scopes(graph, settings, groups):
        try:
            records = await fetch(driver, ids) or []
        except GroupsEdgesNotFoundError:  # graphiti's way of saying "no edges here"
            records = []
        for record in records:
            found.setdefault(record["uuid"] if isinstance(record, dict) else record.uuid, record)
    return list(found.values())


async def search_facts(
    graph: Any,
    settings: Any,
    query: str,
    groups: list[str] | None,
    limit: int,
    *,
    keyword: bool = False,
    include_invalidated: bool = False,
    center_node_uuid: str | None = None,
    search_filter: Any = None,
) -> tuple[list[tuple[Any, float]], dict[str, Any]]:
    """Current, identity-safe reranked edges for a query."""
    from kg_mcp.retrieval import (
        candidate_limit,
        compatible_query,
        current_edges,
        keyword_edge_search_config,
        rerank_edges,
        retrieval_stats,
    )

    safe_query = compatible_query(query, settings)
    candidates = candidate_limit(limit, settings)
    if keyword:
        result = await graph.search_(
            safe_query,
            config=keyword_edge_search_config(candidates),
            group_ids=groups,
            center_node_uuid=center_node_uuid,
            search_filter=search_filter,
        )
        edges = result.edges[:candidates]
    else:
        edges = await graph.search(
            safe_query,
            group_ids=groups,
            num_results=candidates,
            center_node_uuid=center_node_uuid,
            search_filter=search_filter,
        )
    current, suppressed = current_edges(list(edges), include_invalidated=include_invalidated)
    ranked = await rerank_edges(graph, safe_query, current, settings, limit)
    return ranked, retrieval_stats(
        requested=limit,
        candidate_ceiling=candidates,
        candidates_seen=len(edges),
        eligible=len(current),
        returned=len(ranked),
        suppressed=suppressed,
    )


async def search_nodes(
    graph: Any,
    settings: Any,
    query: str,
    groups: list[str] | None,
    limit: int,
    *,
    keyword: bool = False,
    labels: list[str] | None = None,
    center_node_uuid: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Entities matching a query, optionally narrowed to labels or ranked around a node."""
    from graphiti_core.search.search_config_recipes import (
        NODE_HYBRID_SEARCH_NODE_DISTANCE,
        NODE_HYBRID_SEARCH_RRF,
    )
    from graphiti_core.search.search_filters import SearchFilters

    from kg_mcp.retrieval import (
        candidate_limit,
        compatible_query,
        keyword_node_search_config,
        retrieval_stats,
    )

    safe_query = compatible_query(query, settings)
    candidates = candidate_limit(limit, settings)
    if keyword:
        config = keyword_node_search_config(candidates)
    else:
        recipe = NODE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else NODE_HYBRID_SEARCH_RRF
        config = recipe.model_copy(update={"limit": candidates})

    async def search(node_labels: list[str] | None) -> list[Any]:
        result = await graph.search_(
            safe_query,
            config=config,
            group_ids=groups,
            center_node_uuid=center_node_uuid,
            search_filter=SearchFilters(node_labels=node_labels),
        )
        return list(result.nodes or [])

    labels = labels or []
    if settings.database.provider != "falkordb" or len(labels) < 2:
        found = await search(labels or None)
    else:
        # RediSearch cannot parse the combined ``n:A|B`` label expression that
        # graphiti-core emits. Query labels independently, then merge by UUID.
        merged: dict[str, Any] = {}
        for nodes in await asyncio.gather(*(search([label]) for label in labels)):
            for node in nodes:
                merged.setdefault(node.uuid, node)
        found = list(merged.values())
    returned = found[:limit]
    return returned, retrieval_stats(
        requested=limit,
        candidate_ceiling=candidates,
        candidates_seen=len(found),
        eligible=len(found),
        returned=len(returned),
    )


async def find_edge(
    drivers: list[Any], uuid: str, timeout: float, label: str
) -> tuple[Any, Any] | tuple[None, None]:
    """The edge with this UUID and the driver it was found on, from the first that has it."""
    from graphiti_core.edges import EntityEdge

    for driver in drivers:
        try:
            return await bounded(EntityEdge.get_by_uuid(driver, uuid), timeout, label), driver
        except TimeoutError:
            raise
        except Exception:
            continue
    return None, None


async def count_nodes(driver: Any, timeout: float, label: str) -> int:
    rows, _, _ = await bounded(
        driver.execute_query("MATCH (n) RETURN count(n) AS count"), timeout, label
    )
    return rows[0]["count"]
