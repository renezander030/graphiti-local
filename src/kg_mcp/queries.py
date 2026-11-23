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


def rank_merged(results: Any, limit: int) -> Any:
    """Search results ranked best first across groups and cut to ``limit``.

    graphiti-core searches each FalkorDB group on its own and concatenates the
    per-group results, so the first group's weak matches come before the next group's
    strong ones and the list grows to groups x limit (getzep/graphiti#2014). Each kind
    is reordered by its reranker score, one copy per uuid at its best score.
    """
    from graphiti_core.search.search_config import SearchResults

    ranked: dict[str, Any] = {}
    for kind, scored in (
        ("edges", "edge_reranker_scores"),
        ("nodes", "node_reranker_scores"),
        ("episodes", "episode_reranker_scores"),
        ("communities", "community_reranker_scores"),
    ):
        items, scores = getattr(results, kind), getattr(results, scored)
        if len(scores) != len(items):  # no comparable scores: keep the order, cut only
            scores = [0.0] * len(items)
        best: dict[str, tuple[Any, float]] = {}
        for item, score in zip(items, scores, strict=True):
            if item.uuid not in best or score > best[item.uuid][1]:
                best[item.uuid] = (item, score)
        top = sorted(best.values(), key=lambda pair: pair[1], reverse=True)[:limit]
        ranked[kind] = [item for item, _ in top]
        ranked[scored] = [score for _, score in top]
    return SearchResults(**ranked)


async def ranked_search(
    graph: Any, settings: Any, query: str, config: Any, groups: list[str] | None, **kwargs: Any
) -> Any:
    """``graph.search_`` whose results over several FalkorDB groups are ranked as one list."""
    result = await graph.search_(query, config=config, group_ids=groups, **kwargs)
    if settings.database.provider == "falkordb" and groups and len(groups) > 1:
        return rank_merged(result, config.limit)
    return result


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
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_NODE_DISTANCE,
        EDGE_HYBRID_SEARCH_RRF,
    )

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
        config = keyword_edge_search_config(candidates)
    else:
        # The recipe ``graph.search`` uses, through ``search_`` so the scores come back.
        recipe = EDGE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else EDGE_HYBRID_SEARCH_RRF
        config = recipe.model_copy(update={"limit": candidates})
    result = await ranked_search(
        graph,
        settings,
        safe_query,
        config,
        groups,
        center_node_uuid=center_node_uuid,
        search_filter=search_filter,
    )
    edges = result.edges[:candidates]
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
        result = await ranked_search(
            graph,
            settings,
            safe_query,
            config,
            groups,
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
    """The edge with this UUID and the driver it was found on, from the first that has it.

    ``(None, None)`` means no driver has it. A driver that failed to answer is not an
    absence: when no driver has the edge and one failed, the failure is raised.
    """
    from graphiti_core.edges import EntityEdge
    from graphiti_core.errors import EdgeNotFoundError

    failure: Exception | None = None
    for driver in drivers:
        try:
            return await bounded(EntityEdge.get_by_uuid(driver, uuid), timeout, label), driver
        except TimeoutError:
            raise
        except EdgeNotFoundError:
            continue
        except Exception as exc:
            failure = exc
    if failure is not None:
        raise RuntimeError(f"edge lookup failed: {failure}") from failure
    return None, None


async def _episode_entity_edges(
    graph: Any, settings: Any, groups: list[str], episodes: list[str]
) -> list[Any]:
    """Every fact edge these episodes name: those they gave and those they ended."""
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EpisodicNode

    async def fetch(driver: Any, _ids: list[str]) -> list[Any]:
        found = await EpisodicNode.get_by_uuids(driver, episodes)
        uuids = list(dict.fromkeys(uuid for item in found for uuid in item.entity_edges or []))
        return await EntityEdge.get_by_uuids(driver, uuids) if uuids else []

    edges = await collect(graph, settings, groups, fetch)
    if settings.database.provider != "ladybug":
        edges = [edge for edge in edges if edge.group_id in groups]
    return edges


async def retraction_edges(
    graph: Any, settings: Any, groups: list[str], episodes: list[str]
) -> tuple[list[str], list[str], list[Any]]:
    """What retracting these episodes touches, from one read of the facts they name.

    The current facts they produced split into those no other episode supports and
    those another episode restates; retracting retires only the first, because a fact
    another source also gave stays true on that source's word. The third list is the
    facts they ended: edges they name but do not support, no longer current.
    graphiti-core keeps no record of what ended a fact, and can drop or overwrite the
    evidence for its true end, so retracting cannot restore these safely; they stay
    ended and need their own review (getzep/graphiti#2165).
    """
    from kg_mcp.retrieval import current_edge

    mine = set(episodes)
    only: list[str] = []
    shared: list[str] = []
    ended: list[Any] = []
    for edge in await _episode_entity_edges(graph, settings, groups, episodes):
        supporters = set(edge.episodes or [])
        if current_edge(edge):
            (only if supporters <= mine else shared).append(edge.uuid)
        elif not mine & supporters:
            ended.append(edge)
    return only, shared, ended


async def count_nodes(driver: Any, timeout: float, label: str) -> int:
    rows, _, _ = await bounded(
        driver.execute_query("MATCH (n) RETURN count(n) AS count"), timeout, label
    )
    return rows[0]["count"]
