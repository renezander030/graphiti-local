"""Read-only reports of entities, and of facts, that repeat once casing and punctuation go.

Graphiti resolves duplicates by embedding similarity, so ``ACME Corp`` and ``Acme
corp.`` can end up as two entities when their vectors fall under the threshold. Nothing
here merges anything: the report shows a human what split, and a correction goes through
the proposal queue like any other fact.
"""

from __future__ import annotations

import re
from typing import Any

_NOISE = re.compile(r"[^0-9a-z]+")


def normalized_name(name: str) -> str:
    return " ".join(_NOISE.sub(" ", (name or "").casefold()).split())


def _member(node: Any) -> dict[str, Any]:
    return {
        "uuid": node.uuid,
        "name": node.name,
        "group_id": node.group_id,
        "summary": (getattr(node, "summary", "") or "")[:120],
    }


def cluster(nodes: list[Any]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Any]] = {}
    for node in nodes:
        key = normalized_name(node.name)
        if key:
            buckets.setdefault(key, []).append(node)
    clusters = [
        {"name": key, "count": len(members), "members": [_member(node) for node in members]}
        for key, members in buckets.items()
        if len(members) > 1
    ]
    clusters.sort(key=lambda item: (-item["count"], item["name"]))
    return clusters


async def find_duplicates(
    graph: Any, settings: Any, groups: list[str], *, limit: int = 50
) -> dict[str, Any]:
    from graphiti_core.nodes import EntityNode

    from kg_mcp.queries import collect

    nodes = await collect(graph, settings, groups, EntityNode.get_by_group_ids)
    clusters = cluster(nodes)
    return {
        "groups": groups,
        "entities": len(nodes),
        "clusters": clusters[:limit],
        "duplicates": sum(item["count"] for item in clusters),
    }


def cluster_facts(edges: list[Any]) -> list[dict[str, Any]]:
    """Current facts that say the same thing about the same two entities."""
    from kg_mcp.retrieval import current_edge

    buckets: dict[tuple[str, str, str], list[Any]] = {}
    for edge in edges:
        key = normalized_name(edge.fact)
        if key and current_edge(edge):
            buckets.setdefault((edge.source_node_uuid, edge.target_node_uuid, key), []).append(edge)
    clusters = [
        {
            "fact": members[0].fact,
            "count": len(members),
            "members": [
                {
                    "uuid": edge.uuid,
                    "group_id": edge.group_id,
                    "created_at": edge.created_at.isoformat() if edge.created_at else None,
                }
                for edge in sorted(members, key=lambda edge: str(edge.created_at))
            ],
        }
        for members in buckets.values()
        if len(members) > 1
    ]
    clusters.sort(key=lambda item: (-item["count"], item["fact"]))
    return clusters


async def find_fact_duplicates(
    graph: Any, settings: Any, groups: list[str], *, limit: int = 50
) -> dict[str, Any]:
    """Repeated copies of one fact; ``kg retire`` takes the extra copies' uuids."""
    from graphiti_core.edges import EntityEdge

    from kg_mcp.queries import collect

    edges = await collect(graph, settings, groups, EntityEdge.get_by_group_ids)
    clusters = cluster_facts(edges)
    return {
        "groups": groups,
        "facts": len(edges),
        "clusters": clusters[:limit],
        "duplicates": sum(item["count"] for item in clusters),
    }
