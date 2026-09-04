"""Read-only report of entities that share a name once casing and punctuation are ignored.

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


async def _entities(graph: Any, settings: Any, groups: list[str]) -> list[Any]:
    from graphiti_core.nodes import EntityNode

    provider = settings.database.provider
    if provider == "falkordb":
        nodes: list[Any] = []
        for group in groups:
            driver = graph.driver.clone(database=group)
            nodes.extend(await EntityNode.get_by_group_ids(driver, [group]) or [])
        return nodes
    if provider == "ladybug":
        seen: set[str] = set()
        nodes = []
        for candidate in (groups, [""]):
            for node in await EntityNode.get_by_group_ids(graph.driver, candidate) or []:
                if node.uuid not in seen:
                    seen.add(node.uuid)
                    nodes.append(node)
        return nodes
    return list(await EntityNode.get_by_group_ids(graph.driver, groups) or [])


async def find_duplicates(
    graph: Any, settings: Any, groups: list[str], *, limit: int = 50
) -> dict[str, Any]:
    nodes = await _entities(graph, settings, groups)
    clusters = cluster(nodes)
    return {
        "groups": groups,
        "entities": len(nodes),
        "clusters": clusters[:limit],
        "duplicates": sum(item["count"] for item in clusters),
    }
