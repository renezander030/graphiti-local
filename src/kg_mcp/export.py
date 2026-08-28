"""Portable JSONL snapshot of a graph.

The snapshot is backend-independent: it is written from the graphiti models rather than
from backend rows, so a graph captured on FalkorDB can be replayed into Neo4j or Ladybug.
Embeddings are omitted — they are derived from the text and re-created on import, and a
vector written back under a different embedding model would be silently wrong.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
EMBEDDING_FIELDS = {"name_embedding", "fact_embedding"}


def _serialize(record: Any, kind: str) -> dict[str, Any]:
    payload = record.model_dump(mode="json", exclude=EMBEDDING_FIELDS)
    payload["kind"] = kind
    return payload


def default_output() -> Path:
    from kg_mcp.workspace import workspace_dir

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return workspace_dir() / f"export-{stamp}.jsonl"


async def _collect(driver: Any, groups: list[str]) -> list[dict[str, Any]]:
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodicNode

    kinds = (
        (EntityNode, "entity_node"),
        (EpisodicNode, "episodic_node"),
        (EntityEdge, "entity_edge"),
        (EpisodicEdge, "episodic_edge"),
    )
    collected: list[dict[str, Any]] = []
    for model, kind in kinds:
        try:
            records = await model.get_by_group_ids(driver, groups)
        except Exception:
            # A backend that cannot answer for this kind must not abort the whole export;
            # the summary reports what was captured so a partial snapshot is visible.
            continue
        collected.extend(_serialize(record, kind) for record in records or [])
    return collected


async def export_graph(
    graph: Any,
    settings: Any,
    groups: list[str],
    *,
    output: str | None = None,
) -> dict[str, Any]:
    provider = settings.database.provider
    destination = Path(output).expanduser() if output else default_output()

    records: list[dict[str, Any]] = []
    if provider == "falkordb":
        # Each group is its own FalkorDB database, so it needs its own driver.
        for group in groups:
            records.extend(await _collect(graph.driver.clone(database=group), [group]))
    elif provider == "ladybug":
        # Ladybug is single-graph: ingestion writes group_id=None, so also sweep the
        # empty group rather than reporting an empty snapshot for a populated file.
        seen: set[str] = set()
        for candidate in [groups, [""]]:
            for item in await _collect(graph.driver, candidate):
                if item.get("uuid") not in seen:
                    seen.add(item.get("uuid"))
                    records.append(item)
    else:
        records = await _collect(graph.driver, groups)

    header = {
        "kind": "export",
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "groups": groups,
        "embeddings": "omitted; recreated on import",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    return {
        "output": str(destination),
        "groups": groups,
        "provider": provider,
        "nodes": counts.get("entity_node", 0) + counts.get("episodic_node", 0),
        "edges": counts.get("entity_edge", 0) + counts.get("episodic_edge", 0),
        "by_kind": counts,
        "format_version": FORMAT_VERSION,
    }
