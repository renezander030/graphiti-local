"""Portable JSONL snapshot of a graph.

The snapshot is backend-independent: it is written from the graphiti models rather than
from backend rows, so a graph captured on FalkorDB can be replayed into Neo4j or Ladybug.
Embeddings are omitted — they are derived from the text and re-created on import, and a
vector written back under a different embedding model would be silently wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT_VERSION = 2
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
    from graphiti_core.errors import GroupsEdgesNotFoundError
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
        except GroupsEdgesNotFoundError:
            # graphiti reports "no edges in these groups" as an error; it is an empty
            # result, not a partial collection.
            records = []
        except Exception as exc:
            raise RuntimeError(f"snapshot export could not collect {kind}: {exc}") from exc
        collected.extend(_serialize(record, kind) for record in records or [])
    return collected


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_snapshot(
    destination: Path, header: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_json_line(header))
            for record in records:
                handle.write(_json_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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

    record_bytes = [_json_line(record) for record in records]
    digest = hashlib.sha256(b"".join(record_bytes)).hexdigest()
    header = {
        "kind": "export",
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "groups": groups,
        "record_count": len(records),
        "sha256": digest,
        "embeddings": "omitted; derived from the text under the embedder in use",
    }
    _atomic_snapshot(destination, header, records)

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
        "sha256": digest,
        "record_count": len(records),
    }
