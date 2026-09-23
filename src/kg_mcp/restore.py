"""Replay a ``kg export`` snapshot into the configured backend.

The snapshot carries the graphiti models without embeddings, so a restore re-embeds
every entity name and fact under the configured embedder and saves nodes before the
edges that reference them. Records are saved by their original UUIDs, so restoring a
snapshot twice updates rather than duplicates. Like ingestion, it is a dry run without
``--apply`` and one failing record does not cost the batch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kg_mcp.config import Settings, allowed_groups, load_config
from kg_mcp.export import EMBEDDING_FIELDS, FORMAT_VERSION

KINDS = ("entity_node", "episodic_node", "entity_edge", "episodic_edge")
SUPPORTED_FORMATS = (1, FORMAT_VERSION)


def read_snapshot(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: each line must be an object")
        if header is None:
            if item.get("kind") != "export":
                raise ValueError(f"{path}: not a kg export snapshot (no export header)")
            if item.get("format_version") not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"{path}: snapshot format {item.get('format_version')} is not "
                    f"supported (expected one of {SUPPORTED_FORMATS})"
                )
            header = item
            continue
        if item.get("kind") not in KINDS:
            raise ValueError(f"{path}:{line_number}: unknown record kind {item.get('kind')!r}")
        records.append(item)
        digest.update(line.rstrip("\r\n").encode("utf-8") + b"\n")
    if header is None:
        raise ValueError(f"{path}: empty file, not a kg export snapshot")
    if header.get("format_version") == FORMAT_VERSION:
        expected_count = header.get("record_count")
        if expected_count != len(records):
            raise ValueError(
                f"{path}: snapshot record count mismatch: expected {expected_count}, "
                f"found {len(records)}"
            )
        actual_digest = digest.hexdigest()
        if header.get("sha256") != actual_digest:
            raise ValueError(
                f"{path}: snapshot checksum mismatch: expected {header.get('sha256')}, "
                f"found {actual_digest}"
            )
    return header, records


def target_group(record: dict[str, Any], override: str | None, settings: Settings) -> str:
    if settings.database.provider == "ladybug":
        # Ladybug is single-graph; ingestion stores the empty group there.
        return ""
    group = override or str(record.get("group_id") or "")
    if not group:
        raise ValueError(
            f"record {record.get('uuid')} has no group_id; pass --group to choose the "
            f"target group on {settings.database.provider}"
        )
    allowed_groups(group, settings)
    return group


def file_group(record: dict[str, Any], override: str | None, settings: Settings) -> str:
    """The group whose file a record is restored into, with the per-group Ladybug layout."""
    group = override or str(record.get("group_id") or "")
    if not group:
        raise ValueError(
            f"record {record.get('uuid')} has no group_id; pass --group to choose the "
            "group file it is restored into"
        )
    allowed_groups(group, settings)
    return group


def build_model(record: dict[str, Any], group: str) -> Any:
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodicNode

    models = {
        "entity_node": EntityNode,
        "episodic_node": EpisodicNode,
        "entity_edge": EntityEdge,
        "episodic_edge": EpisodicEdge,
    }
    payload = {
        key: value for key, value in record.items() if key != "kind" and key not in EMBEDDING_FIELDS
    }
    payload["group_id"] = group
    return models[record["kind"]].model_validate(payload)


async def _embed(model: Any, embedder: Any) -> None:
    if hasattr(model, "generate_name_embedding"):
        await model.generate_name_embedding(embedder)
    elif hasattr(model, "generate_embedding"):
        await model.generate_embedding(embedder)


def _driver_for(graph: Any, settings: Settings, group: str) -> Any:
    if settings.database.provider == "falkordb":
        return graph.driver.clone(database=group)
    return graph.driver


ENDPOINTS = {
    # An edge is written with MATCH on both endpoints; when one is missing the backend
    # creates nothing and reports nothing, so the endpoints are checked here.
    "entity_edge": ("Entity", "Entity"),
    "episodic_edge": ("Episodic", "Entity"),
}


async def _known_uuids(driver: Any, label: str) -> set[str]:
    rows, _, _ = await driver.execute_query(f"MATCH (n:{label}) RETURN n.uuid AS uuid")
    return {row["uuid"] for row in rows}


def _missing_endpoint(model: Any, kind: str, known: dict[str, set[str]]) -> str | None:
    source_label, target_label = ENDPOINTS[kind]
    if model.source_node_uuid not in known[source_label]:
        return f"source {source_label} {model.source_node_uuid} is not in the graph"
    if model.target_node_uuid not in known[target_label]:
        return f"target {target_label} {model.target_node_uuid} is not in the graph"
    return None


async def restore_snapshot(
    path: Path,
    *,
    apply: bool,
    group: str | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    from kg_mcp.config import per_group_ladybug, settings_for_group

    header, records = read_snapshot(path)
    settings = load_config()
    if group:
        allowed_groups(group, settings)
    per_group = per_group_ladybug(settings)
    if per_group:
        # One file per group: the record's group chooses the file; inside it the
        # record is stored like any other Ladybug record.
        files = [(record, file_group(record, group, settings)) for record in records]
        planned = [(record, "") for record, _ in files]
        groups = sorted({target for _, target in files})
    else:
        planned = [(record, target_group(record, group, settings)) for record in records]
        groups = sorted({target for _, target in planned})
    counts = {kind: 0 for kind in KINDS}
    for record, _ in planned:
        counts[record["kind"]] += 1
    summary = {
        "source": {
            "provider": header.get("provider"),
            "groups": header.get("groups"),
            "created_at": header.get("created_at"),
        },
        "groups": groups,
        "planned": counts,
    }
    if not apply:
        return {"applied": False, "restored": {}, "failed": [], **summary}

    if not per_group:
        restored, failures = await _restore_into(settings, planned, fail_fast=fail_fast)
        return {"applied": True, "restored": restored, "failed": failures, **summary}

    bound = {target: settings_for_group(settings, target) for target in groups}
    from kg_mcp import fingerprint

    for target in bound.values():
        message = fingerprint.drift(target)
        if message:
            raise ValueError(message)
    restored = {kind: 0 for kind in KINDS}
    failures: list[dict[str, Any]] = []
    for target in groups:
        subset = [(record, "") for record, chosen in files if chosen == target]
        part, failed = await _restore_into(bound[target], subset, fail_fast=fail_fast)
        for kind, count in part.items():
            restored[kind] += count
        failures.extend(failed)
        if fail_fast and failures:
            break
    return {"applied": True, "restored": restored, "failed": failures, **summary}


async def _restore_into(
    settings: Settings,
    planned: list[tuple[dict[str, Any], str]],
    *,
    fail_fast: bool,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    from kg_mcp import fingerprint
    from kg_mcp.runtime import build_graphiti
    from kg_mcp.write_lock import graph_writer_lock

    message = fingerprint.drift(settings)
    if message:
        raise ValueError(message)

    restored = {kind: 0 for kind in KINDS}
    failures: list[dict[str, Any]] = []
    with graph_writer_lock(settings):
        graph = build_graphiti(settings, read_only=False)
        try:
            await graph.build_indices_and_constraints()
            known: dict[str, dict[str, set[str]]] = {}
            for kind in KINDS:
                for record, target in planned:
                    if record["kind"] != kind:
                        continue
                    driver = _driver_for(graph, settings, target)
                    try:
                        model = build_model(record, target)
                        if kind in ENDPOINTS:
                            if target not in known:
                                known[target] = {
                                    label: await _known_uuids(driver, label)
                                    for label in ("Entity", "Episodic")
                                }
                            missing = _missing_endpoint(model, kind, known[target])
                            if missing:
                                raise LookupError(missing)
                        await _embed(model, graph.embedder)
                        await model.save(driver)
                    except Exception as exc:  # one bad record must not cost the snapshot
                        detail = str(exc).strip() or f"{type(exc).__name__} (no message)"
                        failures.append(
                            {
                                "kind": kind,
                                "uuid": record.get("uuid"),
                                "operation": "restore record",
                                "error_type": type(exc).__name__,
                                "error": detail,
                            }
                        )
                        if fail_fast:
                            break
                        continue
                    restored[kind] += 1
                if fail_fast and failures:
                    break
            if any(restored.values()):
                fingerprint.record(settings)
        finally:
            await graph.close()
    return restored, failures


def human_lines(result: dict[str, Any]) -> list[str]:
    counts = result["planned"]
    total = sum(counts.values())
    if not result["applied"]:
        lines = [f"would restore {count} {kind}(s)" for kind, count in counts.items() if count]
        lines.append(
            f"{total} record(s) from {result['source']['provider']} into "
            f"{', '.join(result['groups']) or 'the embedded graph'}; dry run, add --apply"
        )
        return lines
    done = sum(result["restored"].values())
    lines = [f"{done} of {total} record(s) restored"]
    lines.extend(
        f"FAILED {item['kind']} {item['uuid']}: {item['error']}" for item in result["failed"]
    )
    return lines
