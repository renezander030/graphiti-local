"""Typed, auditable decisions over a staged extraction snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_mcp.export import _atomic_snapshot, _json_line
from kg_mcp.restore import read_snapshot

REVIEW_FORMAT_VERSION = 1
DECISIONS = {"accept", "refuse", "contested", "review"}


def default_decisions_path(snapshot: Path) -> Path:
    return snapshot.with_suffix(snapshot.suffix + ".review.jsonl")


def default_approved_path(snapshot: Path) -> Path:
    return snapshot.with_suffix(snapshot.suffix + ".approved.jsonl")


def create_decision_template(snapshot: Path, output: Path | None = None) -> dict[str, Any]:
    """Write one explicit decision row for every extracted fact edge."""
    header, records = read_snapshot(snapshot)
    destination = output or default_decisions_path(snapshot)
    edges = [record for record in records if record["kind"] == "entity_edge"]
    review_header = {
        "kind": "review",
        "format_version": REVIEW_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": header.get("sha256"),
        "snapshot_record_count": len(records),
        "decision_scope": "entity_edge",
        "instructions": "change each decision from review to accept, refuse, or contested",
    }
    decision_rows = [
        {
            "kind": "decision",
            "record_kind": "entity_edge",
            "uuid": edge["uuid"],
            "fact": edge.get("fact", ""),
            "decision": "review",
            "reason": "",
        }
        for edge in edges
    ]
    _atomic_snapshot(destination, review_header, decision_rows)
    return {
        "output": str(destination),
        "facts": len(edges),
        "snapshot_sha256": header.get("sha256"),
    }


def _read_decisions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: each line must be an object")
        if header is None:
            if item.get("kind") != "review":
                raise ValueError(f"{path}: not a review decision file")
            if item.get("format_version") != REVIEW_FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported review format {item.get('format_version')}")
            header = item
            continue
        if item.get("kind") != "decision":
            raise ValueError(f"{path}:{line_number}: expected a decision row")
        if item.get("record_kind") != "entity_edge":
            raise ValueError(f"{path}:{line_number}: decisions are scoped to entity_edge")
        if item.get("decision") not in DECISIONS:
            raise ValueError(f"{path}:{line_number}: invalid decision {item.get('decision')!r}")
        if item.get("decision") == "refuse" and not str(item.get("reason") or "").strip():
            raise ValueError(f"{path}:{line_number}: refused facts require a reason")
        rows.append(item)
    if header is None:
        raise ValueError(f"{path}: empty review decision file")
    return header, rows


def apply_review_decisions(
    snapshot: Path,
    decisions: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Create a checksum-valid promotion snapshot containing accepted fact edges only."""
    snapshot_header, records = read_snapshot(snapshot)
    review_header, rows = _read_decisions(decisions)
    if review_header.get("snapshot_sha256") != snapshot_header.get("sha256"):
        raise ValueError(
            f"{decisions}: decisions target snapshot {review_header.get('snapshot_sha256')}, "
            f"not {snapshot_header.get('sha256')}"
        )

    expected = {
        record["uuid"] for record in records if record.get("kind") == "entity_edge"
    }
    by_uuid: dict[str, dict[str, Any]] = {}
    for row in rows:
        uuid = str(row.get("uuid") or "")
        if uuid in by_uuid:
            raise ValueError(f"{decisions}: duplicate decision for {uuid}")
        if uuid not in expected:
            raise ValueError(f"{decisions}: decision references unknown fact edge {uuid}")
        by_uuid[uuid] = row
    missing = sorted(expected - set(by_uuid))
    if missing:
        raise ValueError(f"{decisions}: no decision for fact edge {missing[0]}")

    unresolved = [row for row in rows if row["decision"] in {"review", "contested"}]
    if unresolved:
        first = unresolved[0]
        raise ValueError(
            f"{decisions}: {len(unresolved)} fact decision(s) are unresolved; "
            f"first is {first['uuid']} ({first['decision']})"
        )

    refused = {uuid for uuid, row in by_uuid.items() if row["decision"] == "refuse"}
    selected: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        if record.get("kind") == "entity_edge" and record.get("uuid") in refused:
            continue
        if record.get("kind") == "episodic_node" and isinstance(
            record.get("entity_edges"), list
        ):
            record["entity_edges"] = [
                uuid for uuid in record["entity_edges"] if uuid not in refused
            ]
        selected.append(record)

    record_bytes = [_json_line(record) for record in selected]
    digest = hashlib.sha256(b"".join(record_bytes)).hexdigest()
    decisions_digest = hashlib.sha256(decisions.read_bytes()).hexdigest()
    promotion_header = dict(snapshot_header)
    promotion_header.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        record_count=len(selected),
        sha256=digest,
        review={
            "source_sha256": snapshot_header.get("sha256"),
            "decisions_sha256": decisions_digest,
            "accepted": len(expected) - len(refused),
            "refused": len(refused),
        },
    )
    destination = output or default_approved_path(snapshot)
    _atomic_snapshot(destination, promotion_header, selected)
    return {
        "output": str(destination),
        "source_sha256": snapshot_header.get("sha256"),
        "sha256": digest,
        "decisions_sha256": decisions_digest,
        "accepted": len(expected) - len(refused),
        "refused": len(refused),
        "record_count": len(selected),
    }
