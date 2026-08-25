"""Explicit, dry-run-by-default JSONL ingestion for KG-MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_mcp.config import allowed_groups, load_config


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_number}: each line must be an object")
            missing = [name for name in ("name", "body") if not str(record.get(name, "")).strip()]
            if missing:
                raise SystemExit(f"{path}:{line_number}: missing {', '.join(missing)}")
            records.append(record)
    return records


def _reference_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


async def ingest_records(records: list[dict[str, Any]], *, apply: bool) -> int:
    settings = load_config()
    for record in records:
        domain = str(record.get("domain") or settings.graph.groups[0])
        allowed_groups(domain, settings)
    if not apply:
        for record in records:
            domain = record.get("domain") or settings.graph.groups[0]
            print(f"would ingest [{domain}] {record['name']}")
        return 0

    from graphiti_core.nodes import EpisodeType

    from kg_mcp.runtime import build_graphiti

    graph = build_graphiti(settings, read_only=False)
    completed = 0
    try:
        await graph.build_indices_and_constraints()
        for record in records:
            domain = str(record.get("domain") or settings.graph.groups[0])
            await graph.add_episode(
                name=str(record["name"]).strip(),
                episode_body=str(record["body"]).strip(),
                source=EpisodeType.text,
                source_description=str(record.get("provenance") or "kg-mcp JSONL import"),
                reference_time=_reference_time(record.get("valid_at")),
                group_id=None if settings.database.provider == "ladybug" else domain,
            )
            completed += 1
        return completed
    finally:
        await graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    records = read_records(args.input)
    count = asyncio.run(ingest_records(records, apply=args.apply))
    print(f"{count} episode(s) ingested" if args.apply else f"{len(records)} episode(s) validated")
