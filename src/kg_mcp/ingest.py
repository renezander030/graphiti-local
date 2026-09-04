"""Explicit, dry-run-by-default JSONL ingestion for Graphiti Local.

Ingestion is resumable. Each applied record is recorded in a content-keyed ledger, so
a re-run skips what already landed instead of duplicating it, and one failing record
no longer costs the whole batch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_mcp.config import allowed_groups, load_config

LEDGER = "ingest-ledger.jsonl"


def read_records(path: Path, *, skip_invalid: bool = False) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if skip_invalid:
                continue
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            if skip_invalid:
                continue
            raise SystemExit(f"{path}:{line_number}: each line must be an object")
        if line_number == 1 and record.get("kind") == "export":
            raise SystemExit(f"{path}: this is a kg export snapshot; pass --restore to replay it")
        missing = [name for name in ("name", "body") if not str(record.get(name, "")).strip()]
        if missing:
            if skip_invalid:
                continue
            raise SystemExit(f"{path}:{line_number}: missing {missing[0]}")
        records.append(record)
    return records


def _reference_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def record_key(record: dict[str, Any], domain: str) -> str:
    """Content hash so a re-run recognises a record it already ingested."""
    material = "\x1f".join(
        [
            domain,
            str(record.get("name", "")).strip(),
            str(record.get("body", "")).strip(),
            str(record.get("valid_at") or ""),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _ledger_path() -> Path:
    from kg_mcp.workspace import workspace_dir

    return workspace_dir() / LEDGER


def completed_keys() -> set[str]:
    path = _ledger_path()
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("key"):
            keys.add(entry["key"])
    return keys


def _record_completed(key: str, domain: str, name: str) -> None:
    from filelock import FileLock

    from kg_mcp.workspace import workspace_dir

    directory = workspace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "key": key,
        "domain": domain,
        "name": name,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    with (
        FileLock(str(directory / ".lock")),
        (directory / LEDGER).open("a", encoding="utf-8") as handle,
    ):
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


@contextmanager
def _stop_on_signal():
    """Turn SIGTERM/SIGINT into a stop request honoured at the next record boundary.

    An unattended run is usually wrapped in a timeout. Dying mid-write leaves an
    embedded backend with a partial write it may refuse to reopen, so a termination
    signal finishes the record in flight, closes the driver, and exits.
    """
    state = {"requested": False}

    def request_stop(signum, frame):
        del signum, frame
        state["requested"] = True

    previous = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[number] = signal.signal(number, request_stop)
        except (ValueError, OSError):
            # Not the main thread, or the platform lacks the signal; nothing to restore.
            continue
    try:
        yield state
    finally:
        for number, handler in previous.items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                continue


async def ingest_records(
    records: list[dict[str, Any]],
    *,
    apply: bool,
    resume: bool = True,
    fail_fast: bool = False,
    ledger: bool = True,
) -> dict[str, Any]:
    settings = load_config()
    for record in records:
        domain = str(record.get("domain") or settings.graph.groups[0])
        allowed_groups(domain, settings)

    seen = completed_keys() if (resume and ledger) else set()
    planned, skipped = [], 0
    for record in records:
        domain = str(record.get("domain") or settings.graph.groups[0])
        key = record_key(record, domain)
        if key in seen:
            skipped += 1
            continue
        planned.append((record, domain, key))

    if not apply:
        return {
            "applied": False,
            "planned": [{"domain": d, "name": r["name"]} for r, d, _ in planned],
            "skipped": skipped,
            "ingested": 0,
            "failed": [],
            "interrupted": False,
        }

    from graphiti_core.nodes import EpisodeType

    from kg_mcp import fingerprint
    from kg_mcp.runtime import build_graphiti

    # Vectors from two embedders in one graph rank wrongly and say nothing; refuse before
    # the first write rather than after.
    drift = fingerprint.drift(settings)
    if drift:
        raise ValueError(drift)

    graph = build_graphiti(settings, read_only=False)
    ingested, failures, interrupted = 0, [], False
    try:
        with _stop_on_signal() as stop:
            await graph.build_indices_and_constraints()
            for record, domain, key in planned:
                if stop["requested"]:
                    interrupted = True
                    break
                try:
                    await graph.add_episode(
                        name=str(record["name"]).strip(),
                        episode_body=str(record["body"]).strip(),
                        source=EpisodeType.text,
                        source_description=str(
                            record.get("provenance") or "Graphiti Local JSONL import"
                        ),
                        reference_time=_reference_time(record.get("valid_at")),
                        group_id=None if settings.database.provider == "ladybug" else domain,
                    )
                except Exception as exc:  # one bad episode must not cost the batch
                    failures.append({"name": record["name"], "domain": domain, "error": str(exc)})
                    if fail_fast:
                        break
                    continue
                if ledger:
                    _record_completed(key, domain, str(record["name"]).strip())
                ingested += 1
        if ingested:
            fingerprint.record(settings)
    finally:
        # Always close: an embedded backend left with a partial write may refuse to reopen.
        await graph.close()
    return {
        "applied": True,
        "ingested": ingested,
        "skipped": skipped,
        "failed": failures,
        "interrupted": interrupted,
        "planned": [],
    }


def main() -> None:
    from kg_mcp.output import emit, fail

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("-H", "--human", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ingest every record even if the ledger already recorded it",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failing record instead of isolating it",
    )
    parser.add_argument("--skip-invalid", action="store_true", help="ignore malformed input lines")
    parser.add_argument(
        "--restore",
        action="store_true",
        help="the input is a kg export snapshot; replay it instead of ingesting episodes",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="with --restore: the group every restored record is written to",
    )
    args = parser.parse_args()
    as_json = not args.human
    if args.restore:
        _restore_main(args, as_json=as_json)
        return
    try:
        records = read_records(args.input, skip_invalid=args.skip_invalid)
        result = asyncio.run(
            ingest_records(
                records,
                apply=args.apply,
                resume=not args.no_resume,
                fail_fast=args.fail_fast,
            )
        )
    except ValueError as exc:
        fail(str(exc), code=2, as_json=as_json)
        return
    except (FileNotFoundError, RuntimeError) as exc:
        fail(str(exc), as_json=as_json)
        return
    result["total"] = len(records)

    def human() -> list[str]:
        if not result["applied"]:
            lines = [
                f"would ingest [{item['domain']}] {item['name']}" for item in result["planned"]
            ]
            lines.append(
                f"{len(result['planned'])} episode(s) validated, "
                f"{result['skipped']} already ingested"
            )
            return lines
        lines = [f"{result['ingested']} episode(s) ingested, {result['skipped']} skipped"]
        if result.get("interrupted"):
            lines.append("stopped early on a termination signal; re-run to continue")
        lines.extend(f"FAILED [{f['domain']}] {f['name']}: {f['error']}" for f in result["failed"])
        return lines

    emit(result, human=human, as_json=as_json)
    if result["failed"]:
        raise SystemExit(1)


def _restore_main(args: argparse.Namespace, *, as_json: bool) -> None:
    from kg_mcp.output import emit, fail
    from kg_mcp.restore import human_lines, restore_snapshot

    try:
        result = asyncio.run(
            restore_snapshot(
                args.input,
                apply=args.apply,
                group=args.group,
                fail_fast=args.fail_fast,
            )
        )
    except ValueError as exc:
        fail(str(exc), code=2, as_json=as_json)
        return
    except (FileNotFoundError, RuntimeError) as exc:
        fail(str(exc), as_json=as_json)
        return
    emit(result, human=lambda: human_lines(result), as_json=as_json)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
