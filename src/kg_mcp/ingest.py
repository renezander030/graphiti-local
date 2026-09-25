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
import re
import shlex
import signal
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_mcp.config import Settings, allowed_groups, load_config

LEDGER = "ingest-ledger.jsonl"
RECEIPTS = "ingest-receipts.jsonl"
FAILURES = "ingest-failures.jsonl"
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


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


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    """Split one overlong paragraph without losing any non-whitespace text."""
    pieces: list[str] = []
    remaining = piece.strip()
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        boundaries = [window.rfind(mark) for mark in ("\n", ". ", "! ", "? ", " ")]
        cut = max(boundaries)
        if cut < max_chars // 2:
            cut = max_chars
        elif window[cut : cut + 2] in {". ", "! ", "? "}:
            cut += 1
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Make stable, paragraph-aware chunks bounded by ``max_chars``."""
    stripped = text.strip()
    if not stripped:
        return []
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
    atomic = [
        piece
        for paragraph in paragraphs
        for piece in _split_long_piece(paragraph, max_chars)
    ]
    chunks: list[str] = []
    current = ""
    for piece in atomic:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _chunk_record(record: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    chunks = chunk_text(str(record.get("body", "")), max_chars)
    if not chunks:
        return []
    if len(chunks) == 1:
        item = dict(record)
        item["body"] = chunks[0]
        return [item]
    name = str(record.get("name", "document")).strip()
    result = []
    for index, body in enumerate(chunks, 1):
        item = dict(record)
        item.update(
            name=f"{name} [part {index}/{len(chunks)}]",
            body=body,
            chunk_index=index,
            chunk_count=len(chunks),
        )
        result.append(item)
    return result


def read_input(
    path: Path,
    *,
    input_format: str = "auto",
    domain: str | None = None,
    max_chars: int = 12000,
    skip_invalid: bool = False,
) -> list[dict[str, Any]]:
    """Read JSONL/JSON records or local text documents and directories.

    Directory discovery is deterministic, does not follow paths that resolve outside
    the requested root, and fails when no supported non-empty source was found.
    """
    source = path.expanduser()
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_dir():
        root = source.resolve()
        records: list[dict[str, Any]] = []
        for candidate in sorted(item for item in source.rglob("*") if item.is_file()):
            resolved = candidate.resolve()
            relative = candidate.relative_to(source)
            hidden = any(part.startswith(".") for part in relative.parts)
            if not resolved.is_relative_to(root) or hidden:
                continue
            suffix = candidate.suffix.lower()
            if suffix not in TEXT_SUFFIXES | {".json", ".jsonl"}:
                continue
            discovered = read_input(
                candidate,
                input_format="auto",
                domain=domain,
                max_chars=max_chars,
                skip_invalid=skip_invalid,
            )
            for record in discovered:
                if record.get("name") == candidate.name:
                    record["name"] = relative.as_posix()
                    record["provenance"] = f"local document: {relative.as_posix()}"
            records.extend(discovered)
        if not records:
            raise ValueError(f"{source}: no supported non-empty input files")
        return records

    selected = input_format
    if selected == "auto":
        selected = (
            "jsonl"
            if source.suffix.lower() == ".jsonl"
            else "json"
            if source.suffix.lower() == ".json"
            else "text"
            if source.suffix.lower() in TEXT_SUFFIXES
            else ""
        )
    if selected == "jsonl":
        raw_records = read_records(source, skip_invalid=skip_invalid)
    elif selected == "json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
        items = payload if isinstance(payload, list) else [payload]
        if not items or not all(isinstance(item, dict) for item in items):
            raise ValueError(f"{source}: JSON input must be an object or non-empty object list")
        raw_records = list(items)
        for item in raw_records:
            if not str(item.get("name", "")).strip() or not str(item.get("body", "")).strip():
                raise ValueError(f"{source}: every JSON record needs non-empty name and body")
    elif selected == "text":
        body = source.read_text(encoding="utf-8").strip()
        if not body:
            raise ValueError(f"{source}: input contains no text")
        raw_records = [
            {
                "name": source.name,
                "body": body,
                "provenance": f"local document: {source.name}",
            }
        ]
    else:
        raise ValueError(
            f"{source}: unsupported input type; use .jsonl, .json, .md, .markdown, or .txt"
        )

    prepared: list[dict[str, Any]] = []
    for record in raw_records:
        item = dict(record)
        if domain and not item.get("domain"):
            item["domain"] = domain
        prepared.extend(_chunk_record(item, max_chars))
    if not prepared:
        raise ValueError(f"{source}: input produced zero records")
    return prepared


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


def episode_uuid(key: str) -> str:
    """Stable episode identity used across retries and reconciliations."""
    return str(uuid.UUID(hex=key))


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


def _record_event(filename: str, entry: dict[str, Any]) -> None:
    from filelock import FileLock

    from kg_mcp.workspace import workspace_dir

    directory = workspace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".lock")), (directory / filename).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, json.JSONDecodeError)):
        return True
    if type(exc).__name__ in {"RateLimitError", "EmptyResponseError", "ValidationError"}:
        return True
    detail = str(exc).lower()
    return any(
        token in detail
        for token in (
            "timeout",
            "timed out",
            "temporar",
            "rate limit",
            "empty response",
            "connection",
            "unavailable",
            "try again",
        )
    )


def _retry_delay(base: float, attempt: int, key: str) -> float:
    if base <= 0:
        return 0.0
    jitter = (int(key[:8], 16) / 0xFFFFFFFF) * base * 0.25
    return base * (2 ** (attempt - 1)) + jitter


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
    settings: Settings | None = None,
    track_fingerprint: bool = True,
) -> dict[str, Any]:
    settings = settings or load_config()
    for record in records:
        domain = str(record.get("domain") or settings.graph.groups[0])
        allowed_groups(domain, settings)

    from kg_mcp.config import per_group_ladybug, settings_for_group

    if per_group_ladybug(settings):
        # One file per group: each domain's records go to its own file, under its own
        # writer lock and embedder record, and never open another group's file.
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_domain.setdefault(
                str(record.get("domain") or settings.graph.groups[0]), []
            ).append(record)
        merged: dict[str, Any] = {
            "applied": apply,
            "ingested": 0,
            "skipped": 0,
            "failed": [],
            "interrupted": False,
            "planned": [],
            "receipts": [],
        }
        for domain, subset in by_domain.items():
            part = await ingest_records(
                subset,
                apply=apply,
                resume=resume,
                fail_fast=fail_fast,
                ledger=ledger,
                settings=settings_for_group(settings, domain),
                track_fingerprint=track_fingerprint,
            )
            for key in ("ingested", "skipped"):
                merged[key] += part[key]
            merged["failed"].extend(part["failed"])
            merged["planned"].extend(part["planned"])
            merged["receipts"].extend(part["receipts"])
            merged["interrupted"] = merged["interrupted"] or part["interrupted"]
            if part["interrupted"] or (fail_fast and part["failed"]):
                break
        return merged

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
            "receipts": [],
        }

    from graphiti_core.nodes import EpisodeType

    from kg_mcp import fingerprint
    from kg_mcp.runtime import build_graphiti

    # Vectors from two embedders in one graph rank wrongly and say nothing; refuse before
    # the first write rather than after.
    if track_fingerprint:
        drift = fingerprint.drift(settings)
        if drift:
            raise ValueError(drift)

    from kg_mcp.write_lock import graph_writer_lock

    ingested, failures, receipts, interrupted = 0, [], [], False
    with graph_writer_lock(settings):
        graph = build_graphiti(settings, read_only=False)
        try:
            with _stop_on_signal() as stop:
                await graph.build_indices_and_constraints()
                for record, domain, key in planned:
                    if stop["requested"]:
                        interrupted = True
                        break
                    result = None
                    attempts = 0
                    last_error: Exception | None = None
                    while attempts < settings.graph.ingest_max_attempts:
                        attempts += 1
                        try:
                            result = await graph.add_episode(
                                name=str(record["name"]).strip(),
                                episode_body=str(record["body"]).strip(),
                                source=EpisodeType.text,
                                source_description=str(
                                    record.get("provenance") or "Graphiti Local import"
                                ),
                                reference_time=_reference_time(record.get("valid_at")),
                                group_id=(
                                    None if settings.database.provider == "ladybug" else domain
                                ),
                                uuid=episode_uuid(key),
                            )
                            last_error = None
                            break
                        except Exception as exc:  # one bad episode must not cost the batch
                            last_error = exc
                            exhausted = attempts >= settings.graph.ingest_max_attempts
                            if not _retryable(exc) or exhausted:
                                break
                            delay = _retry_delay(
                                settings.graph.ingest_retry_base_seconds, attempts, key
                            )
                            if delay:
                                await asyncio.sleep(delay)
                    if last_error is not None:
                        exc = last_error
                        detail = str(exc).strip() or f"{type(exc).__name__} (no message)"
                        failure = {
                            "key": key,
                            "episode_uuid": episode_uuid(key),
                            "name": record["name"],
                            "domain": domain,
                            "operation": "extract episode",
                            "error_type": type(exc).__name__,
                            "error": detail,
                            "attempts": attempts,
                            "retryable": _retryable(exc),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        failures.append(failure)
                        if ledger:
                            _record_event(FAILURES, failure)
                        if fail_fast:
                            break
                        continue
                    if ledger:
                        _record_completed(key, domain, str(record["name"]).strip())
                    receipt = {
                        "key": key,
                        "episode_uuid": getattr(getattr(result, "episode", None), "uuid", None)
                        or episode_uuid(key),
                        "name": str(record["name"]).strip(),
                        "domain": domain,
                        "nodes": len(getattr(result, "nodes", []) or []),
                        "edges": len(getattr(result, "edges", []) or []),
                        "attempts": attempts,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    }
                    receipts.append(receipt)
                    if ledger:
                        _record_event(RECEIPTS, receipt)
                    ingested += 1
            if ingested and track_fingerprint:
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
        "receipts": receipts,
    }


async def stage_extraction(
    records: list[dict[str, Any]],
    output: Path,
    *,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Extract into a disposable graph and export the resolved facts for review."""
    settings = load_config()
    domains = {str(record.get("domain") or settings.graph.groups[0]) for record in records}
    for domain in domains:
        allowed_groups(domain, settings)
    if len(domains) != 1:
        raise ValueError(
            "--review-output accepts one domain per snapshot; split the input by domain"
        )
    domain = next(iter(domains))
    destination = output.expanduser().resolve()

    from kg_mcp.export import export_graph
    from kg_mcp.ladybug import setup_database
    from kg_mcp.runtime import build_graphiti

    with tempfile.TemporaryDirectory(prefix="graphiti-local-review-") as directory:
        root = Path(directory)
        database_path = root / "review.ladybug"
        stage_graph = settings.graph.model_copy(
            update={"groups": [domain], "workspace_dir": str(root / "workspace")}
        )
        stage_database = settings.database.model_copy(
            update={
                "provider": "ladybug",
                "ladybug": settings.database.ladybug.model_copy(
                    update={"path": str(database_path), "layout": "single"}
                ),
            }
        )
        stage_settings = settings.model_copy(
            update={"graph": stage_graph, "database": stage_database}
        )
        setup_database(
            database_path,
            quiet=True,
            lock_timeout=stage_graph.writer_lock_timeout_seconds,
        )
        extraction = await ingest_records(
            records,
            apply=True,
            resume=False,
            fail_fast=fail_fast,
            ledger=False,
            settings=stage_settings,
            track_fingerprint=False,
        )
        graph = build_graphiti(stage_settings, read_only=True)
        try:
            snapshot = await export_graph(
                graph,
                stage_settings,
                [domain],
                output=str(destination),
            )
        finally:
            await graph.close()

    decision_template = None
    if destination.exists():
        from kg_mcp.review import create_decision_template

        decision_template = create_decision_template(destination)

    review_flags = ""
    if decision_template is not None:
        approved = destination.with_suffix(destination.suffix + ".approved.jsonl")
        review_flags = (
            f" --review-decisions {shlex.quote(decision_template['output'])}"
            f" --approved-output {shlex.quote(str(approved))}"
        )

    return {
        "staged": True,
        "production_graph_changed": False,
        "domain": domain,
        "extracted": extraction["ingested"],
        "failed": extraction["failed"],
        "review_snapshot": snapshot,
        "review_decisions": decision_template,
        "promote": (
            f"kg-ingest {shlex.quote(str(destination))} --restore "
            f"{review_flags.strip()} "
            f"--group {shlex.quote(domain)} --apply"
        ).replace("  ", " "),
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
        "--input-format",
        choices=("auto", "jsonl", "json", "text"),
        default="auto",
        help="input type (default: infer it from the file extension)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="group for document/JSON input records that do not already name one",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="paragraph-aware maximum episode size; 0 disables splitting",
    )
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
    parser.add_argument(
        "--review-decisions",
        type=Path,
        help="with --restore: typed decisions for every staged fact edge",
    )
    parser.add_argument(
        "--approved-output",
        type=Path,
        help="with --review-decisions: destination for the filtered promotion snapshot",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        help=(
            "extract into a disposable graph and write a reviewable snapshot; "
            "the configured production graph is not changed"
        ),
    )
    args = parser.parse_args()
    as_json = not args.human
    if (args.review_decisions or args.approved_output) and not args.restore:
        fail(
            "--review-decisions and --approved-output require --restore",
            code=2,
            as_json=as_json,
        )
        return
    if args.approved_output and not args.review_decisions:
        fail("--approved-output requires --review-decisions", code=2, as_json=as_json)
        return
    if args.review_output:
        if args.apply or args.restore:
            fail(
                "--review-output cannot be combined with --apply or --restore",
                code=2,
                as_json=as_json,
            )
            return
        try:
            records = read_input(
                args.input,
                input_format=args.input_format,
                domain=args.domain,
                max_chars=args.max_chars,
                skip_invalid=args.skip_invalid,
            )
            result = asyncio.run(
                stage_extraction(records, args.review_output, fail_fast=args.fail_fast)
            )
        except ValueError as exc:
            fail(str(exc), code=2, as_json=as_json)
            return
        except (FileNotFoundError, RuntimeError) as exc:
            fail(str(exc), as_json=as_json)
            return

        def review_human() -> list[str]:
            snapshot = result["review_snapshot"]
            lines = [
                f"staged {result['extracted']} episode(s) without changing the production graph",
                f"review {snapshot['record_count']} graph record(s) in {snapshot['output']}",
                f"promote unchanged: {result['promote']}",
            ]
            if result.get("review_decisions"):
                lines.append(
                    f"record accept/refuse/contested decisions in "
                    f"{result['review_decisions']['output']}"
                )
            lines.extend(
                f"FAILED [{item['domain']}] {item['name']}: {item['error']}"
                for item in result["failed"]
            )
            return lines

        emit(result, human=review_human, as_json=as_json)
        if result["failed"]:
            raise SystemExit(1)
        return
    if args.restore:
        _restore_main(args, as_json=as_json)
        return
    try:
        records = read_input(
            args.input,
            input_format=args.input_format,
            domain=args.domain,
            max_chars=args.max_chars,
            skip_invalid=args.skip_invalid,
        )
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
        source = args.input
        review_receipt = None
        if args.review_decisions:
            from kg_mcp.review import apply_review_decisions

            review_receipt = apply_review_decisions(
                args.input,
                args.review_decisions,
                args.approved_output,
            )
            source = Path(review_receipt["output"])
        result = asyncio.run(
            restore_snapshot(
                source,
                apply=args.apply,
                group=args.group,
                fail_fast=args.fail_fast,
            )
        )
        if review_receipt is not None:
            result["review_receipt"] = review_receipt
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
