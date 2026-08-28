"""Append-only proposal queue with an explicit human approval boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

FACT_TYPES = ("belief", "source-fact", "action-record", "live-finding")
OPERATIONS = ("assert", "revise", "invalidate")
# Must accept every name GraphConfig accepts, or a legally configured group could be
# declared in config and still be unproposable.
DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def workspace_dir() -> Path:
    from kg_mcp.config import load_config

    override = os.environ.get("KG_WORKSPACE_DIR")
    configured = load_config().graph.workspace_dir
    return Path(override or configured).expanduser().resolve()


def _path(name: str) -> Path:
    return workspace_dir() / name


def _read(name: str = "pending.jsonl") -> list[dict]:
    path = _path(name)
    if not path.exists():
        return []
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: corrupt queue JSON") from exc
        if isinstance(item, dict):
            items.append(item)
    return items


def _write(name: str, items: list[dict]) -> None:
    directory = workspace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(name)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _append(name: str, item: dict) -> None:
    directory = workspace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with _path(name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def pending_for(groups: list[str] | None = None) -> list[dict]:
    pending = [item for item in _read() if item.get("status") == "pending"]
    if groups is None:
        return pending
    return [item for item in pending if item.get("domain") in groups]


def add_proposal(
    domain: str,
    text: str,
    *,
    fact_type: str = "belief",
    provenance: str = "",
    operation: str = "assert",
    supersedes: str = "",
) -> dict:
    if not DOMAIN_PATTERN.fullmatch(domain):
        raise SystemExit("domain must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    if fact_type not in FACT_TYPES:
        raise SystemExit(f"fact type must be one of: {', '.join(FACT_TYPES)}")
    if operation not in OPERATIONS:
        raise SystemExit(f"operation must be one of: {', '.join(OPERATIONS)}")
    text = text.strip()
    if not text:
        raise SystemExit("fact text must not be empty")
    if operation in ("revise", "invalidate") and not supersedes.strip():
        raise SystemExit(f"{operation} requires --supersedes")

    item = {
        "id": "proposal-" + uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "operation": operation,
        "text": text,
        "fact_type": fact_type,
        "provenance": provenance.strip(),
        "supersedes": supersedes.strip(),
        "status": "pending",
    }
    workspace_dir().mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_path(".lock")))
    with lock:
        _append("pending.jsonl", item)
    return item


def _human_set_status(ids: set[str], status: str) -> int:
    workspace_dir().mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_path(".lock")))
    with lock:
        items = _read()
        changed = 0
        for item in items:
            if item.get("id") in ids and item.get("status") == "pending":
                item["status"] = status
                item["decided_at"] = datetime.now(timezone.utc).isoformat()
                changed += 1
        _write("pending.jsonl", items)
    return changed


def _episode_body(item: dict) -> str:
    if item["operation"] == "assert":
        return item["text"]
    return (
        f"Correction as of {item['created_at'][:10]}: {item['text']} "
        f"This supersedes: {item['supersedes']}"
    )


def _archive(item_id: str) -> None:
    workspace_dir().mkdir(parents=True, exist_ok=True)
    with FileLock(str(_path(".lock"))):
        remaining = []
        for queued in _read():
            if queued.get("id") == item_id:
                queued["drained_at"] = datetime.now(timezone.utc).isoformat()
                _append("archive.jsonl", queued)
            else:
                remaining.append(queued)
        _write("pending.jsonl", remaining)


async def drain(*, apply: bool) -> dict:
    approved = [item for item in _read() if item.get("status") == "approved"]
    if not approved or not apply:
        return {
            "applied": False,
            "ingested": 0,
            "failed": [],
            "planned": [
                {"id": item["id"], "domain": item["domain"], "body": _episode_body(item)}
                for item in approved
            ],
        }

    from kg_mcp.ingest import ingest_records

    completed, failures = 0, []
    for item in approved:
        record = {
            "name": f"workspace {item['id']}",
            "body": _episode_body(item),
            "domain": item["domain"],
            "valid_at": item["created_at"],
            "provenance": (
                f"Graphiti Local workspace; {item['fact_type']}; "
                f"{item.get('provenance') or 'unspecified'}"
            ),
        }
        result = await ingest_records([record], apply=True, resume=False)
        # Archive only what actually landed. An isolated failure leaves the proposal
        # approved so the next drain retries it instead of losing it.
        if result["ingested"] != 1:
            error = result["failed"][0]["error"] if result["failed"] else "not ingested"
            failures.append({"id": item["id"], "domain": item["domain"], "error": error})
            continue
        _archive(item["id"])
        completed += 1
    return {"applied": True, "ingested": completed, "failed": failures, "planned": []}


def main() -> None:
    from kg_mcp.output import emit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--human", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("domain", nargs="?")
    for command in ("approve", "reject"):
        decision = subparsers.add_parser(command)
        decision.add_argument("ids", nargs="+")
    drain_parser = subparsers.add_parser("drain")
    drain_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    as_json = not args.human

    if args.command == "list":
        items = pending_for([args.domain] if args.domain else None)
        payload = {"pending": items}

        def human() -> list[str]:
            lines = [
                f"{item['id']} [{item['domain']} {item['fact_type']} {item['operation']}] "
                f"{item['text']}"
                for item in items
            ]
            return [*lines, f"({len(items)} pending)"]

        emit(payload, human=human, as_json=as_json)
        return
    if args.command in ("approve", "reject"):
        status = "approved" if args.command == "approve" else "rejected"
        changed = _human_set_status(set(args.ids), status)
        emit(
            {"status": status, "changed": changed},
            human=lambda: [f"{changed} proposal(s) {status}"],
            as_json=as_json,
        )
        return

    result = asyncio.run(drain(apply=args.apply))

    def drain_human() -> list[str]:
        if not result["applied"]:
            if not result["planned"]:
                return ["nothing approved"]
            lines = [
                f"would ingest {item['id']} [{item['domain']}] {item['body'][:80]}"
                for item in result["planned"]
            ]
            return [*lines, "dry run; add --apply"]
        lines = [f"{result['ingested']} proposal(s) ingested"]
        lines.extend(
            f"FAILED {item['id']} [{item['domain']}]: {item['error']}" for item in result["failed"]
        )
        return lines

    emit(result, human=drain_human, as_json=as_json)
    if result["failed"]:
        raise SystemExit(1)
