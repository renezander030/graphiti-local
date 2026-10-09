"""Append-only proposal queue with an explicit human approval boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from kg_mcp.gates import content_digest, rejected_matches, secret_shapes, sentences
from kg_mcp.output import (
    EXIT_NOT_ATOMIC,
    EXIT_REJECTED,
    EXIT_RESEMBLES_REJECTED,
    EXIT_SECRET_SHAPED,
    CommandError,
)

FACT_TYPES = ("belief", "source-fact", "action-record", "live-finding")
OPERATIONS = ("assert", "revise", "invalidate")
# Must accept every name GraphConfig accepts, or a legally configured group could be
# declared in config and still be unproposable.
DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Clock skew between the proposing machine and this one.
FUTURE_TOLERANCE = timedelta(minutes=5)


def workspace_dir() -> Path:
    from kg_mcp.config import load_config

    override = os.environ.get("KG_WORKSPACE_DIR")
    configured = load_config().graph.workspace_dir
    return Path(override or configured).expanduser().resolve()


def _workspace_settings():
    from kg_mcp.config import load_config

    return load_config().workspace


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


def _lock() -> FileLock:
    workspace_dir().mkdir(parents=True, exist_ok=True)
    return FileLock(str(_path(".lock")))


def pending_for(groups: list[str] | None = None) -> list[dict]:
    pending = [item for item in _read() if item.get("status") == "pending"]
    if groups is None:
        return pending
    return [item for item in pending if item.get("domain") in groups]


def rejected_for(groups: list[str] | None = None) -> list[dict]:
    """Rejected proposals, newest decision first, including any left in the queue file."""
    rejected = [
        item for item in _read("archive.jsonl") + _read() if item.get("status") == "rejected"
    ]
    if groups is not None:
        rejected = [item for item in rejected if item.get("domain") in groups]
    return sorted(rejected, key=lambda item: str(item.get("decided_at") or ""), reverse=True)


def _timestamp(value: str, flag: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise CommandError(
            f"{flag} must be an ISO date or datetime, got {value!r}", code=EXIT_REJECTED
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed > datetime.now(timezone.utc) + FUTURE_TOLERANCE:
        raise CommandError(
            f"{flag} {value} is in the future; a fact records what already happened",
            code=EXIT_REJECTED,
        )
    return parsed.isoformat()


def _check_fact(domain: str, fact: str, *, supersedes_rejection: str, settings) -> None:
    """The atomic and rejection gates for one fact text."""
    if settings.atomic_proposals:
        found = sentences(fact)
        if len(found) > 1:
            listed = "; ".join(f"{index}. {sentence}" for index, sentence in enumerate(found, 1))
            raise CommandError(
                f"proposal has {len(found)} sentences; propose one fact each, or file them "
                f"together with `kg bundle`: {listed}",
                code=EXIT_NOT_ATOMIC,
            )
    matches = rejected_matches(domain, fact, rejected_for([domain]), settings.reject_similarity)
    if matches and supersedes_rejection not in {item.get("id") for _, item in matches}:
        listed = "; ".join(
            f"{item['id']} ({ratio:.2f}) {item.get('text', '')[:120]}"
            + (f" (reason: {item['reason']})" if item.get("reason") else "")
            for ratio, item in matches[:5]
        )
        raise CommandError(
            f"proposal resembles {len(matches)} rejected proposal(s) in {domain}: {listed}. "
            "Re-propose with --supersedes-rejection <id> to acknowledge the rejection.",
            code=EXIT_RESEMBLES_REJECTED,
        )


def _new_item(
    domain: str,
    *,
    fact_type: str,
    provenance: str,
    source: str,
    link: str,
    valid_at: str,
    learned_at: str,
    supersedes_rejection: str,
    texts: list[str],
) -> dict:
    """Validate everything a proposal and a bundle share, and return the base record."""
    if not DOMAIN_PATTERN.fullmatch(domain):
        raise SystemExit("domain must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    if fact_type not in FACT_TYPES:
        raise SystemExit(f"fact type must be one of: {', '.join(FACT_TYPES)}")
    provenance, source, link = provenance.strip(), source.strip(), link.strip()
    shapes = secret_shapes(*texts, provenance, source, link)
    if shapes:
        raise CommandError(
            f"proposal carries a {', '.join(shapes)} shape; state the fact without the value",
            code=EXIT_SECRET_SHAPED,
        )
    settings = _workspace_settings()
    if settings.require_source and not (source and link):
        raise CommandError(
            "workspace.require_source is set: name the source system with --source and "
            "the record with --link",
            code=EXIT_REJECTED,
        )
    item: dict[str, Any] = {
        "id": "proposal-" + uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "fact_type": fact_type,
        "provenance": provenance,
    }
    if source:
        item["source"] = source
    if link:
        item["link"] = link
    if valid_at.strip():
        item["valid_at"] = _timestamp(valid_at, "--valid-at")
    if learned_at.strip():
        item["learned_at"] = _timestamp(learned_at, "--learned-at")
    if supersedes_rejection.strip():
        item["supersedes_rejection"] = supersedes_rejection.strip()
    return item


def add_proposal(
    domain: str,
    text: str,
    *,
    fact_type: str = "belief",
    provenance: str = "",
    operation: str = "assert",
    supersedes: str = "",
    valid_at: str = "",
    learned_at: str = "",
    source: str = "",
    link: str = "",
    supersedes_rejection: str = "",
) -> dict:
    if operation not in OPERATIONS:
        raise SystemExit(f"operation must be one of: {', '.join(OPERATIONS)}")
    text = text.strip()
    if not text:
        raise SystemExit("fact text must not be empty")
    if operation in ("revise", "invalidate") and not supersedes.strip():
        raise SystemExit(f"{operation} requires --supersedes")
    item = _new_item(
        domain,
        fact_type=fact_type,
        provenance=provenance,
        source=source,
        link=link,
        valid_at=valid_at,
        learned_at=learned_at,
        supersedes_rejection=supersedes_rejection,
        texts=[text, supersedes],
    )
    _check_fact(
        domain,
        text,
        supersedes_rejection=supersedes_rejection.strip(),
        settings=_workspace_settings(),
    )
    item.update(
        operation=operation,
        text=text,
        supersedes=supersedes.strip(),
        status="pending",
    )
    with _lock():
        _append("pending.jsonl", item)
    return item


def add_bundle(
    domain: str,
    facts: list[str],
    *,
    fact_type: str = "belief",
    provenance: str = "",
    source: str = "",
    link: str = "",
    valid_at: str = "",
    learned_at: str = "",
    supersedes_rejection: str = "",
) -> dict:
    """One reviewed item that asserts several facts from one source."""
    facts = [fact.strip() for fact in facts if fact.strip()]
    if len(facts) < 2:
        raise CommandError("a bundle needs at least two --fact values", code=EXIT_REJECTED)
    if not (provenance.strip() or source.strip() or link.strip()):
        raise CommandError(
            "a bundle names the one source its facts come from: --provenance, --source or --link",
            code=EXIT_REJECTED,
        )
    item = _new_item(
        domain,
        fact_type=fact_type,
        provenance=provenance,
        source=source,
        link=link,
        valid_at=valid_at,
        learned_at=learned_at,
        supersedes_rejection=supersedes_rejection,
        texts=facts,
    )
    settings = _workspace_settings()
    for fact in facts:
        _check_fact(
            domain, fact, supersedes_rejection=supersedes_rejection.strip(), settings=settings
        )
    item.update(
        operation="assert",
        text=" | ".join(facts),
        facts=facts,
        supersedes="",
        status="pending",
    )
    with _lock():
        _append("pending.jsonl", item)
    return item


def _human_set_status(ids: set[str], status: str, *, reason: str = "") -> dict[str, Any]:
    """Approve or reject. An approval is bound to the digest of the content it saw."""
    with _lock():
        items = _read()
        changed, rejected, remaining, decided = 0, [], [], set()
        now = datetime.now(timezone.utc).isoformat()
        for item in items:
            if item.get("id") not in ids:
                remaining.append(item)
                continue
            decidable = item.get("status") == "pending" or (
                status == "approved"
                and item.get("status") == "approved"
                and "approved_digest" not in item
            )
            if not decidable:
                remaining.append(item)
                continue
            item["status"] = status
            item["decided_at"] = now
            decided.add(item["id"])
            changed += 1
            if status == "approved":
                item["approved_digest"] = content_digest(item)
                remaining.append(item)
            else:
                if reason.strip():
                    item["reason"] = reason.strip()
                rejected.append(item)
        # Rejections leave the queue for the archive, where the rejection gate reads them.
        for item in rejected:
            _append("archive.jsonl", item)
        _write("pending.jsonl", remaining)
    return {"status": status, "changed": changed, "unknown": sorted(ids - decided)}


def _episode_body(item: dict, text: str | None = None) -> str:
    body = text if text is not None else item["text"]
    if item["operation"] == "assert":
        return body
    as_of = (item.get("valid_at") or item["created_at"])[:10]
    return f"Correction as of {as_of}: {body} This supersedes: {item['supersedes']}"


def _provenance(item: dict) -> str:
    parts = [
        "Graphiti Local workspace",
        item["fact_type"],
        item.get("provenance") or "unspecified",
    ]
    if item.get("source"):
        parts.append(f"source: {item['source']}")
    if item.get("link"):
        parts.append(f"link: {item['link']}")
    if item.get("learned_at"):
        parts.append(f"learned_at: {item['learned_at']}")
    return "; ".join(parts)


def _texts(item: dict) -> list[str]:
    return list(item.get("facts") or [item["text"]])


def _binding_error(item: dict) -> str | None:
    recorded = item.get("approved_digest")
    if not recorded:
        return "approved without a content digest; approve it again to bind the approval"
    if recorded != content_digest(item):
        return "content changed after approval; approve it again after review"
    return None


def _update(item_id: str, **fields: Any) -> None:
    with _lock():
        items = _read()
        for queued in items:
            if queued.get("id") == item_id:
                queued.update(fields)
        _write("pending.jsonl", items)


def _archive(item_id: str, **fields: Any) -> None:
    with _lock():
        remaining = []
        for queued in _read():
            if queued.get("id") == item_id:
                queued.update(fields)
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
                {
                    "id": item["id"],
                    "domain": item["domain"],
                    "body": _episode_body(item, text),
                    "valid_at": item.get("valid_at") or item["created_at"],
                    "binding": _binding_error(item) or "ok",
                }
                for item in approved
                for index, text in enumerate(_texts(item))
                if index not in set(item.get("landed") or [])
            ],
        }

    from kg_mcp.ingest import ingest_records

    completed, failures = 0, []
    for item in approved:
        error = _binding_error(item)
        if error:
            failures.append({"id": item["id"], "domain": item["domain"], "error": error})
            continue
        texts = _texts(item)
        landed = list(item.get("landed") or [])
        episodes = list(item.get("episodes") or [])
        for index, text in enumerate(texts):
            if index in landed:
                continue
            suffix = f" #{index + 1}" if len(texts) > 1 else ""
            record = {
                "name": f"workspace {item['id']}{suffix}",
                "body": _episode_body(item, text),
                "domain": item["domain"],
                # When the fact happened, not when it was filed.
                "valid_at": item.get("valid_at") or item["created_at"],
                "provenance": _provenance(item),
            }
            result = await ingest_records([record], apply=True, resume=False)
            # Archive only what actually landed. A failure, or an extraction that produced
            # no fact, leaves the proposal approved so the next drain retries it.
            if result["ingested"] != 1:
                error = result["failed"][0]["error"] if result["failed"] else "not ingested"
            elif not result["receipts"] or result["receipts"][0]["edges"] == 0:
                error = "extraction produced no facts from this proposal"
            else:
                error = None
            if error:
                failure = {"id": item["id"], "domain": item["domain"], "error": error}
                if suffix:
                    failure["fact"] = index + 1
                failures.append(failure)
                continue
            landed.append(index)
            episodes.append(result["receipts"][0]["episode_uuid"])
            if len(landed) < len(texts):
                _update(item["id"], landed=landed, episodes=episodes)
        if len(landed) == len(texts):
            _archive(item["id"], landed=landed, episodes=episodes)
            completed += 1
    return {"applied": True, "ingested": completed, "failed": failures, "planned": []}


def main() -> None:
    from kg_mcp.output import emit, fail

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--human", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("domain", nargs="?")
    rejected = subparsers.add_parser("rejected", help="rejected proposals and their reasons")
    rejected.add_argument("domain", nargs="?")
    for command in ("approve", "reject"):
        decision = subparsers.add_parser(command)
        decision.add_argument("ids", nargs="+")
        if command == "reject":
            decision.add_argument("--reason", default="", help="shown when a similar fact returns")
    drain_parser = subparsers.add_parser("drain")
    drain_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    as_json = not args.human

    if args.command in ("list", "rejected"):
        groups = [args.domain] if args.domain else None
        items = pending_for(groups) if args.command == "list" else rejected_for(groups)
        payload = {"pending" if args.command == "list" else "rejected": items}

        def human() -> list[str]:
            if args.command == "rejected":
                lines = [
                    f"{item['id']} [{item['domain']}] {item['text']}"
                    + (f" (reason: {item['reason']})" if item.get("reason") else "")
                    for item in items
                ]
                return [*lines, f"({len(items)} rejected)"]
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
        result = _human_set_status(
            set(args.ids), status, reason=getattr(args, "reason", "") or ""
        )

        def decision_human() -> list[str]:
            lines = [f"{result['changed']} proposal(s) {status}"]
            lines.extend(f"unknown or already decided: {item}" for item in result["unknown"])
            return lines

        emit(result, human=decision_human, as_json=as_json)
        if result["unknown"]:
            raise SystemExit(EXIT_REJECTED)
        return

    try:
        result = asyncio.run(drain(apply=args.apply))
    except ValueError as exc:  # a refused write, e.g. the embedder changed since the last one
        fail(str(exc), code=2, as_json=as_json)
        return
    except (FileNotFoundError, RuntimeError) as exc:
        fail(str(exc), as_json=as_json)
        return

    def drain_human() -> list[str]:
        if not result["applied"]:
            if not result["planned"]:
                return ["nothing approved"]
            lines = [
                f"would ingest {item['id']} [{item['domain']}] {item['body'][:80]}"
                + ("" if item["binding"] == "ok" else f" (refused: {item['binding']})")
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
