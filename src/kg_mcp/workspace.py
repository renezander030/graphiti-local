"""Append-only proposal queue with an explicit human approval boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from kg_mcp.config import FACT_TYPES
from kg_mcp.gates import content_digest, rejected_matches, secret_shapes, sentences
from kg_mcp.output import (
    EXIT_NOT_ATOMIC,
    EXIT_REJECTED,
    EXIT_RESEMBLES_REJECTED,
    EXIT_SECRET_SHAPED,
    CommandError,
)

OPERATIONS = ("assert", "revise", "invalidate")
# Clock skew between the proposing machine and this one.
FUTURE_TOLERANCE = timedelta(minutes=5)
DRAIN_BUSY = "another drain is running"


def workspace_dir(settings: Any = None) -> Path:
    """The workspace directory: ``KG_WORKSPACE_DIR`` if set, else the configured one."""
    override = os.environ.get("KG_WORKSPACE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if settings is None:
        from kg_mcp.config import load_config

        settings = load_config()
    return Path(settings.graph.workspace_dir).expanduser().resolve()


def _path(name: str) -> Path:
    return workspace_dir() / name


def _read(name: str = "pending.jsonl") -> list[dict]:
    path = _path(name)
    if not path.exists():
        return []
    items = []
    # Iterate the file, not str.splitlines(): a fact may hold U+2028 or U+2029, which
    # JSON writes unescaped and splitlines() would treat as a line break.
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: corrupt queue JSON") from exc
            if isinstance(item, dict):
                items.append(item)
    return items


def replace_text(path: Path, text: str) -> None:
    """Write a whole file so a reader sees the old content or the new, never a part."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write(name: str, items: list[dict]) -> None:
    workspace_dir().mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    replace_text(_path(name), lines)


def _append(name: str, item: dict, *, sort_keys: bool = False) -> None:
    directory = workspace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with _path(name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=sort_keys) + "\n")


def workspace_lock() -> FileLock:
    """The lock every writer of a workspace file holds."""
    workspace_dir().mkdir(parents=True, exist_ok=True)
    return FileLock(str(_path(".lock")))


def append_locked(name: str, item: dict, *, sort_keys: bool = False) -> None:
    """Append one JSON line to a workspace file under the shared workspace lock."""
    with workspace_lock():
        _append(name, item, sort_keys=sort_keys)


def pending_for(groups: list[str] | None = None, *, status: str = "pending") -> list[dict]:
    """Queued items with ``status``: ``pending`` awaits a reviewer, ``approved`` a drain."""
    pending = [item for item in _read() if item.get("status") == status]
    if groups is None:
        return pending
    return [item for item in pending if item.get("domain") in groups]


def drained(item_id: str) -> dict:
    """An archived proposal whose facts a drain landed, with the episodes it produced."""
    for item in _read("archive.jsonl"):
        if item.get("id") == item_id and item.get("status") == "approved":
            if item.get("operation") == "retire" or not item.get("episodes"):
                break
            return item
    raise CommandError(
        f"{item_id} is not a drained proposal with landed facts; only those can be retracted",
        code=EXIT_REJECTED,
    )


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


SOURCE_REQUIRED = (
    "workspace.require_source is set: name the source system with --source and "
    "the record with --link"
)


def _config_for(domain: str) -> Any:
    """The configuration, once ``domain`` is known to be an allowed group."""
    from kg_mcp.config import allowed_groups, load_config

    config = load_config()
    try:
        allowed_groups(domain, config)
    except ValueError as exc:
        raise CommandError(str(exc), code=EXIT_REJECTED) from exc
    return config


def _refuse_secrets(*texts: str, subject: str) -> None:
    shapes = secret_shapes(*texts)
    if shapes:
        raise CommandError(
            f"proposal carries a {', '.join(shapes)} shape; state the {subject} without the value",
            code=EXIT_SECRET_SHAPED,
        )


def _new_item(
    domain: str,
    fact_type: str,
    *,
    provenance: str,
    source: str,
    link: str,
    valid_at: str,
    learned_at: str,
) -> dict[str, Any]:
    """The fields every queued item starts with; a malformed date is refused here."""
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
    return item


def file_facts(
    domain: str,
    facts: list[str],
    *,
    operation: str = "assert",
    supersedes: str = "",
    fact_type: str = "belief",
    provenance: str = "",
    source: str = "",
    link: str = "",
    valid_at: str = "",
    learned_at: str = "",
    supersedes_rejection: str = "",
    said: str = "",
) -> dict:
    """Gate and queue one reviewable item: one fact, or several facts from one source.

    ``said`` records the reviewer's own words when the fact is their statement. A new
    fact that matches a ``workspace.auto_approve`` rule is queued already approved.
    Every refusal is a ``CommandError`` carrying the exit code a caller should use.
    """
    config = _config_for(domain)
    facts = [fact.strip() for fact in facts if fact.strip()]
    provenance, source, link = provenance.strip(), source.strip(), link.strip()
    supersedes, supersedes_rejection = supersedes.strip(), supersedes_rejection.strip()
    said = said.strip()
    settings = config.workspace
    for refused, message in (
        (fact_type not in FACT_TYPES, f"fact type must be one of: {', '.join(FACT_TYPES)}"),
        (operation not in OPERATIONS, f"operation must be one of: {', '.join(OPERATIONS)}"),
        (not facts, "fact text must not be empty"),
        (operation != "assert" and not supersedes, f"{operation} requires --supersedes"),
        (
            len(facts) > 1 and not (provenance or source or link or said),
            "a bundle names the one source its facts come from: "
            "--provenance, --source, --link or --said",
        ),
        (settings.require_source and not (source and link), SOURCE_REQUIRED),
    ):
        if refused:
            raise CommandError(message, code=EXIT_REJECTED)
    _refuse_secrets(*facts, supersedes, provenance, source, link, said, subject="fact")
    item = _new_item(
        domain,
        fact_type,
        provenance=provenance,
        source=source,
        link=link,
        valid_at=valid_at,
        learned_at=learned_at,
    )
    if supersedes_rejection:
        item["supersedes_rejection"] = supersedes_rejection
    if said:
        item["said"] = said
    for fact in facts:
        _check_fact(domain, fact, supersedes_rejection=supersedes_rejection, settings=settings)
    item.update(
        operation=operation,
        text=" | ".join(facts),
        **({"facts": facts} if len(facts) > 1 else {}),
        supersedes=supersedes,
        status="pending",
    )
    rule = _approving_rule(item, settings.auto_approve)
    if rule is not None:
        _approve(item, by=f"policy:{rule.fact_type}:{rule.require}")
    append_locked("pending.jsonl", item)
    return item


def _approve(item: dict, *, by: str | None = None, at: str | None = None) -> None:
    """Approve an item, bound to the digest of the content approved; ``by`` names a rule."""
    item.update(status="approved", decided_at=at or datetime.now(timezone.utc).isoformat())
    if by:
        item["decided_by"] = by
    item["approved_digest"] = content_digest(item)


def _approving_rule(item: dict, rules: list[Any]) -> Any:
    """The first rule that approves a new fact at filing. Corrections, and a fact filed
    over a person's rejection, always wait for a person."""
    if item["operation"] != "assert" or item.get("supersedes_rejection"):
        return None
    return next((rule for rule in rules if rule.approves(item)), None)


def retire_facts(
    domain: str,
    edges: list[str],
    *,
    reason: str,
    provenance: str = "",
    source: str = "",
    link: str = "",
    valid_at: str = "",
    learned_at: str = "",
    retracts: str = "",
) -> dict:
    """Queue one reviewable item that retires the named fact edges, with no replacement.

    Once approved, ``drain`` marks exactly these edges invalid as of ``valid_at`` (when
    the facts stopped being true; the drain time if omitted). ``retracts`` names the
    drained proposal whose facts these are. Refusals are ``CommandError``.
    """
    config = _config_for(domain)
    edges = list(dict.fromkeys(edge.strip() for edge in edges if edge.strip()))
    reason, provenance = reason.strip(), provenance.strip()
    source, link = source.strip(), link.strip()
    for refused, message in (
        (not edges, "retire names at least one fact edge: --edge <uuid>"),
        (not reason, "retire needs --reason: why these facts are no longer true"),
        (config.workspace.require_source and not (source and link), SOURCE_REQUIRED),
    ):
        if refused:
            raise CommandError(message, code=EXIT_REJECTED)
    _refuse_secrets(reason, provenance, source, link, subject="reason")
    item = _new_item(
        domain,
        "action-record",
        provenance=provenance,
        source=source,
        link=link,
        valid_at=valid_at,
        learned_at=learned_at,
    )
    item.update(operation="retire", text=reason, edges=edges, supersedes="", status="pending")
    if retracts:
        item["retracts"] = retracts
    append_locked("pending.jsonl", item)
    return item


def add_proposal(domain: str, text: str, **fields: Any) -> dict:
    """One fact; ``fields`` are the keyword arguments of ``file_facts``."""
    return file_facts(domain, [text], **fields)


def add_bundle(domain: str, facts: list[str], **fields: Any) -> dict:
    """Several facts from one source, reviewed as one item; ``fields`` as ``file_facts``."""
    if len([fact for fact in facts if fact.strip()]) < 2:
        raise CommandError("a bundle needs at least two --fact values", code=EXIT_REJECTED)
    return file_facts(domain, facts, **fields)


def _rewrite(change: Callable[[dict], bool]) -> None:
    """Under the lock, let ``change`` edit each queued item; True moves it to the archive."""
    with workspace_lock():
        remaining = []
        for item in _read():
            if change(item):
                _append("archive.jsonl", item)
            else:
                remaining.append(item)
        _write("pending.jsonl", remaining)


def _decidable(item: dict, status: str) -> bool:
    """Whether a reviewer can now give ``item`` the ``status`` approved or rejected."""
    if item.get("status") == "pending":
        return True
    if item.get("status") != "approved":
        return False
    if status == "approved":
        # Re-approving binds an approval that predates content digests.
        return "approved_digest" not in item
    # An approved item can still be rejected, whoever approved it, until a drain has
    # claimed it or landed any of its facts.
    return not item.get("draining_at") and not item.get("landed")


def _human_set_status(ids: set[str], status: str, *, reason: str = "") -> dict[str, Any]:
    """Approve or reject. An approval is bound to the digest of the content it saw."""
    decided: set[str] = set()
    now = datetime.now(timezone.utc).isoformat()

    def decide(item: dict) -> bool:
        if item.get("id") not in ids or not _decidable(item, status):
            return False
        decided.add(item["id"])
        if status == "approved":
            _approve(item, at=now)
            return False
        item.update(status=status, decided_at=now)
        # The rejection is the reviewer's; drop any earlier approval and its binding.
        item.pop("approved_digest", None)
        item["decided_by"] = "reviewer"
        if reason.strip():
            item["reason"] = reason.strip()
        # Rejections leave the queue for the archive, where the rejection gate reads them.
        return True

    _rewrite(decide)
    return {"status": status, "changed": len(decided), "unknown": sorted(ids - decided)}


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
    if item.get("said"):
        parts.append(f"said: {item['said']}")
    return "; ".join(parts)


def _texts(item: dict) -> list[str]:
    return list(item.get("facts") or [item["text"]])


def _episodes(item: dict) -> list[tuple[int, dict]]:
    """The episode each fact of an approved proposal becomes, skipping facts that landed."""
    if item["operation"] == "retire":
        return []
    texts = _texts(item)
    landed = set(item.get("landed") or [])
    return [
        (
            index,
            {
                "name": f"workspace {item['id']}" + (f" #{index + 1}" if len(texts) > 1 else ""),
                "body": _episode_body(item, text),
                "domain": item["domain"],
                # When the fact happened, not when it was filed.
                "valid_at": item.get("valid_at") or item["created_at"],
                "provenance": _provenance(item),
            },
        )
        for index, text in enumerate(texts)
        if index not in landed
    ]


def _planned(item: dict) -> list[dict]:
    """What a drain would do with one approved item, in queue order: each fact still to
    land, or the one retirement."""
    if item["operation"] == "retire":
        changes = [(f"retire {', '.join(item['edges'])}: {item['text']}", item.get("valid_at"))]
    else:
        changes = [(record["body"], record["valid_at"]) for _, record in _episodes(item)]
    binding = _binding_error(item) or "ok"
    return [
        {"id": item["id"], "domain": item["domain"], "body": body, "valid_at": valid_at,
         "binding": binding, "decided_by": item.get("decided_by", "reviewer")}
        for body, valid_at in changes
    ]


def _binding_error(item: dict) -> str | None:
    recorded = item.get("approved_digest")
    if not recorded:
        return "approved without a content digest; approve it again to bind the approval"
    if recorded != content_digest(item):
        return "content changed after approval; approve it again after review"
    return None


def _claim(item: dict) -> bool:
    """Under the lock, mark an item as being drained if it is still approved with the
    content the drain read. A claimed item can no longer be rejected."""
    claimed = False

    def change(queued: dict) -> bool:
        nonlocal claimed
        if (
            queued.get("id") == item["id"]
            and queued.get("status") == "approved"
            and queued.get("approved_digest") == item.get("approved_digest")
            and _binding_error(queued) is None
        ):
            queued.setdefault("draining_at", datetime.now(timezone.utc).isoformat())
            claimed = True
        return False

    _rewrite(change)
    return claimed


def _update(item_id: str, *, archive: bool = False, **fields: Any) -> None:
    """Change one queued item; ``archive`` moves it from the queue to the archive."""

    def change(item: dict) -> bool:
        if item.get("id") != item_id:
            return False
        item.update(fields)
        if archive:
            item["drained_at"] = datetime.now(timezone.utc).isoformat()
        return archive

    _rewrite(change)


async def _land(record: dict) -> tuple[str | None, str | None]:
    """Ingest one episode: ``(episode_uuid, None)`` once a fact landed, else
    ``(None, reason)``. An extraction that produced no fact has not landed."""
    from kg_mcp.ingest import ingest_records

    result = await ingest_records([record], apply=True, resume=False)
    if result["ingested"] != 1:
        return None, result["failed"][0]["error"] if result["failed"] else "not ingested"
    if not result["receipts"] or result["receipts"][0]["edges"] == 0:
        return None, "extraction produced no facts from this proposal"
    return result["receipts"][0]["episode_uuid"], None


async def _drain_retire(item: dict) -> tuple[bool, list[dict]]:
    """Retire all of what the reviewer approved or nothing, leaving the item approved.
    Returns whether the item was archived, and its failures."""
    from kg_mcp.ingest import retire_edges

    outcome = await retire_edges(
        item["domain"],
        item["edges"],
        ended=datetime.fromisoformat(item["valid_at"]) if item.get("valid_at") else None,
    )
    if outcome["missing"]:
        return False, [
            {
                "id": item["id"],
                "domain": item["domain"],
                "error": "fact edge(s) not found in this group: " + ", ".join(outcome["missing"]),
            }
        ]
    _update(item["id"], archive=True, **outcome)
    return True, []


async def _drain_facts(item: dict) -> tuple[bool, list[dict]]:
    """Land each fact still to land. Archive only once every fact landed; a failure leaves
    the proposal approved, with what did land recorded, so the next drain retries the rest."""
    total = len(_texts(item))
    landed = list(item.get("landed") or [])
    episodes = list(item.get("episodes") or [])
    failures = []
    for index, record in _episodes(item):
        episode, error = await _land(record)
        if error:
            failure = {"id": item["id"], "domain": item["domain"], "error": error}
            if total > 1:
                failure["fact"] = index + 1
            failures.append(failure)
            continue
        landed.append(index)
        episodes.append(episode)
        if len(landed) < total:
            _update(item["id"], landed=landed, episodes=episodes)
    done = len(landed) == total
    if done:
        _update(item["id"], archive=True, landed=landed, episodes=episodes)
    return done, failures


def _release(item_id: str | None = None) -> int:
    """Hand claims back to the reviewer: the item stays approved, and can be rejected again.

    With ``item_id`` only that item is released, once a drain failed it; without it,
    every claim a stopped drain left behind. An item with landed facts keeps its claim:
    rejecting it would leave those facts in the graph unreviewed.
    """
    released = 0

    def change(item: dict) -> bool:
        nonlocal released
        if (
            (item_id is None or item.get("id") == item_id)
            and item.get("status") == "approved"
            and item.get("draining_at")
            and not item.get("landed")
        ):
            item.pop("draining_at")
            released += 1
        return False

    _rewrite(change)
    return released


def _drain_lock() -> FileLock:
    workspace_dir().mkdir(parents=True, exist_ok=True)
    return FileLock(str(_path(".drain.lock")), timeout=0)


async def drain(*, apply: bool) -> dict:
    """Land every approved item. One drain runs at a time per workspace: a second one
    returns at once with ``busy`` set. Claims a stopped drain left are released first."""
    if not apply:
        approved = [item for item in _read() if item.get("status") == "approved"]
        return {
            "applied": False,
            "ingested": 0,
            "failed": [],
            "planned": [row for item in approved for row in _planned(item)],
        }
    lock = _drain_lock()
    try:
        lock.acquire()
    except Timeout:
        return {"applied": False, "ingested": 0, "failed": [], "planned": [], "busy": DRAIN_BUSY}
    try:
        _release()
        return await _drain_approved()
    finally:
        lock.release()


async def _drain_approved() -> dict:
    approved = [item for item in _read() if item.get("status") == "approved"]
    if not approved:
        return {"applied": False, "ingested": 0, "failed": [], "planned": []}

    completed, failures = 0, []
    for item in approved:
        error = _binding_error(item)
        if error:
            failures.append({"id": item["id"], "domain": item["domain"], "error": error})
            continue
        if not _claim(item):
            continue  # rejected or changed since this drain read the queue
        land = _drain_retire if item["operation"] == "retire" else _drain_facts
        done, failed = await land(item)
        failures.extend(failed)
        completed += done
        if failed and not done:
            _release(item["id"])
    return {"applied": True, "ingested": completed, "failed": failures, "planned": []}


def main() -> None:
    from kg_mcp.output import emit, refusals

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--human", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("domain", nargs="?")
    listing.add_argument(
        "--approved",
        action="store_true",
        help="list approved items still waiting for a drain, with who approved them",
    )
    rejected = subparsers.add_parser("rejected", help="rejected proposals and their reasons")
    rejected.add_argument("domain", nargs="?")
    for command in ("approve", "reject"):
        decision = subparsers.add_parser(command)
        decision.add_argument("ids", nargs="+")
        if command == "reject":
            decision.add_argument("--reason", default="", help="shown when a similar fact returns")
        else:
            decision.add_argument(
                "--drain", action="store_true", help="land what was approved right away"
            )
    drain_parser = subparsers.add_parser("drain")
    drain_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    as_json = not args.human

    if args.command in ("list", "rejected"):
        groups = [args.domain] if args.domain else None
        with refusals(as_json=as_json):
            if args.command == "rejected":
                items = rejected_for(groups)
            else:
                items = pending_for(groups, status="approved" if args.approved else "pending")
        key = "rejected" if args.command == "rejected" else "pending"
        if args.command == "list" and args.approved:
            key = "approved"
        payload = {key: items}

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
                + (f" (approved by {item.get('decided_by', 'reviewer')})" if args.approved else "")
                for item in items
            ]
            return [*lines, f"({len(items)} {key})"]

        emit(payload, human=human, as_json=as_json)
        return
    if args.command in ("approve", "reject"):
        status = "approved" if args.command == "approve" else "rejected"
        with refusals(as_json=as_json):
            result = _human_set_status(
                set(args.ids), status, reason=getattr(args, "reason", "") or ""
            )
            if getattr(args, "drain", False) and result["changed"]:
                result["drain"] = asyncio.run(drain(apply=True))

        def decision_human() -> list[str]:
            lines = [f"{result['changed']} proposal(s) {status}"]
            lines.extend(f"unknown or already decided: {item}" for item in result["unknown"])
            if "drain" in result:
                lines.extend(_drain_lines(result["drain"]))
            return lines

        emit(result, human=decision_human, as_json=as_json)
        if result["unknown"]:
            raise SystemExit(EXIT_REJECTED)
        if result.get("drain", {}).get("failed"):
            raise SystemExit(1)
        return

    with refusals(as_json=as_json):
        result = asyncio.run(drain(apply=args.apply))
    emit(result, human=lambda: _drain_lines(result), as_json=as_json)
    if result["failed"]:
        raise SystemExit(1)


def _drain_lines(result: dict) -> list[str]:
    """The reader-friendly form of a drain result."""
    if result.get("busy"):
        return [result["busy"]]
    if not result["applied"]:
        if not result["planned"]:
            return ["nothing approved"]
        lines = [
            f"would ingest {item['id']} [{item['domain']}] {item['body'][:80]}"
            + ("" if item["binding"] == "ok" else f" (refused: {item['binding']})")
            + ("" if item["decided_by"] == "reviewer" else f" ({item['decided_by']})")
            for item in result["planned"]
        ]
        return [*lines, "dry run; add --apply"]
    lines = [f"{result['ingested']} proposal(s) ingested"]
    lines.extend(
        f"FAILED {item['id']} [{item['domain']}]: {item['error']}" for item in result["failed"]
    )
    return lines
