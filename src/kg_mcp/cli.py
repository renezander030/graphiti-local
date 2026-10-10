"""Read-first command-line interface for Graphiti Local.

Output is JSON on stdout by default so an agent or cron job can consume a result
directly; ``-H``/``--human`` prints the reader-friendly form instead. Any refusal
exits non-zero, so a caller never has to parse text to learn that a call failed.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Any

from kg_mcp import __version__, queries
from kg_mcp.config import allowed_groups, load_config
from kg_mcp.output import CommandError, emit, refusals
from kg_mcp.runtime import bounded


def _pending_items(groups: list[str] | None) -> list[dict[str, Any]]:
    from kg_mcp.workspace import pending_for

    return [
        {"id": item["id"], "domain": item["domain"], "text": item["text"]}
        for item in pending_for(groups)
    ]


def _pending_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return []
    lines = ["", "[PROPOSED: unreviewed, not graph truth]"]
    lines.extend(f" ? [{item['domain']}] {item['text'][:112]}" for item in items[:10])
    return lines


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else (str(value) if value else None)


async def _pending_command(args: argparse.Namespace) -> dict[str, Any]:
    groups = [args.group] if args.group else None
    if args.group:
        allowed_groups(args.group, load_config())
    return {"pending": _pending_items(groups)}


def _propose_command(args: argparse.Namespace) -> dict[str, Any]:
    from kg_mcp.workspace import add_proposal

    common = {
        "fact_type": args.type,
        "provenance": args.provenance,
        "source": args.source,
        "link": args.link,
        "valid_at": args.valid_at,
        "learned_at": args.learned_at,
        "supersedes_rejection": args.supersedes_rejection,
    }
    if args.command == "bundle":
        from kg_mcp.workspace import add_bundle

        item = add_bundle(args.group, args.fact, **common)
    else:
        item = add_proposal(
            args.group,
            args.fact,
            operation=args.operation,
            supersedes=args.supersedes,
            **common,
        )
    payload = {
        "proposed": item["id"],
        "domain": item["domain"],
        "fact_type": item["fact_type"],
        "operation": item["operation"],
        "status": item["status"],
    }
    for key in ("facts", "valid_at", "learned_at", "source", "link"):
        if key in item:
            payload[key] = item[key]
    return payload


def _doctor_command(args: argparse.Namespace) -> dict[str, Any]:
    from kg_mcp.doctor import run_checks, worst_status

    results = run_checks(offline=args.offline)
    return {"status": worst_status(results), "checks": results}


async def _graph_command(args: argparse.Namespace) -> dict[str, Any]:
    from kg_mcp.config import per_group_ladybug, settings_for_group, single_group

    settings = load_config()
    if not per_group_ladybug(settings):
        return await _graph_command_on(args, settings)
    if args.command in ("status", "edge"):
        return await _each_group_file(args, settings)
    # One file per group: a read opens only the file of the one group it names.
    requested = list(getattr(args, "groups", None) or [])
    allowed_groups(requested or None, settings)
    group = single_group(requested or settings.graph.groups, settings)
    args.groups = [group]
    return await _graph_command_on(args, settings_for_group(settings, group))


async def _each_group_file(args: argparse.Namespace, settings: Any) -> dict[str, Any]:
    """``status`` and ``edge`` visit every configured group's file that exists."""
    from kg_mcp.config import existing_group_files, settings_for_group

    existing = existing_group_files(settings)
    if args.command == "status":
        counts = {group: 0 for group in settings.graph.groups}
        for group in existing:
            payload = await _graph_command_on(args, settings_for_group(settings, group))
            counts[group] = payload["groups"][0]["nodes"]
        return {
            "provider": "ladybug",
            "groups": [{"group": group, "nodes": count} for group, count in counts.items()],
        }
    for group in existing:
        try:
            return await _graph_command_on(args, settings_for_group(settings, group))
        except CommandError:
            continue
    raise CommandError(f"edge not found: {args.uuid}")


async def _graph_command_on(args: argparse.Namespace, settings: Any) -> dict[str, Any]:
    from kg_mcp.runtime import build_graphiti

    timeout = settings.graph.query_timeout_seconds
    # Refuse a group outside the allow-list before any model client is built.
    groups = allowed_groups(getattr(args, "groups", None), settings)
    graph = build_graphiti(settings, read_only=True)
    try:
        if args.command == "ask":
            results, retrieval = await bounded(
                queries.search_facts(
                    graph,
                    settings,
                    args.query,
                    groups,
                    args.limit,
                    keyword=getattr(args, "keyword", False),
                    include_invalidated=getattr(args, "history", False),
                ),
                timeout,
                "ask",
            )
            return {
                "query": args.query,
                "groups": groups or settings.graph.groups,
                "facts": [
                    {
                        "fact": result.fact,
                        "uuid": getattr(result, "uuid", None),
                        "group_id": getattr(result, "group_id", None),
                        "source_node_uuid": getattr(result, "source_node_uuid", None),
                        "target_node_uuid": getattr(result, "target_node_uuid", None),
                        "episodes": list(getattr(result, "episodes", None) or []),
                        "valid_at": _iso(getattr(result, "valid_at", None)),
                        "invalid_at": _iso(getattr(result, "invalid_at", None)),
                        "expired_at": _iso(getattr(result, "expired_at", None)),
                        "score": score,
                    }
                    for result, score in results
                ],
                "suppressed_invalidated": retrieval["suppressed"],
                "retrieval": retrieval,
                "ranker": settings.reranker.provider,
                "pending": _pending_items(groups or settings.graph.groups),
            }
        if args.command == "nodes":
            nodes, retrieval = await bounded(
                queries.search_nodes(
                    graph, settings, args.query, groups, args.limit, keyword=args.keyword
                ),
                timeout,
                "nodes",
            )
            return {
                "query": args.query,
                "nodes": [
                    {"name": node.name, "group_id": node.group_id, "uuid": node.uuid}
                    for node in nodes
                ],
                "retrieval": retrieval,
            }
        if args.command == "episodes":
            episodes = await bounded(
                graph.retrieve_episodes(
                    reference_time=datetime.now(timezone.utc),
                    last_n=args.limit,
                    group_ids=groups,
                ),
                timeout,
                "episodes",
            )
            return {
                "episodes": [
                    {
                        "uuid": episode.uuid,
                        "name": episode.name,
                        "group_id": episode.group_id,
                        "valid_at": _iso(episode.valid_at),
                        "created_at": _iso(episode.created_at),
                    }
                    for episode in episodes
                ]
            }
        if args.command == "edge":
            edge, found_on = await queries.find_edge(
                queries.graph_drivers(graph, settings), args.uuid, timeout, "edge"
            )
            if edge is None:
                raise CommandError(f"edge not found: {args.uuid}")
            return {
                "uuid": args.uuid,
                "fact": edge.fact,
                "valid_at": _iso(edge.valid_at),
                "invalid_at": _iso(edge.invalid_at),
                "group_id": edge.group_id,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "episodes": await _episode_sources(found_on, edge.episodes or [], timeout),
            }
        if args.command == "status":
            # One driver per group on FalkorDB; elsewhere one count under the first group.
            drivers = queries.graph_drivers(graph, settings)
            return {
                "provider": settings.database.provider,
                "groups": [
                    {"group": group, "nodes": await queries.count_nodes(driver, timeout, "status")}
                    for group, driver in zip(settings.graph.groups, drivers, strict=False)
                ],
            }
        if args.command == "export":
            from kg_mcp.export import export_graph

            return await bounded(
                export_graph(graph, settings, groups or settings.graph.groups, output=args.output),
                timeout,
                "export",
            )
        if args.command == "duplicates":
            from kg_mcp.duplicates import find_duplicates

            return await bounded(
                find_duplicates(graph, settings, groups or settings.graph.groups, limit=args.limit),
                timeout,
                "duplicates",
            )
        raise CommandError(f"unknown command: {args.command}")
    finally:
        await graph.close()


async def _episode_sources(driver: Any, uuids: list[str], timeout: float) -> list[dict]:
    """The episodes that produced a fact, with the provenance each was ingested under."""
    if not uuids:
        return []
    from graphiti_core.nodes import EpisodicNode

    try:
        episodes = await bounded(EpisodicNode.get_by_uuids(driver, uuids), timeout, "edge")
    except TimeoutError:
        raise
    except Exception:
        episodes = []
    by_uuid = {episode.uuid: episode for episode in episodes}
    return [
        {
            "uuid": uuid,
            "name": getattr(by_uuid.get(uuid), "name", None),
            "source_description": getattr(by_uuid.get(uuid), "source_description", None),
            "valid_at": _iso(getattr(by_uuid.get(uuid), "valid_at", None)),
        }
        for uuid in uuids
    ]


def _render(command: str, payload: dict[str, Any]) -> list[str]:
    if command == "ask":
        lines = [f" • {item['fact']}" for item in payload["facts"]] or [" (no facts)"]
        return lines + _pending_lines(payload["pending"])
    if command == "nodes":
        return [f" • {n['name']} [{n['group_id']}] {n['uuid']}" for n in payload["nodes"]]
    if command == "episodes":
        return [
            f" • [{(e['valid_at'] or e['created_at'] or '')[:10]}] {e['name']} {e['uuid']}"
            for e in payload["episodes"]
        ]
    if command == "edge":
        return [
            f"fact:       {payload['fact']}",
            f"valid_at:   {payload['valid_at']}",
            f"invalid_at: {payload['invalid_at']}",
            f"group:      {payload['group_id']}",
            f"from:       {payload['source_node_uuid']} -> {payload['target_node_uuid']}",
            *(
                f"episode:    {episode['uuid']} {episode['name'] or ''} "
                f"({episode['source_description'] or 'no provenance'})"
                for episode in payload["episodes"]
            ),
        ]
    if command == "status":
        head = f"provider: {payload['provider']}; groups: " + ", ".join(
            item["group"] for item in payload["groups"]
        )
        return [head] + [f"  {item['group']}: {item['nodes']} nodes" for item in payload["groups"]]
    if command == "pending":
        items = payload["pending"]
        return [f"{i['id']} [{i['domain']}] {i['text']}" for i in items] + [
            f"({len(items)} pending)"
        ]
    if command in ("propose", "bundle"):
        head = f"proposed {payload['proposed']} [{payload['domain']} {payload['fact_type']}]"
        return [head, *(f"  - {fact}" for fact in payload.get("facts", []))]
    if command in ("doctor", "verify"):
        mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}
        lines = [
            f"{mark[check['status']]} {check['check']:<20} {check['detail']}"
            for check in payload["checks"]
        ]
        return [*lines, f"overall: {payload['status']}"]
    if command == "export":
        return [
            f"exported {payload['nodes']} node(s) and {payload['edges']} edge(s) "
            f"from {', '.join(payload['groups'])} to {payload['output']}"
        ]
    if command == "duplicates":
        lines = []
        for item in payload["clusters"]:
            lines.append(f"{item['count']}x {item['name']}")
            lines.extend(
                f"   {member['uuid']}  {member['name']} [{member['group_id']}]"
                for member in item["members"]
            )
        lines.append(
            f"({payload['duplicates']} entities in {len(payload['clusters'])} cluster(s) "
            f"out of {payload['entities']})"
        )
        return lines
    return [str(payload)]


async def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "pending":
        return await _pending_command(args)
    if args.command in ("propose", "bundle"):
        return _propose_command(args)
    if args.command == "doctor":
        return _doctor_command(args)
    if args.command == "verify":
        from kg_mcp.verify import run_verification

        return await run_verification(offline=args.offline, query=args.query)
    return await _graph_command(args)


def _core_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("graphiti-core")
    except metadata.PackageNotFoundError:
        return "unknown"


def parser() -> argparse.ArgumentParser:
    from kg_mcp.workspace import FACT_TYPES, OPERATIONS

    # The reader flag is accepted before and after the subcommand. The subcommand copy
    # suppresses its default so it cannot overwrite a value given up front.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-H",
        "--human",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print the reader-friendly form instead of JSON",
    )
    root = argparse.ArgumentParser(prog="kg", description=__doc__)
    root.add_argument(
        "-H",
        "--human",
        action="store_true",
        default=False,
        help="print the reader-friendly form instead of JSON",
    )
    root.add_argument(
        "--version",
        action="version",
        version=f"kg {__version__} (graphiti-core {_core_version()})",
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("ask", "nodes"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("query")
        command.add_argument("groups", nargs="*")
        command.add_argument("--limit", type=int, default=10)
        command.add_argument(
            "--keyword",
            action="store_true",
            help=(
                "Match on keywords only (BM25), skipping the vector half of the search. "
                "No embedding model is called, so this answers on a machine with no "
                "inference backend running, at the cost of missing paraphrases."
            ),
        )
        if name == "ask":
            command.add_argument(
                "--history",
                action="store_true",
                help="include facts whose invalidation timestamp has passed",
            )
    episodes = commands.add_parser("episodes", parents=[common])
    episodes.add_argument("groups", nargs="*")
    episodes.add_argument("--limit", type=int, default=10)
    edge = commands.add_parser("edge", parents=[common])
    edge.add_argument("uuid")
    commands.add_parser("status", parents=[common])
    pending = commands.add_parser("pending", parents=[common])
    pending.add_argument("group", nargs="?")
    propose = commands.add_parser("propose", parents=[common])
    bundle = commands.add_parser(
        "bundle",
        parents=[common],
        help="several facts from one source, reviewed and approved as one item",
    )
    for command in (propose, bundle):
        command.add_argument("group")
        if command is propose:
            command.add_argument("fact")
        else:
            command.add_argument("--fact", action="append", required=True, default=[])
        command.add_argument(
            "--type",
            choices=FACT_TYPES,
            default="belief",
        )
        command.add_argument("--provenance", "--prov", default="")
        command.add_argument("--source", default="", help="the system the fact comes from")
        command.add_argument("--link", default="", help="a link to the record in that system")
        command.add_argument(
            "--valid-at", default="", help="ISO date or datetime the fact became true"
        )
        command.add_argument(
            "--learned-at", default="", help="ISO date or datetime the fact was observed"
        )
        command.add_argument(
            "--supersedes-rejection",
            default="",
            help="id of a rejected proposal this one knowingly resembles",
        )
    propose.add_argument(
        "--operation",
        "--op",
        choices=OPERATIONS,
        default="assert",
    )
    propose.add_argument("--supersedes", default="")
    doctor = commands.add_parser("doctor", parents=[common])
    doctor.add_argument(
        "--offline",
        action="store_true",
        help="skip checks that need the network",
    )
    export = commands.add_parser("export", parents=[common])
    export.add_argument("groups", nargs="*")
    export.add_argument(
        "--output",
        default=None,
        help="destination JSONL path (default: a timestamped file in the workspace)",
    )
    duplicates = commands.add_parser(
        "duplicates",
        parents=[common],
        help="entities that share a name once casing and punctuation are ignored",
    )
    duplicates.add_argument("groups", nargs="*")
    duplicates.add_argument("--limit", type=int, default=50, help="clusters to report")
    verify = commands.add_parser("verify", parents=[common])
    verify.add_argument("--offline", action="store_true")
    verify.add_argument("--query", default="graphiti local verification probe")
    return root


def main() -> None:
    args = parser().parse_args()
    as_json = not getattr(args, "human", False)
    with refusals(as_json=as_json):
        payload = asyncio.run(_dispatch(args))
    emit(payload, human=lambda: _render(args.command, payload), as_json=as_json)
    if payload.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
