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

from kg_mcp import __version__
from kg_mcp.config import allowed_groups, load_config
from kg_mcp.output import EXIT_REJECTED, EXIT_TIMEOUT, CommandError, emit, fail
from kg_mcp.runtime import bounded

READ_COMMANDS = {"ask", "nodes", "episodes", "edge", "status", "export", "duplicates"}


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

    settings = load_config()
    # Reads have always enforced the allow-list; the write path used to skip it, so a
    # fact addressed to an unconfigured domain queued silently and exited 0.
    try:
        allowed_groups(args.group, settings)
    except ValueError as exc:
        raise CommandError(str(exc), code=EXIT_REJECTED) from exc
    item = add_proposal(
        args.group,
        args.fact,
        fact_type=args.type,
        provenance=args.provenance,
        operation=args.operation,
        supersedes=args.supersedes,
    )
    return {
        "proposed": item["id"],
        "domain": item["domain"],
        "fact_type": item["fact_type"],
        "operation": item["operation"],
        "status": item["status"],
    }


def _doctor_command(args: argparse.Namespace) -> dict[str, Any]:
    from kg_mcp.doctor import run_checks, worst_status

    results = run_checks(offline=args.offline)
    return {"status": worst_status(results), "checks": results}


async def _graph_command(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_config()
    from kg_mcp.runtime import build_graphiti

    timeout = settings.graph.query_timeout_seconds
    graph = build_graphiti(settings, read_only=True)
    try:
        if args.command == "ask":
            groups = allowed_groups(args.groups, settings)
            results = await bounded(
                graph.search(args.query, group_ids=groups, num_results=args.limit),
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
                        "valid_at": _iso(getattr(result, "valid_at", None)),
                        "invalid_at": _iso(getattr(result, "invalid_at", None)),
                    }
                    for result in results
                ],
                "pending": _pending_items(groups),
            }
        if args.command == "nodes":
            groups = allowed_groups(args.groups, settings)
            from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

            result = await bounded(
                graph.search_(args.query, config=NODE_HYBRID_SEARCH_RRF, group_ids=groups),
                timeout,
                "nodes",
            )
            return {
                "query": args.query,
                "nodes": [
                    {"name": node.name, "group_id": node.group_id, "uuid": node.uuid}
                    for node in result.nodes[: args.limit]
                ],
            }
        if args.command == "episodes":
            groups = allowed_groups(args.groups, settings)
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
            from graphiti_core.edges import EntityEdge

            edge = None
            drivers = [graph.driver]
            if settings.database.provider == "falkordb":
                drivers = [graph.driver.clone(database=group) for group in settings.graph.groups]
            for driver in drivers:
                try:
                    edge = await bounded(EntityEdge.get_by_uuid(driver, args.uuid), timeout, "edge")
                    break
                except TimeoutError:
                    raise
                except Exception:
                    continue
            if edge is None:
                raise CommandError(f"edge not found: {args.uuid}")
            return {
                "uuid": args.uuid,
                "fact": edge.fact,
                "valid_at": _iso(edge.valid_at),
                "invalid_at": _iso(edge.invalid_at),
                "group_id": edge.group_id,
            }
        if args.command == "status":
            provider = settings.database.provider
            groups = []
            for group in settings.graph.groups:
                driver = (
                    graph.driver.clone(database=group)
                    if provider == "falkordb"
                    else graph.driver
                )
                rows, _, _ = await bounded(
                    driver.execute_query("MATCH (n) RETURN count(n) AS count"),
                    timeout,
                    "status",
                )
                groups.append({"group": group, "nodes": rows[0]["count"]})
                if provider != "falkordb":
                    break
            return {"provider": provider, "groups": groups}
        if args.command == "export":
            from kg_mcp.export import export_graph

            groups = allowed_groups(args.groups, settings) or settings.graph.groups
            return await bounded(
                export_graph(graph, settings, groups, output=args.output), timeout, "export"
            )
        if args.command == "duplicates":
            from kg_mcp.duplicates import find_duplicates

            groups = allowed_groups(args.groups, settings) or settings.graph.groups
            return await bounded(
                find_duplicates(graph, settings, groups, limit=args.limit),
                timeout,
                "duplicates",
            )
        raise CommandError(f"unknown command: {args.command}")
    finally:
        await graph.close()


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
    if command == "propose":
        return [f"proposed {payload['proposed']} [{payload['domain']} {payload['fact_type']}]"]
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
    if args.command == "propose":
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
    episodes = commands.add_parser("episodes", parents=[common])
    episodes.add_argument("groups", nargs="*")
    episodes.add_argument("--limit", type=int, default=10)
    edge = commands.add_parser("edge", parents=[common])
    edge.add_argument("uuid")
    commands.add_parser("status", parents=[common])
    pending = commands.add_parser("pending", parents=[common])
    pending.add_argument("group", nargs="?")
    propose = commands.add_parser("propose", parents=[common])
    propose.add_argument("group")
    propose.add_argument("fact")
    propose.add_argument(
        "--type",
        choices=("belief", "source-fact", "action-record", "live-finding"),
        default="belief",
    )
    propose.add_argument("--provenance", "--prov", default="")
    propose.add_argument(
        "--operation",
        "--op",
        choices=("assert", "revise", "invalidate"),
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
    try:
        payload = asyncio.run(_dispatch(args))
    except CommandError as exc:
        fail(str(exc), code=exc.code, as_json=as_json)
        return
    except ValueError as exc:
        fail(str(exc), code=EXIT_REJECTED, as_json=as_json)
        return
    except TimeoutError as exc:
        fail(str(exc), code=EXIT_TIMEOUT, as_json=as_json)
        return
    except (FileNotFoundError, RuntimeError) as exc:
        fail(str(exc), as_json=as_json)
        return
    emit(payload, human=lambda: _render(args.command, payload), as_json=as_json)
    if payload.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
