"""Read-first command-line interface for KG-MCP."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from kg_mcp.config import allowed_groups, load_config


def _pending(groups: list[str] | None) -> None:
    from kg_mcp.workspace import pending_for

    items = pending_for(groups)
    if not items:
        return
    print("\n[PROPOSED: unreviewed, not graph truth]")
    for item in items[:10]:
        print(f" ? [{item['domain']}] {item['text'][:112]}")


async def _run(args: argparse.Namespace) -> None:
    if args.command == "pending":
        _pending([args.group] if args.group else None)
        return
    if args.command == "propose":
        from kg_mcp.workspace import add_proposal

        item = add_proposal(
            args.group,
            args.fact,
            fact_type=args.type,
            provenance=args.provenance,
            operation=args.operation,
            supersedes=args.supersedes,
        )
        print(f"proposed {item['id']} [{item['domain']} {item['fact_type']}]")
        return

    settings = load_config()
    from kg_mcp.runtime import build_graphiti

    graph = build_graphiti(settings, read_only=True)
    try:
        if args.command == "ask":
            groups = allowed_groups(args.groups, settings)
            results = await graph.search(args.query, group_ids=groups, num_results=args.limit)
            for result in results:
                print(f" • {result.fact}")
            if not results:
                print(" (no facts)")
            _pending(groups)
        elif args.command == "nodes":
            groups = allowed_groups(args.groups, settings)
            from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

            result = await graph.search_(
                args.query,
                config=NODE_HYBRID_SEARCH_RRF,
                group_ids=groups,
            )
            for node in result.nodes[: args.limit]:
                print(f" • {node.name} [{node.group_id}] {node.uuid}")
        elif args.command == "episodes":
            groups = allowed_groups(args.groups, settings)
            episodes = await graph.retrieve_episodes(
                reference_time=datetime.now(timezone.utc),
                last_n=args.limit,
                group_ids=groups,
            )
            for episode in episodes:
                when = episode.valid_at or episode.created_at
                print(f" • [{str(when)[:10]}] {episode.name} {episode.uuid}")
        elif args.command == "edge":
            from graphiti_core.edges import EntityEdge

            edge = None
            drivers = [graph.driver]
            if settings.database.provider == "falkordb":
                drivers = [graph.driver.clone(database=group) for group in settings.graph.groups]
            for driver in drivers:
                try:
                    edge = await EntityEdge.get_by_uuid(driver, args.uuid)
                    break
                except Exception:
                    continue
            if edge is None:
                raise SystemExit(f"edge not found: {args.uuid}")
            print(f"fact:       {edge.fact}")
            print(f"valid_at:   {edge.valid_at}")
            print(f"invalid_at: {edge.invalid_at}")
            print(f"group:      {edge.group_id}")
        elif args.command == "status":
            provider = settings.database.provider
            print(f"provider: {provider}; groups: {', '.join(settings.graph.groups)}")
            for group in settings.graph.groups:
                driver = (
                    graph.driver.clone(database=group)
                    if provider == "falkordb"
                    else graph.driver
                )
                rows, _, _ = await driver.execute_query("MATCH (n) RETURN count(n) AS count")
                print(f"  {group}: {rows[0]['count']} nodes")
                if provider != "falkordb":
                    break
    finally:
        await graph.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kg", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("ask", "nodes"):
        command = commands.add_parser(name)
        command.add_argument("query")
        command.add_argument("groups", nargs="*")
        command.add_argument("--limit", type=int, default=10)
    episodes = commands.add_parser("episodes")
    episodes.add_argument("groups", nargs="*")
    episodes.add_argument("--limit", type=int, default=10)
    edge = commands.add_parser("edge")
    edge.add_argument("uuid")
    commands.add_parser("status")
    pending = commands.add_parser("pending")
    pending.add_argument("group", nargs="?")
    propose = commands.add_parser("propose")
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
    return root


def main() -> None:
    asyncio.run(_run(parser().parse_args()))


if __name__ == "__main__":
    main()
