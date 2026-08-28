"""Prove the installation still behaves after a config, model or backend change.

Read-only by construction: it asserts the tool surface, the preflight checks and live
retrieval, and it never writes to the graph. Run it before and after any upgrade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kg_mcp.doctor import FAIL, OK, WARN, _check, run_checks, worst_status

EXPECTED_TOOLS = {
    "search_nodes",
    "search_memory_facts",
    "get_entity_edge",
    "get_episodes",
    "get_episode_entities",
    "get_status",
}
FORBIDDEN_SUBSTRINGS = ("add", "write", "delete", "clear", "approve", "drain", "ingest", "propose")


async def check_tool_surface() -> list[dict[str, str]]:
    from kg_mcp.server import create_server

    try:
        tools = await create_server().list_tools()
    except Exception as exc:
        return [_check("tool-surface", FAIL, f"server did not build: {exc}")]
    names = {tool.name for tool in tools}
    results = []
    if names == EXPECTED_TOOLS:
        results.append(_check("tool-surface", OK, f"exactly {len(names)} read tools registered"))
    else:
        missing = sorted(EXPECTED_TOOLS - names)
        extra = sorted(names - EXPECTED_TOOLS)
        results.append(
            _check("tool-surface", FAIL, f"missing={missing or '-'} unexpected={extra or '-'}")
        )
    # Negative control: a write tool must never appear, whatever else changed.
    offenders = sorted(
        name for name in names if any(word in name.lower() for word in FORBIDDEN_SUBSTRINGS)
    )
    results.append(
        _check("no-write-tools", FAIL if offenders else OK, ", ".join(offenders) or "none exposed")
    )
    return results


async def check_retrieval(query: str) -> list[dict[str, Any]]:
    from kg_mcp.config import allowed_groups, load_config
    from kg_mcp.runtime import build_graphiti

    settings = load_config()
    graph = build_graphiti(settings, read_only=True)
    results: list[dict[str, Any]] = []
    try:
        groups = allowed_groups(None, settings)
        try:
            facts = await graph.search(query, group_ids=groups, num_results=10)
        except Exception as exc:
            return [_check("retrieval", FAIL, f"search raised: {exc}")]
        if facts:
            results.append(_check("retrieval", OK, f"search returned {len(facts)} fact(s)"))
        else:
            results.append(
                _check("retrieval", WARN, "search returned nothing; the graph may be empty")
            )

        # Invalidation is the temporal guarantee that distinguishes this from a vector
        # store, so assert a superseded fact is not served as current.
        try:
            from graphiti_core.edges import EntityEdge

            drivers = (
                [graph.driver.clone(database=group) for group in settings.graph.groups]
                if settings.database.provider == "falkordb"
                else [graph.driver]
            )
            invalidated = []
            for driver in drivers:
                edges = await EntityEdge.get_by_group_ids(driver, settings.graph.groups) or []
                now = datetime.now(timezone.utc)
                invalidated.extend(
                    edge
                    for edge in edges
                    if edge.invalid_at is not None and edge.invalid_at <= now
                )
            if not invalidated:
                results.append(
                    _check("invalidation", WARN, "no superseded facts present to check")
                )
            else:
                sample = invalidated[0]
                current = await graph.search(sample.fact, group_ids=groups, num_results=10)
                leaked = any(item.uuid == sample.uuid for item in current)
                results.append(
                    _check(
                        "invalidation",
                        FAIL if leaked else OK,
                        f"{len(invalidated)} superseded fact(s); "
                        + ("a superseded fact was served as current" if leaked else "none served"),
                    )
                )
        except Exception as exc:
            results.append(_check("invalidation", WARN, f"not checked: {exc}"))
    finally:
        await graph.close()
    return results


async def run_verification(*, offline: bool = False, query: str = "verification probe") -> dict:
    checks: list[dict[str, Any]] = list(run_checks(offline=offline))
    checks.extend(await check_tool_surface())
    if offline:
        checks.append(_check("retrieval", WARN, "skipped (--offline)"))
    elif any(item["check"] == "config" and item["status"] == FAIL for item in checks):
        checks.append(_check("retrieval", WARN, "skipped (configuration did not load)"))
    else:
        checks.extend(await check_retrieval(query))
    return {"status": worst_status(checks), "checks": checks}
