"""Read-only MCP server for Graphiti Local."""

from __future__ import annotations

import argparse
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kg_mcp.config import Settings, allowed_groups, load_config

logger = logging.getLogger(__name__)


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _node(node: Any) -> dict[str, Any]:
    attributes = {
        key: value
        for key, value in (getattr(node, "attributes", None) or {}).items()
        if "embedding" not in key.lower()
    }
    return {
        "uuid": node.uuid,
        "name": node.name,
        "labels": node.labels or [],
        "created_at": _iso(node.created_at),
        "summary": node.summary,
        "group_id": node.group_id,
        "attributes": attributes,
    }


def _edge(edge: Any) -> dict[str, Any]:
    return {
        "uuid": edge.uuid,
        "name": edge.name,
        "fact": edge.fact,
        "source_node_uuid": edge.source_node_uuid,
        "target_node_uuid": edge.target_node_uuid,
        "group_id": edge.group_id,
        "created_at": _iso(edge.created_at),
        "valid_at": _iso(edge.valid_at),
        "invalid_at": _iso(edge.invalid_at),
    }


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower().endswith("z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _date_range(after: str | None, before: str | None):
    from graphiti_core.search.search_filters import ComparisonOperator, DateFilter

    conditions = []
    if after:
        conditions.append(
            DateFilter(
                date=_parse_time(after),
                comparison_operator=ComparisonOperator.greater_than_equal,
            )
        )
    if before:
        conditions.append(
            DateFilter(
                date=_parse_time(before),
                comparison_operator=ComparisonOperator.less_than_equal,
            )
        )
    return [conditions] if conditions else None


def create_server(settings: Settings | None = None) -> FastMCP:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        from kg_mcp.runtime import build_graphiti

        active = settings or load_config()
        graph = build_graphiti(active, read_only=True)
        state.update(settings=active, graph=graph)
        try:
            yield state
        finally:
            state.clear()
            await graph.close()

    selected = settings
    mcp = FastMCP(
        "Graphiti Local",
        instructions="Read-only access to an allow-listed temporal knowledge graph.",
        host=selected.server.host if selected else "127.0.0.1",
        port=selected.server.port if selected else 8000,
        lifespan=lifespan,
    )

    def runtime() -> tuple[Any, Settings]:
        if "graph" not in state:
            raise RuntimeError("Graphiti Local runtime is not initialized")
        return state["graph"], state["settings"]

    @mcp.tool()
    async def search_nodes(
        query: str,
        group_ids: str | list[str] | None = None,
        max_nodes: int = 10,
        entity_types: list[str] | None = None,
        center_node_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Search graph entities. Requests are restricted to configured groups."""
        if max_nodes < 1:
            return {"error": "max_nodes must be a positive integer"}
        try:
            graph, active = runtime()
            groups = allowed_groups(group_ids, active)
            from graphiti_core.search.search_config_recipes import (
                NODE_HYBRID_SEARCH_NODE_DISTANCE,
                NODE_HYBRID_SEARCH_RRF,
            )
            from graphiti_core.search.search_filters import SearchFilters

            recipe = (
                NODE_HYBRID_SEARCH_NODE_DISTANCE
                if center_node_uuid
                else NODE_HYBRID_SEARCH_RRF
            )
            result = await graph.search_(
                query=query,
                config=recipe,
                group_ids=groups,
                center_node_uuid=center_node_uuid,
                search_filter=SearchFilters(node_labels=entity_types),
            )
            nodes = [_node(node) for node in (result.nodes or [])[:max_nodes]]
            return {
                "message": "Nodes retrieved successfully" if nodes else "No relevant nodes found",
                "nodes": nodes,
            }
        except Exception as exc:
            logger.exception("node search failed")
            return {"error": f"Error searching nodes: {exc}"}

    @mcp.tool()
    async def search_memory_facts(
        query: str,
        group_ids: str | list[str] | None = None,
        max_facts: int = 10,
        center_node_uuid: str | None = None,
        edge_types: list[str] | None = None,
        valid_at_after: str | None = None,
        valid_at_before: str | None = None,
        invalid_at_after: str | None = None,
        invalid_at_before: str | None = None,
    ) -> dict[str, Any]:
        """Search temporal facts with optional type, time, group, and center filters."""
        if max_facts < 1:
            return {"error": "max_facts must be a positive integer"}
        try:
            graph, active = runtime()
            groups = allowed_groups(group_ids, active)
            from graphiti_core.search.search_filters import SearchFilters

            search_filter = SearchFilters(
                edge_types=edge_types or None,
                valid_at=_date_range(valid_at_after, valid_at_before),
                invalid_at=_date_range(invalid_at_after, invalid_at_before),
            )
            edges = await graph.search(
                query=query,
                group_ids=groups,
                num_results=max_facts,
                center_node_uuid=center_node_uuid,
                search_filter=search_filter,
            )
            facts = [_edge(edge) for edge in edges]
            return {
                "message": "Facts retrieved successfully" if facts else "No relevant facts found",
                "facts": facts,
            }
        except Exception as exc:
            logger.exception("fact search failed")
            return {"error": f"Error searching facts: {exc}"}

    @mcp.tool()
    async def get_entity_edge(uuid: str) -> dict[str, Any]:
        """Get one fact edge by UUID from an allow-listed graph."""
        try:
            graph, active = runtime()
            from graphiti_core.edges import EntityEdge

            if active.database.provider == "falkordb":
                for group in active.graph.groups:
                    try:
                        driver = graph.driver.clone(database=group)
                        return _edge(await EntityEdge.get_by_uuid(driver, uuid))
                    except Exception:
                        continue
                raise LookupError(f"edge not found: {uuid}")
            return _edge(await EntityEdge.get_by_uuid(graph.driver, uuid))
        except Exception as exc:
            logger.exception("edge lookup failed")
            return {"error": f"Error getting entity edge: {exc}"}

    @mcp.tool()
    async def get_episodes(
        group_ids: str | list[str] | None = None,
        max_episodes: int = 10,
    ) -> dict[str, Any]:
        """Return the most recent source episodes from configured groups."""
        if max_episodes < 1:
            return {"error": "max_episodes must be a positive integer"}
        try:
            graph, active = runtime()
            groups = allowed_groups(group_ids, active)
            episodes = await graph.retrieve_episodes(
                reference_time=datetime.now(timezone.utc),
                last_n=max_episodes,
                group_ids=groups,
            )
            results = [
                {
                    "uuid": episode.uuid,
                    "name": episode.name,
                    "content": episode.content,
                    "created_at": _iso(episode.created_at),
                    "valid_at": _iso(episode.valid_at),
                    "source": getattr(episode.source, "value", str(episode.source)),
                    "source_description": episode.source_description,
                    "group_id": episode.group_id,
                }
                for episode in episodes
            ]
            return {
                "message": "Episodes retrieved successfully" if results else "No episodes found",
                "episodes": results,
            }
        except Exception as exc:
            logger.exception("episode lookup failed")
            return {"error": f"Error getting episodes: {exc}"}

    @mcp.tool()
    async def get_episode_entities(
        episode_uuids: list[str],
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """Trace which entities and facts were produced by source episodes."""
        if not episode_uuids:
            return {"error": "episode_uuids must contain at least one UUID"}
        try:
            graph, active = runtime()
            groups = allowed_groups(group_id, active)
            driver = graph.driver
            if active.database.provider == "falkordb":
                driver = graph.driver.clone(database=(groups or active.graph.groups)[0])
            from graphiti_core.edges import EntityEdge
            from graphiti_core.nodes import EpisodicNode
            from graphiti_core.search.search_utils import get_mentioned_nodes

            episodes = await EpisodicNode.get_by_uuids(driver, episode_uuids)
            edges = [
                edge
                for episode in episodes
                for edge in await EntityEdge.get_by_uuids(driver, episode.entity_edges)
            ]
            nodes = await get_mentioned_nodes(driver, episodes)
            return {
                "message": f"Retrieved provenance for {len(episode_uuids)} episode(s)",
                "nodes": [_node(node) for node in nodes],
                "edges": [_edge(edge) for edge in edges],
            }
        except Exception as exc:
            logger.exception("episode provenance lookup failed")
            return {"error": f"Error getting episode entities: {exc}"}

    @mcp.tool()
    async def get_status() -> dict[str, str]:
        """Check the server and its configured database connection."""
        try:
            graph, active = runtime()
            await graph.driver.execute_query("MATCH (n) RETURN count(n) AS count")
            return {
                "status": "ok",
                "message": f"Graphiti Local is connected to {active.database.provider}",
            }
        except Exception as exc:
            logger.exception("status check failed")
            return {"status": "error", "message": f"Database connection failed: {exc}"}

    return mcp


class BearerTokenMiddleware:
    """Require a bearer token on every HTTP request before it reaches a tool.

    stdio is a private pipe, but the network transport is reachable by anything that can
    open the port, so it is gated here. The comparison is constant-time: a byte-by-byte
    early exit would leak the token one character at a time.

    The SDK also offers a first-class ``TokenVerifier`` hook, but it expects OAuth
    resource metadata (issuer and resource-server URLs) that a single shared local token
    does not have. Wrapping the ASGI app keeps the configuration to one secret.
    """

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        if not hmac.compare_digest(headers.get(b"authorization", b""), self.expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
            return
        await self.app(scope, receive, send)


def serve(settings: Settings) -> None:
    server = create_server(settings)
    if settings.server.transport != "streamable-http":
        server.run(transport=settings.server.transport)
        return
    import uvicorn

    # Config validation already refuses this transport without a token, so the gate
    # cannot be skipped by omitting configuration.
    app = BearerTokenMiddleware(server.streamable_http_app(), settings.server.auth.token)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.config:
        os.environ["GRAPHITI_LOCAL_CONFIG"] = str(args.config.resolve())
    serve(load_config())


if __name__ == "__main__":
    main()
