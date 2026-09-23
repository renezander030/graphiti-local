"""Read-only MCP server for Graphiti Local."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kg_mcp.config import (
    Settings,
    TokenGrant,
    allowed_groups,
    load_config,
    per_group_ladybug,
    settings_for_group,
    single_group,
)
from kg_mcp.runtime import bounded

logger = logging.getLogger(__name__)
_request_group_scope: ContextVar[frozenset[str] | None] = ContextVar(
    "request_group_scope", default=None
)


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


def _edge(edge: Any, score: float | None = None) -> dict[str, Any]:
    payload = {
        "uuid": edge.uuid,
        "name": edge.name,
        "fact": edge.fact,
        "source_node_uuid": edge.source_node_uuid,
        "target_node_uuid": edge.target_node_uuid,
        "group_id": edge.group_id,
        "created_at": _iso(edge.created_at),
        "valid_at": _iso(edge.valid_at),
        "invalid_at": _iso(edge.invalid_at),
        "expired_at": _iso(getattr(edge, "expired_at", None)),
    }
    if score is not None:
        payload["score"] = score
    return payload


def authorized_groups(requested: str | list[str] | None, settings: Settings) -> list[str] | None:
    """Apply both the configured graph allow-list and the authenticated token scope."""
    groups = granted_groups(requested, settings)
    if settings.database.provider == "ladybug":
        return None
    return groups


def granted_groups(requested: str | list[str] | None, settings: Settings) -> list[str]:
    """The groups a request may read: the allow-list narrowed by the token scope."""
    token_scope = _request_group_scope.get()
    requested_groups = [requested] if isinstance(requested, str) else requested
    if not requested_groups:
        requested_groups = (
            [group for group in settings.graph.groups if group in token_scope]
            if token_scope is not None
            else settings.graph.groups
        )
    allowed_groups(requested_groups, settings)
    if token_scope is not None:
        forbidden = sorted(set(requested_groups) - token_scope)
        if forbidden:
            raise ValueError(f"group is not granted by the bearer token: {forbidden[0]}")
    return list(requested_groups)


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


def transport_security(settings: Settings | None):
    """Host allow-list for the network transport.

    Without ``server.allowed_hosts`` the SDK default applies: on a loopback host only
    loopback Host headers pass, elsewhere the check is off. Behind a reverse proxy the
    Host header carries the proxied name, so it has to be listed to avoid a 421.
    """
    if settings is None or not settings.server.allowed_hosts:
        return None
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = list(settings.server.allowed_hosts)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[f"{scheme}://{host}" for host in hosts for scheme in ("http", "https")],
    )


def create_server(settings: Settings | None = None) -> FastMCP:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        from kg_mcp.runtime import build_graphiti

        active = settings or load_config()
        # With one Ladybug file per group, a group's file is opened on its first read.
        graphs: dict[str, Any] = {}
        if not per_group_ladybug(active):
            graphs[""] = build_graphiti(active, read_only=True)
        state.update(settings=active, graphs=graphs)
        try:
            yield state
        finally:
            state.clear()
            for graph in graphs.values():
                await graph.close()

    selected = settings
    mcp = FastMCP(
        "Graphiti Local",
        instructions="Read-only access to an allow-listed temporal knowledge graph.",
        host=selected.server.host if selected else "127.0.0.1",
        port=selected.server.port if selected else 8000,
        lifespan=lifespan,
        transport_security=transport_security(selected),
    )

    def configured() -> Settings:
        if "graphs" not in state:
            raise RuntimeError("Graphiti Local runtime is not initialized")
        return state["settings"]

    def runtime(requested: str | list[str] | None = None) -> tuple[Any, Settings, float]:
        """The graph a request reads, and the settings it reads under.

        With one Ladybug file per group the request must resolve to exactly one granted
        group, and only that group's file is opened: a token scoped to one group cannot
        reach another group's file. The returned settings are bound to that group.
        """
        active = configured()
        key = ""
        if per_group_ladybug(active):
            key = single_group(granted_groups(requested, active), active)
            active = settings_for_group(active, key)
        graphs = state["graphs"]
        if key not in graphs:
            from kg_mcp.runtime import build_graphiti

            path = Path(active.database.ladybug.path)
            if not path.exists():
                raise LookupError(
                    f"no graph for group '{key}' yet; the first ingest or drain for it "
                    "creates its file"
                )
            graphs[key] = build_graphiti(active, read_only=True)
        graph = graphs[key]
        # An embedded read-only handle sees the file as it was at open time; a drain that
        # landed since must become visible without restarting the server.
        reopen = getattr(graph.driver, "reopen_if_changed", None)
        if reopen is not None:
            reopen()
        return graph, active, active.graph.query_timeout_seconds

    def each_granted(requested: str | list[str] | None = None):
        """Every granted group's graph that exists, for calls without a group argument."""
        active = configured()
        if not per_group_ladybug(active):
            yield runtime(requested)
            return
        for group in granted_groups(requested, active):
            try:
                yield runtime(group)
            except LookupError:
                continue

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
            graph, active, timeout = runtime(group_ids)
            groups = authorized_groups(group_ids, active)
            from kg_mcp.retrieval import compatible_query

            safe_query = compatible_query(query, active)
            from graphiti_core.search.search_config_recipes import (
                NODE_HYBRID_SEARCH_NODE_DISTANCE,
                NODE_HYBRID_SEARCH_RRF,
            )
            from graphiti_core.search.search_filters import SearchFilters

            recipe = (
                NODE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else NODE_HYBRID_SEARCH_RRF
            )

            async def find_nodes() -> list[Any]:
                labels = entity_types or []
                if active.database.provider != "falkordb" or len(labels) < 2:
                    result = await graph.search_(
                        query=safe_query,
                        config=recipe,
                        group_ids=groups,
                        center_node_uuid=center_node_uuid,
                        search_filter=SearchFilters(node_labels=labels or None),
                    )
                    return list(result.nodes or [])
                # RediSearch cannot parse the combined ``n:A|B`` label expression that
                # graphiti-core emits. Query labels independently, then merge by UUID.
                results = await asyncio.gather(
                    *(
                        graph.search_(
                            query=safe_query,
                            config=recipe,
                            group_ids=groups,
                            center_node_uuid=center_node_uuid,
                            search_filter=SearchFilters(node_labels=[label]),
                        )
                        for label in labels
                    )
                )
                seen: set[str] = set()
                merged = []
                for result in results:
                    for node in result.nodes or []:
                        if node.uuid not in seen:
                            seen.add(node.uuid)
                            merged.append(node)
                return merged

            found = await bounded(find_nodes(), timeout, "search_nodes")
            nodes = [_node(node) for node in found[:max_nodes]]
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
        include_invalidated: bool = False,
    ) -> dict[str, Any]:
        """Search temporal facts with optional type, time, group, and center filters."""
        if max_facts < 1:
            return {"error": "max_facts must be a positive integer"}
        try:
            graph, active, timeout = runtime(group_ids)
            groups = authorized_groups(group_ids, active)
            from kg_mcp.retrieval import (
                candidate_limit,
                compatible_query,
                current_edges,
                rerank_edges,
            )

            safe_query = compatible_query(query, active)
            candidates = candidate_limit(max_facts, active)
            from graphiti_core.search.search_filters import SearchFilters

            search_filter = SearchFilters(
                edge_types=edge_types or None,
                valid_at=_date_range(valid_at_after, valid_at_before),
                invalid_at=_date_range(invalid_at_after, invalid_at_before),
            )

            async def find_facts() -> tuple[list[tuple[Any, float]], int]:
                edges = await graph.search(
                    query=safe_query,
                    group_ids=groups,
                    num_results=candidates,
                    center_node_uuid=center_node_uuid,
                    search_filter=search_filter,
                )
                current, suppressed = current_edges(
                    list(edges), include_invalidated=include_invalidated
                )
                ranked = await rerank_edges(graph, safe_query, current, active, max_facts)
                return ranked, suppressed

            ranked, suppressed = await bounded(find_facts(), timeout, "search_memory_facts")
            facts = [_edge(edge, score) for edge, score in ranked]
            return {
                "message": "Facts retrieved successfully" if facts else "No relevant facts found",
                "facts": facts,
                "suppressed_invalidated": suppressed,
                "ranker": active.reranker.provider,
            }
        except Exception as exc:
            logger.exception("fact search failed")
            return {"error": f"Error searching facts: {exc}"}

    @mcp.tool()
    async def get_entity_edge(uuid: str) -> dict[str, Any]:
        """Get one fact edge by UUID from an allow-listed graph."""
        try:
            from graphiti_core.edges import EntityEdge

            if per_group_ladybug(configured()):
                for graph, _, timeout in each_granted():
                    try:
                        edge = await bounded(
                            EntityEdge.get_by_uuid(graph.driver, uuid), timeout, "get_entity_edge"
                        )
                    except TimeoutError:
                        raise
                    except Exception:
                        continue
                    return _edge(edge)
                raise LookupError(f"edge not found: {uuid}")
            graph, active, timeout = runtime()
            groups = authorized_groups(None, active)

            if active.database.provider == "falkordb":
                for group in groups or active.graph.groups:
                    try:
                        driver = graph.driver.clone(database=group)
                        edge = await bounded(
                            EntityEdge.get_by_uuid(driver, uuid), timeout, "get_entity_edge"
                        )
                        return _edge(edge)
                    except TimeoutError:
                        raise
                    except Exception:
                        continue
                raise LookupError(f"edge not found: {uuid}")
            edge = await bounded(
                EntityEdge.get_by_uuid(graph.driver, uuid), timeout, "get_entity_edge"
            )
            scope = _request_group_scope.get()
            if scope is not None and edge.group_id and edge.group_id not in scope:
                raise LookupError(f"edge not found: {uuid}")
            return _edge(edge)
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
            graph, active, timeout = runtime(group_ids)
            groups = authorized_groups(group_ids, active)
            episodes = await bounded(
                graph.retrieve_episodes(
                    reference_time=datetime.now(timezone.utc),
                    last_n=max_episodes,
                    group_ids=groups,
                ),
                timeout,
                "get_episodes",
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
            graph, active, timeout = runtime(group_id)
            groups = authorized_groups(group_id, active)
            from graphiti_core.edges import EntityEdge
            from graphiti_core.nodes import EpisodicNode
            from graphiti_core.search.search_utils import get_mentioned_nodes

            async def trace(driver: Any) -> tuple[list[Any], list[Any]]:
                episodes = await EpisodicNode.get_by_uuids(driver, episode_uuids)
                if active.database.provider != "ladybug":
                    visible = set(groups or active.graph.groups)
                    episodes = [
                        episode for episode in episodes if episode.group_id in visible
                    ]
                edges = [
                    edge
                    for episode in episodes
                    for edge in await EntityEdge.get_by_uuids(driver, episode.entity_edges)
                ]
                nodes = await get_mentioned_nodes(driver, episodes)
                if active.database.provider != "ladybug":
                    nodes = [node for node in nodes if node.group_id in visible]
                    edges = [edge for edge in edges if edge.group_id in visible]
                return nodes, edges

            drivers = [graph.driver]
            if active.database.provider == "falkordb":
                drivers = [
                    graph.driver.clone(database=group) for group in (groups or active.graph.groups)
                ]
            traces = await bounded(
                asyncio.gather(*(trace(driver) for driver in drivers)),
                timeout,
                "get_episode_entities",
            )
            nodes = list({node.uuid: node for result in traces for node in result[0]}.values())
            edges = list({edge.uuid: edge for result in traces for edge in result[1]}.values())
            return {
                "message": f"Retrieved provenance for {len(episode_uuids)} episode(s)",
                "nodes": [_node(node) for node in nodes],
                "edges": [_edge(edge) for edge in edges],
            }
        except Exception as exc:
            logger.exception("episode provenance lookup failed")
            return {"error": f"Error getting episode entities: {exc}"}

    @mcp.tool()
    async def get_status() -> dict[str, Any]:
        """Check the server and its configured database connection."""
        try:
            if per_group_ladybug(configured()):
                granted = granted_groups(None, configured())
                opened = []
                for graph, active, timeout in each_granted():
                    await bounded(
                        graph.driver.execute_query("MATCH (n) RETURN count(n) AS count"),
                        timeout,
                        "get_status",
                    )
                    opened.extend(active.graph.groups)
                return {
                    "status": "ok",
                    "message": (
                        f"Graphiti Local is connected to ladybug; {len(opened)} of "
                        f"{len(granted)} group file(s) exist"
                    ),
                    "groups": granted,
                }
            graph, active, timeout = runtime()
            groups = authorized_groups(None, active)
            driver = graph.driver
            if active.database.provider == "falkordb":
                driver = graph.driver.clone(database=(groups or active.graph.groups)[0])
            await bounded(
                driver.execute_query("MATCH (n) RETURN count(n) AS count"),
                timeout,
                "get_status",
            )
            return {
                "status": "ok",
                "message": f"Graphiti Local is connected to {active.database.provider}",
                "groups": groups or active.graph.groups,
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

    def __init__(
        self,
        app: Any,
        grants: list[TokenGrant] | str,
        all_groups: list[str] | None = None,
    ) -> None:
        self.app = app
        if isinstance(grants, str):
            grants = [TokenGrant(token=grants)]
        all_groups = all_groups or []
        self.grants = [
            (f"Bearer {grant.token}".encode(), frozenset(grant.groups or all_groups))
            for grant in grants
        ]

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        supplied = headers.get(b"authorization", b"")
        matched: frozenset[str] | None = None
        # Compare every configured token so the matching token's position is not exposed.
        for expected, groups in self.grants:
            if hmac.compare_digest(supplied, expected):
                matched = groups
        if matched is None:
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
        marker = _request_group_scope.set(matched)
        try:
            await self.app(scope, receive, send)
        finally:
            _request_group_scope.reset(marker)


def serve(settings: Settings) -> None:
    server = create_server(settings)
    if settings.server.transport != "streamable-http":
        server.run(transport=settings.server.transport)
        return
    import uvicorn

    # Config validation already refuses this transport without a token, so the gate
    # cannot be skipped by omitting configuration.
    app = BearerTokenMiddleware(
        server.streamable_http_app(),
        settings.server.auth.grants(),
        settings.graph.groups,
    )
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
