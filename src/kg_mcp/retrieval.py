"""Target-owned retrieval rules that sit above graphiti-core search recipes."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from kg_mcp.config import Settings


def compatible_query(query: str, settings: Settings) -> str:
    """Remove FalkorDB's reserved standalone underscore without changing identifiers."""
    cleaned = query
    if settings.database.provider == "falkordb":
        cleaned = " ".join(token for token in query.split() if token != "_")
    if not cleaned:
        raise ValueError("search query is empty after removing unsupported FalkorDB syntax")
    return cleaned


def current_edge(edge: Any, *, now: datetime | None = None) -> bool:
    """An edge is current until its invalidation timestamp has passed."""
    invalid_at = getattr(edge, "invalid_at", None)
    if invalid_at is None:
        return True
    if invalid_at.tzinfo is None:
        invalid_at = invalid_at.replace(tzinfo=timezone.utc)
    return invalid_at > (now or datetime.now(timezone.utc))


def current_edges(edges: list[Any], *, include_invalidated: bool) -> tuple[list[Any], int]:
    if include_invalidated:
        return edges, 0
    selected = [edge for edge in edges if current_edge(edge)]
    return selected, len(edges) - len(selected)


async def rerank_edges(
    graph: Any,
    query: str,
    edges: list[Any],
    settings: Settings,
    limit: int,
) -> list[tuple[Any, float]]:
    """Rerank without using fact text as identity.

    Cross encoders return ``(passage, score)`` pairs. Multiple graph edges can carry
    identical fact text, so a queue per passage preserves every edge UUID.
    """
    if not edges:
        return []
    by_fact: dict[str, deque[Any]] = defaultdict(deque)
    for edge in edges:
        by_fact[str(edge.fact)].append(edge)
    ranked = await graph.cross_encoder.rank(query, [str(edge.fact) for edge in edges])
    resolved: list[tuple[Any, float]] = []
    for passage, score in ranked:
        queue = by_fact.get(str(passage))
        if queue:
            resolved.append((queue.popleft(), float(score)))
    # A provider that omitted a score must not silently erase an otherwise valid edge.
    for queue in by_fact.values():
        resolved.extend((edge, 0.0) for edge in queue)
    return [(edge, score) for edge, score in resolved if score >= settings.reranker.min_score][
        :limit
    ]


def candidate_limit(limit: int, settings: Settings) -> int:
    if limit < 1 or limit > 100:
        raise ValueError("result limit must be between 1 and 100")
    return min(1000, limit * settings.reranker.candidate_multiplier)


def retrieval_stats(
    *,
    requested: int,
    candidate_ceiling: int,
    candidates_seen: int,
    eligible: int,
    returned: int,
    suppressed: int = 0,
) -> dict[str, Any]:
    """Explain when bounded candidate generation may have hidden further matches."""
    ceiling_reached = candidates_seen >= candidate_ceiling
    return {
        "requested": requested,
        "candidate_limit": candidate_ceiling,
        "candidates_seen": candidates_seen,
        "eligible": eligible,
        "returned": returned,
        "suppressed": suppressed,
        "recall_may_be_incomplete": ceiling_reached,
    }


def keyword_edge_search_config(limit: int = 10):
    """A graphiti ``SearchConfig`` for facts that matches on keywords alone.

    BM25 over the full-text index, ordered by reciprocal rank fusion. The vector half of
    the default hybrid search embeds the query, so an embedding backend has to answer
    before any question can; this config makes no model call at all, which keeps a graph
    readable on a machine with nothing running. Pass it as ``config=`` to
    ``Graphiti.search_``; the facts are in ``result.edges``.
    """
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        SearchConfig,
    )

    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25], reranker=EdgeReranker.rrf
        ),
        limit=limit,
    )


def keyword_node_search_config(limit: int = 10):
    """Entity equivalent of :func:`keyword_edge_search_config`; results in ``result.nodes``."""
    from graphiti_core.search.search_config import (
        NodeReranker,
        NodeSearchConfig,
        NodeSearchMethod,
        SearchConfig,
    )

    return SearchConfig(
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25], reranker=NodeReranker.rrf
        ),
        limit=limit,
    )
