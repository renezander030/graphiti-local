"""Deterministic entity-resolution guard for the pinned Graphiti release."""

from __future__ import annotations

from typing import Any

_INSTALLED = False


def _resolve_with_similarity(
    extracted_nodes: list[Any],
    indexes: Any,
    state: Any,
) -> None:
    """Resolve equal fuzzy matches with a stable UUID tie-break.

    graphiti-core 0.30.x collects LSH hits in a set and keeps the first candidate
    with the best score. Set iteration depends on ``PYTHONHASHSEED``. This equivalent
    resolver sorts candidate UUIDs and explicitly breaks equal scores by UUID so two
    workers ingesting the same episode choose the same canonical entity.
    """
    from graphiti_core.utils.maintenance import dedup_helpers as helpers

    for idx, node in enumerate(extracted_nodes):
        normalized_exact = helpers._normalize_string_exact(node.name)
        normalized_fuzzy = helpers._normalize_name_for_fuzzy(node.name)

        existing_matches = indexes.normalized_existing.get(normalized_exact, [])
        if len(existing_matches) == 1:
            match = helpers._promote_resolved_node(node, existing_matches[0])
            state.resolved_nodes[idx] = match
            state.uuid_map[node.uuid] = match.uuid
            if match.uuid != node.uuid:
                state.duplicate_pairs.append((node, match))
            continue
        if len(existing_matches) > 1:
            state.unresolved_indices.append(idx)
            continue

        if not helpers._has_high_entropy(normalized_fuzzy):
            state.unresolved_indices.append(idx)
            continue

        shingles = helpers._cached_shingles(normalized_fuzzy)
        signature = helpers._minhash_signature(shingles)
        candidate_ids: set[str] = set()
        for band_index, band in enumerate(helpers._lsh_bands(signature)):
            candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

        best_candidate = None
        best_score = 0.0
        for candidate_id in sorted(candidate_ids):
            candidate = indexes.nodes_by_uuid.get(candidate_id)
            if candidate is None:
                continue
            candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
            score = helpers._jaccard_similarity(shingles, candidate_shingles)
            if score > best_score or (
                score == best_score
                and best_candidate is not None
                and candidate.uuid < best_candidate.uuid
            ):
                best_score = score
                best_candidate = candidate

        if best_candidate is not None and best_score >= helpers._FUZZY_JACCARD_THRESHOLD:
            best_candidate = helpers._promote_resolved_node(node, best_candidate)
            state.resolved_nodes[idx] = best_candidate
            state.uuid_map[node.uuid] = best_candidate.uuid
            if best_candidate.uuid != node.uuid:
                state.duplicate_pairs.append((node, best_candidate))
            continue

        state.unresolved_indices.append(idx)


def install_deterministic_tie_break() -> None:
    """Install the process-wide resolver once, including the imported call-site alias."""
    global _INSTALLED
    if _INSTALLED:
        return
    from graphiti_core.utils.maintenance import dedup_helpers, node_operations

    dedup_helpers._resolve_with_similarity = _resolve_with_similarity
    node_operations._resolve_with_similarity = _resolve_with_similarity
    _INSTALLED = True
