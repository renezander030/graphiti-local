"""The 0.6.0 contract: durable ingestion and auditable retrieval/review."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from kg_mcp.config import Settings


def _settings(**graph_overrides):
    graph = {
        "groups": ["example"],
        "ingest_max_attempts": 3,
        "ingest_retry_base_seconds": 0,
        **graph_overrides,
    }
    return Settings.model_validate(
        {
            "graph": graph,
            "database": {"provider": "falkordb"},
            "llm": {
                "model": "test",
                "api_url": "http://localhost:11434/v1",
                "api_key": "local-test",
            },
            "embedder": {"model": "test", "dimensions": 8},
        }
    )


def test_release_metadata_and_core_pin_are_in_sync():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "server.json").read_text(encoding="utf-8"))
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert manifest["version"] == "0.6.0"
    assert manifest["packages"][0]["version"] == "0.6.0"
    assert 'version = "0.6.0"' in project
    assert 'graphiti-core[falkordb]==0.30.2' in project


def test_chunking_is_bounded_stable_and_lossless():
    from kg_mcp.ingest import chunk_text

    text = "First paragraph has several words.\n\n" + "second " * 20
    first = chunk_text(text, 45)
    second = chunk_text(text, 45)
    assert first == second
    assert all(0 < len(chunk) <= 45 for chunk in first)
    assert " ".join(" ".join(first).split()) == " ".join(text.split())


def test_document_directory_ingest_is_sorted_and_skips_hidden_files(tmp_path: Path):
    from kg_mcp.ingest import read_input

    (tmp_path / "b.md").write_text("bravo", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.txt").write_text("nested alpha", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("secret", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("print('no')", encoding="utf-8")
    records = read_input(tmp_path, domain="example")
    assert [record["name"] for record in records] == ["a.txt", "b.md", "nested/a.txt"]
    assert {record["domain"] for record in records} == {"example"}


def test_document_ingest_rejects_zero_content(tmp_path: Path):
    from kg_mcp.ingest import read_input

    empty = tmp_path / "empty.md"
    empty.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no text"):
        read_input(empty)


def test_document_domain_does_not_override_an_explicit_json_group(tmp_path: Path):
    from kg_mcp.ingest import read_input

    source = tmp_path / "records.json"
    source.write_text(
        json.dumps({"name": "record", "body": "fact", "domain": "explicit"}),
        encoding="utf-8",
    )
    assert read_input(source, domain="fallback")[0]["domain"] == "explicit"


def test_ingest_retries_with_one_stable_episode_id_and_returns_receipt(monkeypatch):
    from kg_mcp.ingest import ingest_records

    calls = []

    class Graph:
        async def build_indices_and_constraints(self):
            return None

        async def add_episode(self, **kwargs):
            calls.append(kwargs["uuid"])
            if len(calls) < 3:
                raise TimeoutError("backend timed out")
            return SimpleNamespace(
                episode=SimpleNamespace(uuid=kwargs["uuid"]),
                nodes=[object(), object()],
                edges=[object()],
            )

        async def close(self):
            return None

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: Graph())
    result = asyncio.run(
        ingest_records(
            [{"name": "one", "body": "fact", "domain": "example"}],
            apply=True,
            ledger=False,
            settings=_settings(),
            track_fingerprint=False,
        )
    )
    assert result["ingested"] == 1 and result["failed"] == []
    assert len(set(calls)) == 1 and len(calls) == 3
    assert result["receipts"][0] == {
        "key": result["receipts"][0]["key"],
        "episode_uuid": calls[0],
        "name": "one",
        "domain": "example",
        "nodes": 2,
        "edges": 1,
        "attempts": 3,
        "ingested_at": result["receipts"][0]["ingested_at"],
    }


def test_exhausted_ingest_failure_is_actionable(monkeypatch):
    from kg_mcp.ingest import ingest_records

    class Graph:
        async def build_indices_and_constraints(self):
            return None

        async def add_episode(self, **kwargs):
            raise TimeoutError(f"query timed out for {kwargs['uuid']}")

        async def close(self):
            return None

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: Graph())
    result = asyncio.run(
        ingest_records(
            [{"name": "one", "body": "fact", "domain": "example"}],
            apply=True,
            ledger=False,
            settings=_settings(),
            track_fingerprint=False,
        )
    )
    failure = result["failed"][0]
    assert failure["attempts"] == 3
    assert failure["retryable"] is True
    assert failure["episode_uuid"] and failure["key"]


def test_success_and_failure_ledgers_are_durable(tmp_path: Path, monkeypatch):
    from kg_mcp.ingest import FAILURES, RECEIPTS, ingest_records

    monkeypatch.setattr("kg_mcp.workspace.workspace_dir", lambda: tmp_path)
    outcomes = [
        SimpleNamespace(
            episode=SimpleNamespace(uuid="stored"), nodes=[object()], edges=[object()]
        ),
        TimeoutError("query timed out"),
    ]

    class Graph:
        async def build_indices_and_constraints(self):
            return None

        async def add_episode(self, **kwargs):
            del kwargs
            outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        async def close(self):
            return None

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: Graph())
    settings = _settings(ingest_max_attempts=1)
    result = asyncio.run(
        ingest_records(
            [
                {"name": "good", "body": "land", "domain": "example"},
                {"name": "bad", "body": "fail", "domain": "example"},
            ],
            apply=True,
            settings=settings,
            track_fingerprint=False,
        )
    )
    assert result["ingested"] == 1 and len(result["failed"]) == 1
    assert json.loads((tmp_path / RECEIPTS).read_text(encoding="utf-8"))["episode_uuid"]
    assert json.loads((tmp_path / FAILURES).read_text(encoding="utf-8"))["attempts"] == 1


def _snapshot(path: Path) -> Path:
    from kg_mcp.export import FORMAT_VERSION, _atomic_snapshot, _json_line

    records = [
        {"kind": "entity_node", "uuid": "n1", "group_id": "example"},
        {
            "kind": "episodic_node",
            "uuid": "ep1",
            "group_id": "example",
            "entity_edges": ["keep", "drop"],
        },
        {"kind": "entity_edge", "uuid": "keep", "fact": "keep me"},
        {"kind": "entity_edge", "uuid": "drop", "fact": "drop me"},
    ]
    digest = hashlib.sha256(b"".join(_json_line(record) for record in records)).hexdigest()
    header = {
        "kind": "export",
        "format_version": FORMAT_VERSION,
        "record_count": len(records),
        "sha256": digest,
        "provider": "ladybug",
        "groups": ["example"],
    }
    _atomic_snapshot(path, header, records)
    return path


def test_review_decisions_filter_facts_and_leave_a_digest_receipt(tmp_path: Path):
    from kg_mcp.restore import read_snapshot
    from kg_mcp.review import apply_review_decisions, create_decision_template

    source = _snapshot(tmp_path / "review.jsonl")
    template = Path(create_decision_template(source)["output"])
    lines = [json.loads(line) for line in template.read_text(encoding="utf-8").splitlines()]
    for row in lines[1:]:
        row["decision"] = "accept" if row["uuid"] == "keep" else "refuse"
        row["reason"] = "reviewed"
    template.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")

    receipt = apply_review_decisions(source, template)
    header, records = read_snapshot(Path(receipt["output"]))
    assert receipt["accepted"] == 1 and receipt["refused"] == 1
    assert {record.get("uuid") for record in records} == {"n1", "ep1", "keep"}
    episode = next(record for record in records if record["kind"] == "episodic_node")
    assert episode["entity_edges"] == ["keep"]
    assert header["review"]["decisions_sha256"] == receipt["decisions_sha256"]


def test_review_refuses_unresolved_decisions(tmp_path: Path):
    from kg_mcp.review import apply_review_decisions, create_decision_template

    source = _snapshot(tmp_path / "review.jsonl")
    template = Path(create_decision_template(source)["output"])
    with pytest.raises(ValueError, match="unresolved"):
        apply_review_decisions(source, template)


def test_retrieval_reports_a_reached_candidate_ceiling():
    from kg_mcp.retrieval import retrieval_stats

    stats = retrieval_stats(
        requested=5,
        candidate_ceiling=15,
        candidates_seen=15,
        eligible=4,
        returned=4,
        suppressed=11,
    )
    assert stats["recall_may_be_incomplete"] is True
    assert stats["suppressed"] == 11


def test_required_schema_keeps_nullable_types_and_recurses():
    from kg_mcp.runtime import require_all_schema_properties

    schema = {
        "type": "object",
        "properties": {
            "valid_at": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "nested": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
    }
    prepared = require_all_schema_properties(schema)
    assert "required" not in schema
    assert prepared["required"] == ["valid_at", "nested"]
    assert prepared["properties"]["nested"]["required"] == ["value"]
    assert {item["type"] for item in prepared["properties"]["valid_at"]["anyOf"]} == {
        "string",
        "null",
    }


def test_generic_llm_uses_the_required_nullable_schema():
    from kg_mcp.runtime import build_llm

    class Result(BaseModel):
        valid_at: str | None = None

    client = build_llm(_settings())
    schema = client._build_response_format(Result)["json_schema"]["schema"]
    assert schema["required"] == ["valid_at"]


def test_dedup_tie_break_prefers_the_lexically_first_uuid(monkeypatch):
    from graphiti_core.utils.maintenance import dedup_helpers as helpers

    from kg_mcp import dedup

    monkeypatch.setattr(helpers, "_FUZZY_JACCARD_THRESHOLD", 0.5)
    extracted = SimpleNamespace(uuid="new", name="deterministic candidate", labels=["Entity"])
    first = SimpleNamespace(uuid="a", name="deterministic candidatf", labels=["Entity"])
    second = SimpleNamespace(uuid="z", name="deterministic candidatg", labels=["Entity"])
    shared = {"shared"}
    signature = helpers._minhash_signature(shared)
    buckets = defaultdict(list)
    for index, band in enumerate(helpers._lsh_bands(signature)):
        buckets[(index, band)] = ["z", "a"]
    indexes = SimpleNamespace(
        normalized_existing={},
        nodes_by_uuid={"a": first, "z": second},
        shingles_by_candidate={"a": shared, "z": shared},
        lsh_buckets=buckets,
    )
    monkeypatch.setattr(helpers, "_cached_shingles", lambda value: shared)
    state = SimpleNamespace(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[], duplicate_pairs=[]
    )
    dedup._resolve_with_similarity([extracted], indexes, state)
    assert state.resolved_nodes == [first]
    assert state.uuid_map == {"new": "a"}


def test_deterministic_resolver_is_installed_at_the_graphiti_call_site():
    from graphiti_core.utils.maintenance import dedup_helpers, node_operations

    from kg_mcp.dedup import _resolve_with_similarity, install_deterministic_tie_break

    install_deterministic_tie_break()
    assert dedup_helpers._resolve_with_similarity is _resolve_with_similarity
    assert node_operations._resolve_with_similarity is _resolve_with_similarity
