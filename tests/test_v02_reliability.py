"""Resumable ingest, fault isolation, the drain safety rule, and the HTTP gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

CONFIG = """
graph:
  groups: [allowed, other]
  workspace_dir: {workspace}
database:
  provider: falkordb
llm:
  model: test-model
embedder:
  model: test-embedder
  dimensions: 8
"""


@pytest.fixture()
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.format(workspace=workspace), encoding="utf-8")
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace))
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


RECORDS = [
    {"name": "one", "body": "first fact", "domain": "allowed"},
    {"name": "two", "body": "second fact", "domain": "allowed"},
]


# --- item 5: a re-run must not re-ingest what already landed ---


def test_record_key_is_stable_and_content_sensitive():
    from kg_mcp.ingest import record_key

    assert record_key(RECORDS[0], "allowed") == record_key(dict(RECORDS[0]), "allowed")
    assert record_key(RECORDS[0], "allowed") != record_key(RECORDS[1], "allowed")
    assert record_key(RECORDS[0], "allowed") != record_key(RECORDS[0], "other")


def test_dry_run_plans_every_record_when_the_ledger_is_empty(configured):
    from kg_mcp.ingest import ingest_records

    result = asyncio.run(ingest_records(RECORDS, apply=False))
    assert result["applied"] is False
    assert len(result["planned"]) == 2
    assert result["skipped"] == 0


def test_a_ledgered_record_is_skipped_on_the_next_run(configured):
    from kg_mcp.ingest import LEDGER, ingest_records, record_key

    (configured / LEDGER).write_text(
        json.dumps({"key": record_key(RECORDS[0], "allowed")}) + "\n", encoding="utf-8"
    )
    result = asyncio.run(ingest_records(RECORDS, apply=False))
    assert result["skipped"] == 1
    assert [item["name"] for item in result["planned"]] == ["two"]


def test_no_resume_ignores_the_ledger(configured):
    from kg_mcp.ingest import LEDGER, ingest_records, record_key

    (configured / LEDGER).write_text(
        json.dumps({"key": record_key(RECORDS[0], "allowed")}) + "\n", encoding="utf-8"
    )
    result = asyncio.run(ingest_records(RECORDS, apply=False, resume=False))
    assert result["skipped"] == 0
    assert len(result["planned"]) == 2


def test_a_corrupt_ledger_line_does_not_abort_the_run(configured):
    from kg_mcp.ingest import LEDGER, completed_keys

    (configured / LEDGER).write_text("not json\n" + json.dumps({"key": "abc"}) + "\n", "utf-8")
    assert completed_keys() == {"abc"}


def test_ingest_still_refuses_an_unconfigured_domain(configured):
    from kg_mcp.ingest import ingest_records

    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(ingest_records([{"name": "x", "body": "y", "domain": "nope"}], apply=False))


def test_skip_invalid_tolerates_malformed_lines(tmp_path: Path):
    from kg_mcp.ingest import read_records

    source = tmp_path / "in.jsonl"
    source.write_text('{"name":"a","body":"b"}\nnot json\n{"name":"c","body":"d"}\n', "utf-8")
    assert len(read_records(source, skip_invalid=True)) == 2
    with pytest.raises(SystemExit):
        read_records(source)


# --- a termination signal must stop cleanly, not mid-write ---


def test_sigterm_is_turned_into_a_stop_request():
    import os
    import signal

    from kg_mcp.ingest import _stop_on_signal

    with _stop_on_signal() as stop:
        assert stop["requested"] is False
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop["requested"] is True, "SIGTERM must be captured, not kill the process"


def test_previous_signal_handlers_are_restored():
    import signal

    from kg_mcp.ingest import _stop_on_signal

    original = signal.getsignal(signal.SIGTERM)
    with _stop_on_signal():
        assert signal.getsignal(signal.SIGTERM) is not original
    assert signal.getsignal(signal.SIGTERM) is original


def test_stop_request_halts_before_the_next_record(configured, monkeypatch):
    """The loop must break at a record boundary rather than abandon a write."""
    from kg_mcp import ingest

    calls = []

    class FakeGraph:
        driver = None

        async def build_indices_and_constraints(self):
            return None

        async def add_episode(self, **kwargs):
            calls.append(kwargs["name"])
            # Ask for shutdown while the first record is in flight.
            import os
            import signal

            os.kill(os.getpid(), signal.SIGTERM)

        async def close(self):
            calls.append("closed")

    monkeypatch.setattr(ingest, "build_graphiti", lambda *a, **k: FakeGraph(), raising=False)
    monkeypatch.setattr(
        "kg_mcp.runtime.build_graphiti", lambda *a, **k: FakeGraph(), raising=False
    )
    result = asyncio.run(ingest.ingest_records(RECORDS, apply=True, ledger=False))

    assert result["interrupted"] is True
    assert result["ingested"] == 1, "the record in flight completes, the next one does not start"
    assert calls[-1] == "closed", "the driver must be closed on the way out"


# --- the drain safety rule: never archive a proposal that did not land ---


def test_drain_keeps_a_proposal_that_failed_to_ingest(configured, monkeypatch):
    from kg_mcp import workspace

    workspace.add_proposal("allowed", "a fact")
    ids = {item["id"] for item in workspace.pending_for()}
    workspace._human_set_status(ids, "approved")

    async def failing_ingest(records, **kwargs):
        return {"applied": True, "ingested": 0, "skipped": 0, "failed": [{"error": "boom"}]}

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", failing_ingest)
    result = asyncio.run(workspace.drain(apply=True))

    assert result["ingested"] == 0
    assert len(result["failed"]) == 1
    remaining = [item["id"] for item in workspace._read()]
    assert remaining == list(ids), "a failed proposal must stay queued for the next drain"
    assert not (configured / "archive.jsonl").exists()


def test_drain_archives_only_what_landed(configured, monkeypatch):
    from kg_mcp import workspace

    workspace.add_proposal("allowed", "a fact")
    workspace._human_set_status({i["id"] for i in workspace.pending_for()}, "approved")

    async def ok_ingest(records, **kwargs):
        return {"applied": True, "ingested": 1, "skipped": 0, "failed": []}

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", ok_ingest)
    result = asyncio.run(workspace.drain(apply=True))

    assert result["ingested"] == 1
    assert workspace._read() == []
    assert (configured / "archive.jsonl").exists()


# --- item 8: the bearer gate ---


def _call_middleware(token_sent: bytes | None, expected: str) -> int:
    from kg_mcp.server import BearerTokenMiddleware

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    headers = [(b"authorization", token_sent)] if token_sent is not None else []
    statuses: list[int] = []

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    async def receive():
        return {"type": "http.request"}

    app = BearerTokenMiddleware(downstream, expected)
    asyncio.run(app({"type": "http", "headers": headers}, receive, send))
    return statuses[0]


def test_missing_token_is_rejected():
    assert _call_middleware(None, "a-sufficiently-long-token") == 401


def test_wrong_token_is_rejected():
    assert _call_middleware(b"Bearer nope", "a-sufficiently-long-token") == 401


def test_correct_token_passes_through():
    assert _call_middleware(b"Bearer a-sufficiently-long-token", "a-sufficiently-long-token") == 200


def test_non_http_scope_is_not_gated():
    from kg_mcp.server import BearerTokenMiddleware

    seen = []

    async def downstream(scope, receive, send):
        seen.append(scope["type"])

    app = BearerTokenMiddleware(downstream, "a-sufficiently-long-token")
    asyncio.run(app({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]


# --- item 7: the export envelope ---


def test_export_omits_embeddings_and_tags_the_kind():
    from kg_mcp.export import EMBEDDING_FIELDS, _serialize

    class FakeNode:
        def model_dump(self, mode: str, exclude: set[str]):
            assert mode == "json"
            assert exclude == EMBEDDING_FIELDS
            return {"uuid": "u1", "name": "n"}

    assert _serialize(FakeNode(), "entity_node") == {
        "uuid": "u1",
        "name": "n",
        "kind": "entity_node",
    }
