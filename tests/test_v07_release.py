"""The 0.7.0 contract: reviewed writes that land exactly what was approved."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kg_mcp.output import (
    EXIT_NOT_ATOMIC,
    EXIT_REJECTED,
    EXIT_RESEMBLES_REJECTED,
    EXIT_SECRET_SHAPED,
    CommandError,
)


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace_block: str = "") -> Path:
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"graph:\n  groups: [example]\n  workspace_dir: {workspace}\n{workspace_block}",
        encoding="utf-8",
    )
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace))
    return workspace


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _configure(tmp_path, monkeypatch)


class Ingest:
    """Records every drained record; ``edges`` decides what each call yields."""

    def __init__(self, *edges: int | None) -> None:
        self.edges = list(edges)
        self.records: list[dict] = []

    async def __call__(self, records, **kwargs):
        self.records.extend(records)
        edges = self.edges.pop(0) if self.edges else 1
        if edges is None:
            return {"applied": True, "ingested": 0, "failed": [{"error": "boom"}]}
        receipt = {"episode_uuid": f"episode-{len(self.records)}", "edges": edges}
        return {"applied": True, "ingested": 1, "failed": [], "receipts": [receipt]}


def _approve_all(workspace) -> None:
    workspace._human_set_status({item["id"] for item in workspace.pending_for()}, "approved")


def _drain(monkeypatch, ingest: Ingest) -> dict:
    from kg_mcp import workspace

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", ingest)
    return asyncio.run(workspace.drain(apply=True))


def test_release_metadata_is_0_7_0():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "server.json").read_text(encoding="utf-8"))
    assert manifest["version"] == manifest["packages"][0]["version"] == "0.7.0"
    assert 'version = "0.7.0"' in (root / "pyproject.toml").read_text(encoding="utf-8")


# --- an approved proposal that yields no fact is not reported as landed ---


def test_drain_keeps_a_proposal_whose_extraction_produced_no_facts(configured, monkeypatch):
    from kg_mcp import workspace

    workspace.add_proposal("example", "Aurora Analytics uses DuckDB.")
    _approve_all(workspace)
    result = _drain(monkeypatch, Ingest(0))
    assert result["ingested"] == 0
    assert result["failed"][0]["error"] == "extraction produced no facts from this proposal"
    assert [item["status"] for item in workspace._read()] == ["approved"]
    assert not (configured / "archive.jsonl").exists()

    retried = _drain(monkeypatch, Ingest(2))
    assert retried["ingested"] == 1 and workspace._read() == []
    archived = workspace._read("archive.jsonl")[0]
    assert archived["episodes"] == ["episode-1"]


def test_ingest_reports_records_that_yield_no_facts(tmp_path, monkeypatch, capsys):
    from kg_mcp import ingest

    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"name": "heading", "body": "Notes"}) + "\n", encoding="utf-8")

    async def fake_ingest(records, **kwargs):
        receipt = {"name": "heading", "domain": "example", "episode_uuid": "e1", "edges": 0}
        return {
            "applied": True,
            "ingested": 1,
            "skipped": 0,
            "failed": [],
            "interrupted": False,
            "planned": [],
            "receipts": [receipt],
        }

    monkeypatch.setattr(ingest, "ingest_records", fake_ingest)
    monkeypatch.setattr(sys, "argv", ["kg-ingest", str(source), "--apply"])
    ingest.main()
    output = json.loads(capsys.readouterr().out)
    assert output["no_facts"] == [{"name": "heading", "domain": "example", "episode_uuid": "e1"}]


# --- a fact is dated by when it happened, not when it was filed ---


def test_drain_dates_a_fact_by_its_valid_at(configured, monkeypatch):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example",
        "Aurora Analytics moved to PostgreSQL.",
        valid_at="2026-03-01",
        learned_at="2026-10-01T09:30:00Z",
    )
    assert item["valid_at"] == "2026-03-01T00:00:00+00:00"
    assert item["learned_at"] == "2026-10-01T09:30:00+00:00"
    _approve_all(workspace)
    ingest = Ingest(1)
    _drain(monkeypatch, ingest)
    assert ingest.records[0]["valid_at"] == "2026-03-01T00:00:00+00:00"
    assert "learned_at: 2026-10-01T09:30:00+00:00" in ingest.records[0]["provenance"]


def test_undated_fact_still_falls_back_to_the_filing_time(configured, monkeypatch):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "A fact without a date.")
    _approve_all(workspace)
    ingest = Ingest(1)
    _drain(monkeypatch, ingest)
    assert ingest.records[0]["valid_at"] == item["created_at"]


@pytest.mark.parametrize("value", ["2999-01-01", "yesterday"])
def test_future_or_malformed_valid_at_is_refused(configured, value):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_proposal("example", "A fact.", valid_at=value)
    assert refused.value.code == EXIT_REJECTED
    assert workspace.pending_for() == []


def test_correction_body_carries_the_valid_date(configured):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example",
        "Aurora uses PostgreSQL.",
        operation="revise",
        supersedes="Aurora uses DuckDB.",
        valid_at="2026-03-01",
    )
    assert workspace._episode_body(item).startswith("Correction as of 2026-03-01:")


# --- rejections are remembered ---


def test_rejection_is_archived_with_its_reason(configured):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "Aurora Analytics uses MongoDB.")
    result = workspace._human_set_status({item["id"]}, "rejected", reason="wrong system")
    assert result == {"status": "rejected", "changed": 1, "unknown": []}
    assert workspace._read() == []
    archived = workspace.rejected_for(["example"])[0]
    assert archived["id"] == item["id"] and archived["reason"] == "wrong system"


def test_a_proposal_resembling_a_rejection_is_refused_until_acknowledged(configured):
    from kg_mcp import workspace

    rejected = workspace.add_proposal("example", "Aurora Analytics uses MongoDB.")
    workspace._human_set_status({rejected["id"]}, "rejected", reason="wrong system")
    with pytest.raises(CommandError) as refused:
        workspace.add_proposal("example", "Aurora Analytics uses MongoDB now.")
    assert refused.value.code == EXIT_RESEMBLES_REJECTED
    assert rejected["id"] in str(refused.value) and "wrong system" in str(refused.value)

    accepted = workspace.add_proposal(
        "example", "Aurora Analytics uses MongoDB now.", supersedes_rejection=rejected["id"]
    )
    assert accepted["supersedes_rejection"] == rejected["id"]
    # The correction itself and unrelated text are not affected.
    workspace.add_proposal("example", "Aurora Analytics uses DuckDB.")
    workspace.add_proposal("example", "Borealis Labs uses MongoDB.")


def test_rejections_left_in_an_older_queue_file_still_count(configured):
    from kg_mcp import workspace

    configured.mkdir(parents=True, exist_ok=True)
    legacy = {
        "id": "proposal-legacy",
        "domain": "example",
        "text": "Aurora Analytics uses MongoDB.",
        "status": "rejected",
    }
    (configured / "pending.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    with pytest.raises(CommandError):
        workspace.add_proposal("example", "Aurora Analytics uses MongoDB.")


def test_rejection_similarity_is_configurable(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "workspace:\n  reject_similarity: 1.0\n")
    from kg_mcp import workspace

    rejected = workspace.add_proposal("example", "Aurora Analytics uses MongoDB.")
    workspace._human_set_status({rejected["id"]}, "rejected")
    workspace.add_proposal("example", "Aurora Analytics uses MongoDB now.")


# --- secrets never enter the queue ---


@pytest.mark.parametrize(
    ("text", "provenance"),
    [
        ("The deploy key is sk-ant-" + "a" * 30 + ".", ""),
        ("The CI token rotated.", "ghp_" + "b" * 36),
        ("The database password: hunter2hunter2.", ""),
        ("Payroll goes to DE89 3704 0044 0532 0130 00.", ""),
        ("-----BEGIN " + "OPENSSH PRIVATE KEY----- is in the vault.", ""),
    ],
)
def test_secret_shapes_are_refused_without_echoing_the_value(configured, text, provenance):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_proposal("example", text, provenance=provenance)
    assert refused.value.code == EXIT_SECRET_SHAPED
    assert "hunter2" not in str(refused.value) and "bbbbbbbb" not in str(refused.value)
    assert workspace.pending_for() == []


def test_identifiers_that_look_technical_are_not_secrets():
    from kg_mcp.gates import secret_shapes

    assert secret_shapes("Commit 4c34c96 fixed the release audit.") == []
    assert secret_shapes("Contact ops" + "@" + "example.com for access.") == []
    assert secret_shapes("Wallet 0x52908400098527886E0F7030069857D2E4169EE7 holds funds.") == []


# --- source system and link ---


def test_source_and_link_travel_into_the_episode_provenance(configured, monkeypatch):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example",
        "Aurora Analytics signed the renewal.",
        source="crm",
        link="https://crm.example.com/deals/42",
    )
    assert (item["source"], item["link"]) == ("crm", "https://crm.example.com/deals/42")
    _approve_all(workspace)
    ingest = Ingest(1)
    _drain(monkeypatch, ingest)
    provenance = ingest.records[0]["provenance"]
    assert "source: crm" in provenance and "link: https://crm.example.com/deals/42" in provenance


def test_require_source_refuses_a_fact_without_one(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "workspace:\n  require_source: true\n")
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_proposal("example", "An unsourced fact.", source="crm")
    assert refused.value.code == EXIT_REJECTED
    workspace.add_proposal("example", "A sourced fact.", source="crm", link="https://x.test/1")


# --- a fact traces back to its episodes ---


def test_mcp_fact_payload_lists_its_episodes():
    from kg_mcp.server import _edge

    edge = SimpleNamespace(
        uuid="e1",
        name="USES",
        fact="Aurora uses DuckDB.",
        source_node_uuid="n1",
        target_node_uuid="n2",
        episodes=["ep1", "ep2"],
        group_id="example",
        created_at=None,
        valid_at=None,
        invalid_at=None,
        expired_at=None,
    )
    assert _edge(edge)["episodes"] == ["ep1", "ep2"]


def test_edge_episode_sources_carry_their_provenance(monkeypatch):
    from graphiti_core.nodes import EpisodicNode

    from kg_mcp.cli import _episode_sources

    async def get_by_uuids(driver, uuids):
        return [
            SimpleNamespace(
                uuid="ep1",
                name="workspace proposal-1",
                source_description="Graphiti Local workspace; source-fact; source: crm",
                valid_at=None,
            )
        ]

    monkeypatch.setattr(EpisodicNode, "get_by_uuids", get_by_uuids)
    sources = asyncio.run(_episode_sources(object(), ["ep1", "gone"], 5))
    assert sources[0]["source_description"].endswith("source: crm")
    assert sources[1]["uuid"] == "gone" and sources[1]["source_description"] is None


# --- one fact per proposal ---


def test_a_compound_proposal_is_refused_with_its_sentences(configured):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_proposal("example", "Aurora uses DuckDB. Borealis uses Postgres.")
    assert refused.value.code == EXIT_NOT_ATOMIC
    assert "1. Aurora uses DuckDB." in str(refused.value)


@pytest.mark.parametrize(
    "text",
    [
        "Aurora runs Postgres 16.2 on 2026-03-01.",
        "Aurora uses a column store, e.g. DuckDB for analytics.",
        "Dr. Weber approved the budget.",
        "See https://example.com/a.b for the spec.",
    ],
)
def test_single_sentences_are_not_split(text):
    from kg_mcp.gates import sentences

    assert sentences(text) == [text]


def test_atomic_gate_can_be_turned_off(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "workspace:\n  atomic_proposals: false\n")
    from kg_mcp import workspace

    workspace.add_proposal("example", "Aurora uses DuckDB. Borealis uses Postgres.")


# --- an approval covers exactly the content it saw ---


def test_content_edited_after_approval_is_not_drained(configured, monkeypatch):
    from kg_mcp import workspace

    workspace.add_proposal("example", "Aurora uses DuckDB.")
    _approve_all(workspace)
    items = workspace._read()
    items[0]["text"] = "Aurora uses MongoDB."
    workspace._write("pending.jsonl", items)

    ingest = Ingest(1)
    result = _drain(monkeypatch, ingest)
    assert ingest.records == []
    assert "changed after approval" in result["failed"][0]["error"]

    workspace._human_set_status({items[0]["id"]}, "approved")  # still bound to the old text
    assert _drain(monkeypatch, ingest)["failed"]


def test_an_approval_without_a_digest_is_bound_by_approving_again(configured, monkeypatch):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "Aurora uses DuckDB.")
    items = workspace._read()
    items[0]["status"] = "approved"  # approved by an earlier version
    workspace._write("pending.jsonl", items)
    assert "approve it again" in _drain(monkeypatch, Ingest(1))["failed"][0]["error"]

    assert workspace._human_set_status({item["id"]}, "approved")["changed"] == 1
    assert _drain(monkeypatch, Ingest(1))["ingested"] == 1


def test_approving_an_unknown_id_exits_non_zero(configured, monkeypatch, capsys):
    from kg_mcp import workspace

    monkeypatch.setattr(sys, "argv", ["kg-workspace", "approve", "proposal-missing"])
    with pytest.raises(SystemExit) as exited:
        workspace.main()
    assert exited.value.code == EXIT_REJECTED
    assert json.loads(capsys.readouterr().out)["unknown"] == ["proposal-missing"]


# --- several facts from one source, one approval ---


def test_bundle_drains_each_fact_and_resumes_after_a_partial_failure(configured, monkeypatch):
    from kg_mcp import workspace

    item = workspace.add_bundle(
        "example",
        ["Aurora uses DuckDB.", "Aurora runs in Frankfurt."],
        source="meeting notes",
        link="file:notes/2026-03-01.md",
        valid_at="2026-03-01",
    )
    assert item["facts"] == ["Aurora uses DuckDB.", "Aurora runs in Frankfurt."]
    _approve_all(workspace)

    first = Ingest(1, None)
    result = _drain(monkeypatch, first)
    assert result["ingested"] == 0 and result["failed"][0]["fact"] == 2
    assert [record["name"] for record in first.records] == [
        f"workspace {item['id']} #1",
        f"workspace {item['id']} #2",
    ]
    assert workspace._read()[0]["landed"] == [0]

    second = Ingest(1)
    assert _drain(monkeypatch, second)["ingested"] == 1
    assert [record["body"] for record in second.records] == ["Aurora runs in Frankfurt."]
    assert workspace._read("archive.jsonl")[0]["landed"] == [0, 1]


def test_every_fact_in_a_bundle_passes_the_gates(configured):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_bundle(
            "example", ["Aurora uses DuckDB.", "One. Two."], provenance="meeting notes"
        )
    assert refused.value.code == EXIT_NOT_ATOMIC
    with pytest.raises(CommandError) as unsourced:
        workspace.add_bundle("example", ["Aurora uses DuckDB.", "Aurora runs in Frankfurt."])
    assert unsourced.value.code == EXIT_REJECTED


def test_cli_bundle_and_gate_exit_codes(configured, monkeypatch, capsys):
    from kg_mcp import cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["kg", "bundle", "example", "--fact", "A uses B.", "--fact", "C uses D.", "--prov", "x"],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["facts"] == ["A uses B.", "C uses D."]

    monkeypatch.setattr(sys, "argv", ["kg", "propose", "example", "One fact. Another fact."])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == EXIT_NOT_ATOMIC
