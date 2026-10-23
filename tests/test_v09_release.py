"""0.9.0: facts whose evidence speaks for itself can be approved by a rule at filing."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from kg_mcp.gates import content_digest
from kg_mcp.output import EXIT_SECRET_SHAPED, CommandError

RULES = """workspace:
  auto_approve:
    - {fact_type: source-fact, require: source-link}
    - {fact_type: action-record, require: said}
"""


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str = "") -> Path:
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"graph:\n  groups: [example]\n  workspace_dir: {workspace}\n{extra}", encoding="utf-8"
    )
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace))
    return workspace


@pytest.fixture
def ruled(tmp_path, monkeypatch) -> Path:
    return _configure(tmp_path, monkeypatch, RULES)


class Ingest:
    def __init__(self) -> None:
        self.records: list[dict] = []

    async def __call__(self, records, **kwargs):
        self.records.extend(records)
        receipt = {"episode_uuid": f"episode-{len(self.records)}", "edges": 1}
        return {"applied": True, "ingested": 1, "failed": [], "receipts": [receipt]}


def test_without_rules_every_proposal_waits(tmp_path, monkeypatch):
    from kg_mcp import workspace

    _configure(tmp_path, monkeypatch)
    item = workspace.add_proposal(
        "example", "Invoice 42 was paid.", fact_type="source-fact", source="ledger", link="x:1"
    )
    assert item["status"] == "pending" and "decided_by" not in item


def test_a_rule_cannot_approve_an_inference(tmp_path, monkeypatch):
    from kg_mcp.config import load_config

    rule = "workspace:\n  auto_approve:\n    - {fact_type: belief, require: said}\n"
    _configure(tmp_path, monkeypatch, rule)
    with pytest.raises(ValueError, match="fact_type"):
        load_config()


def test_a_sourced_record_is_approved_and_bound_at_filing(ruled, monkeypatch):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example",
        "Invoice 42 was paid.",
        fact_type="source-fact",
        source="ledger",
        link="ledger:invoice/42",
    )
    assert item["status"] == "approved"
    assert item["decided_by"] == "policy:source-fact:source-link"
    assert item["approved_digest"] == content_digest(item)
    assert workspace.pending_for() == []

    planned = asyncio.run(workspace.drain(apply=False))["planned"]
    assert [(row["id"], row["binding"], row["decided_by"]) for row in planned] == [
        (item["id"], "ok", "policy:source-fact:source-link")
    ]
    ingest = Ingest()
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", ingest)
    assert asyncio.run(workspace.drain(apply=True))["ingested"] == 1
    assert "source: ledger" in ingest.records[0]["provenance"]


@pytest.mark.parametrize(
    "fields",
    [
        {"fact_type": "source-fact", "source": "ledger"},  # no link
        {"fact_type": "source-fact", "link": "ledger:invoice/42"},  # no source
        {"fact_type": "action-record", "source": "crm", "link": "crm:1"},  # rule wants --said
        {"fact_type": "belief", "said": "I think so", "source": "a", "link": "b"},
        {"fact_type": "live-finding", "source": "a", "link": "b"},  # no rule for the type
    ],
)
def test_a_proposal_without_the_rules_evidence_waits(ruled, fields):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "Invoice 42 was paid.", **fields)
    assert item["status"] == "pending"


def test_a_correction_waits_even_with_the_evidence(ruled):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example",
        "Invoice 42 was refunded.",
        operation="revise",
        supersedes="Invoice 42 was paid.",
        fact_type="source-fact",
        source="ledger",
        link="ledger:invoice/42",
    )
    assert item["status"] == "pending"


def test_the_reviewers_own_words_approve_and_travel_with_the_fact(ruled, monkeypatch):
    from kg_mcp import workspace

    said = "We moved Aurora to the Frankfurt region today."
    item = workspace.add_bundle(
        "example",
        ["Aurora runs in Frankfurt.", "Aurora left Dublin."],
        fact_type="action-record",
        said=said,
    )
    assert (item["status"], item["said"]) == ("approved", said)
    assert item["decided_by"] == "policy:action-record:said"

    ingest = Ingest()
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", ingest)
    assert asyncio.run(workspace.drain(apply=True))["ingested"] == 1
    assert all(f"said: {said}" in record["provenance"] for record in ingest.records)


def test_changed_words_unbind_the_approval(ruled):
    from kg_mcp import workspace

    workspace.add_proposal("example", "Aurora left Dublin.", fact_type="action-record", said="yes")
    queued = workspace._read()
    queued[0]["said"] = "no"
    workspace._write("pending.jsonl", queued)
    planned = asyncio.run(workspace.drain(apply=False))["planned"]
    assert "approve it again" in planned[0]["binding"]


def test_secret_shaped_words_are_refused(ruled):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_proposal(
            "example",
            "Aurora left Dublin.",
            fact_type="action-record",
            said="the key is sk-ant-" + "a" * 30,
        )
    assert refused.value.code == EXIT_SECRET_SHAPED


def test_an_approved_item_can_still_be_rejected_before_it_drains(ruled, monkeypatch, capsys):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example", "Invoice 42 was paid.", fact_type="source-fact", source="l", link="l:42"
    )
    monkeypatch.setattr(
        sys, "argv", ["kg-workspace", "reject", item["id"], "--reason", "wrong invoice"]
    )
    workspace.main()
    result = json.loads(capsys.readouterr().out)
    assert (result["changed"], result["unknown"]) == (1, [])
    assert workspace._read() == []
    assert workspace._read("archive.jsonl")[0]["reason"] == "wrong invoice"
    assert asyncio.run(workspace.drain(apply=True))["ingested"] == 0


def test_a_reviewer_approval_is_labelled_in_the_plan(tmp_path, monkeypatch):
    from kg_mcp import workspace

    _configure(tmp_path, monkeypatch)
    item = workspace.add_proposal("example", "Aurora ships weekly.")
    workspace._human_set_status({item["id"]}, "approved")
    planned = asyncio.run(workspace.drain(apply=False))["planned"]
    assert planned[0]["decided_by"] == "reviewer"


def test_kg_propose_says_who_approved(ruled, monkeypatch, capsys):
    from kg_mcp import cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["kg", "propose", "example", "Aurora left Dublin.", "--type", "action-record",
         "--said", "We left Dublin."],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert (payload["status"], payload["decided_by"]) == ("approved", "policy:action-record:said")


def test_a_fact_filed_over_a_rejection_waits_for_a_person(ruled):
    from kg_mcp import workspace

    first = workspace.add_proposal(
        "example", "Invoice 42 was paid.", fact_type="source-fact", source="l", link="l:42"
    )
    workspace._human_set_status({first["id"]}, "rejected", reason="wrong invoice")
    again = workspace.add_proposal(
        "example",
        "Invoice 42 was paid.",
        fact_type="source-fact",
        source="l",
        link="l:42",
        supersedes_rejection=first["id"],
    )
    assert again["status"] == "pending" and "decided_by" not in again


def test_a_rejection_drops_the_rules_approval(ruled):
    from kg_mcp import workspace

    item = workspace.add_proposal(
        "example", "Invoice 42 was paid.", fact_type="source-fact", source="l", link="l:42"
    )
    workspace._human_set_status({item["id"]}, "rejected")
    archived = workspace._read("archive.jsonl")[0]
    assert archived["decided_by"] == "reviewer" and "approved_digest" not in archived


def test_a_partly_drained_bundle_cannot_be_rejected(ruled, monkeypatch):
    from kg_mcp import workspace

    calls = []

    async def first_lands(records, **kwargs):
        calls.append(records)
        if len(calls) > 1:
            return {"applied": True, "ingested": 0, "failed": [{"error": "down"}], "receipts": []}
        receipt = {"episode_uuid": "episode-1", "edges": 1}
        return {"applied": True, "ingested": 1, "failed": [], "receipts": [receipt]}

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", first_lands)
    item = workspace.add_bundle(
        "example", ["Aurora runs in Frankfurt.", "Aurora left Dublin."],
        fact_type="action-record", said="We moved Aurora.",
    )
    asyncio.run(workspace.drain(apply=True))
    assert workspace._read()[0]["landed"] == [0]
    result = workspace._human_set_status({item["id"]}, "rejected")
    assert result["unknown"] == [item["id"]]
    assert workspace._read()[0]["status"] == "approved"


def test_a_drain_skips_an_item_rejected_after_it_read_the_queue(ruled, monkeypatch):
    from kg_mcp import workspace

    first, second = (
        workspace.add_proposal(
            "example", text, fact_type="source-fact", source="l", link=f"l:{number}"
        )
        for number, text in ((1, "Invoice 1 was paid."), (2, "Invoice 2 was paid."))
    )
    ingest = Ingest()
    outcomes = {}

    async def reject_during_drain(records, **kwargs):
        # While the first item is being written, a reviewer rejects both items.
        outcomes.update(workspace._human_set_status({first["id"], second["id"]}, "rejected"))
        return await ingest(records, **kwargs)

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", reject_during_drain)
    result = asyncio.run(workspace.drain(apply=True))
    # The first was claimed before its write, so the rejection could not take it back;
    # the second was rejected before the drain reached it, so it never reached the graph.
    assert outcomes["unknown"] == [first["id"]]
    assert result["ingested"] == 1
    assert [record["name"] for record in ingest.records] == [f"workspace {first['id']}"]
    archived = {row["id"]: row["status"] for row in workspace._read("archive.jsonl")}
    assert archived == {first["id"]: "approved", second["id"]: "rejected"}
