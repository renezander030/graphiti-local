"""Retracting a proposal names the facts its episodes had ended; they stay ended.

graphiti-core records no link from an ended fact to what ended it, and it can drop or
overwrite the evidence for its true end, so a retraction cannot restore those facts
safely. It lists them for their own review (https://github.com/getzep/graphiti/issues/2165).
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace


def _edge(uuid_, fact, episodes, valid_at, invalid_at=None):
    return SimpleNamespace(
        uuid=uuid_,
        fact=fact,
        episodes=episodes,
        group_id="example",
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


def test_kg_retire_proposal_lists_the_facts_its_episode_ended(tmp_path, monkeypatch, capsys):
    """Scenario S1: ep2 said Petrova is the guest from 11-03 and ended Ivanov's stay."""
    from kg_mcp import cli, queries, workspace

    workspace_dir = tmp_path / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"graph:\n  groups: [example]\n  workspace_dir: {workspace_dir}\n", encoding="utf-8"
    )
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace_dir))

    nov = lambda day: datetime(2026, 11, day, tzinfo=timezone.utc)  # noqa: E731
    edges = [
        _edge("petrova", "Petrova is the guest in room 402", ["ep2"], nov(3)),
        _edge("ivanov", "Ivanov is the guest in room 402", ["ep1"], nov(1), nov(3)),
        # A fact ep2 itself gave that something else ended is not one ep2 ended.
        _edge("own", "Room 402 was cleaned", ["ep2"], nov(3), nov(4)),
    ]

    async def collect(graph, settings, groups, fetch):
        return edges

    @asynccontextmanager
    async def read_only(settings):
        yield None

    async def landed(records, **kwargs):
        return {
            "applied": True,
            "ingested": 1,
            "failed": [],
            "receipts": [{"episode_uuid": "ep2", "edges": 1}],
        }

    monkeypatch.setattr(queries, "collect", collect)
    monkeypatch.setattr(cli, "_read_only", read_only)
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", landed)
    item = workspace.add_proposal("example", "Petrova is the guest in room 402.")
    workspace._human_set_status({item["id"]}, "approved")

    asyncio.run(workspace.drain(apply=True))

    monkeypatch.setattr(
        sys,
        "argv",
        ["kg", "retire", "example", "--proposal", item["id"], "--reason", "approved by mistake"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["edges"] == ["petrova"]
    assert payload["ended_by_retracted"] == [
        {
            "uuid": "ivanov",
            "fact": "Ivanov is the guest in room 402",
            "invalid_at": "2026-11-03T00:00:00+00:00",
        }
    ]
    assert "stay ended" in payload["ended_by_retracted_note"]
    # Nothing in the queued item restores them.
    assert workspace.pending_for()[0]["edges"] == ["petrova"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kg",
            "-H",
            "retire",
            "example",
            "--proposal",
            item["id"],
            "--reason",
            "approved by mistake again",
        ],
    )
    cli.main()
    lines = capsys.readouterr().out.splitlines()
    assert any("ivanov" in line and "stays ended" in line for line in lines), lines
