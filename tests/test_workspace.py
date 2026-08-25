import json
from pathlib import Path

import pytest

from kg_mcp.workspace import add_proposal, pending_for


@pytest.fixture
def configured_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"graph:\n  groups: [example]\n  workspace_dir: {tmp_path / 'queue'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    return tmp_path / "queue"


def test_proposal_is_pending_and_append_only(configured_workspace: Path):
    item = add_proposal(
        "example",
        "A synthetic fact.",
        fact_type="source-fact",
        provenance="unit test",
    )
    assert item["status"] == "pending"
    assert pending_for(["example"]) == [item]
    lines = (configured_workspace / "pending.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["text"] == "A synthetic fact."


def test_revision_requires_superseded_fact(configured_workspace: Path):
    with pytest.raises(SystemExit, match="requires --supersedes"):
        add_proposal("example", "Correction", operation="revise")
