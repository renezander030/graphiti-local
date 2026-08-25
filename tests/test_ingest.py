from pathlib import Path

import pytest

from kg_mcp.ingest import read_records


def test_jsonl_validation_accepts_synthetic_records(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"name":"Example","body":"Synthetic body"}\n', encoding="utf-8")
    assert read_records(path)[0]["name"] == "Example"


def test_jsonl_validation_rejects_missing_body(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"name":"Example"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="missing body"):
        read_records(path)

