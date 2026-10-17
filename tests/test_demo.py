"""kg-demo: a graph cached by an older engine is rebuilt instead of failing the read."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_mcp import demo


def _fake_build(calls: list[Path]):
    def setup(database: Path, quiet: bool = False) -> list[str]:
        calls.append(database)
        database.write_text("graph")
        return []

    return setup


@pytest.fixture
def built(tmp_path, monkeypatch) -> list[Path]:
    calls: list[Path] = []
    monkeypatch.setattr("kg_mcp.ladybug.setup_database", _fake_build(calls))

    async def build(database: Path) -> int:
        return 5

    monkeypatch.setattr(demo, "_build", build)
    return calls


def test_first_run_builds_and_records_the_engine(tmp_path, built, monkeypatch):
    monkeypatch.setattr(demo, "_engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    assert len(built) == 1
    assert (tmp_path / ".built").read_text() == "5 0.16.0\n"


def test_same_engine_reuses_the_graph(tmp_path, built, monkeypatch):
    monkeypatch.setattr(demo, "_engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    demo._ensure_graph(tmp_path)
    assert len(built) == 1


def test_graph_from_an_older_release_is_rebuilt(tmp_path, built, monkeypatch):
    # what 0.4.0 to 0.9.0 left behind: a count without the engine, and a write-ahead log
    (tmp_path / ".built").write_text("5\n")
    (tmp_path / "graph.ladybug").write_text("old")
    (tmp_path / "graph.ladybug.wal").write_text("old")
    monkeypatch.setattr(demo, "_engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    assert len(built) == 1
    assert (tmp_path / "graph.ladybug").read_text() == "graph"
    assert not (tmp_path / "graph.ladybug.wal").exists()
    assert (tmp_path / ".built").read_text() == "5 0.16.0\n"
