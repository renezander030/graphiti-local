"""kg-demo: a graph cached by an older engine or older facts is rebuilt instead of failing
the read, and the answer puts what holds now before what a later fact replaced."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr("kg_mcp.ladybug.engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    assert len(built) == 1
    assert (tmp_path / ".built").read_text() == f"5 0.16.0 {demo._facts_digest()}\n"


def test_same_engine_reuses_the_graph(tmp_path, built, monkeypatch):
    monkeypatch.setattr("kg_mcp.ladybug.engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    demo._ensure_graph(tmp_path)
    assert len(built) == 1


def test_graph_from_an_older_release_is_rebuilt(tmp_path, built, monkeypatch):
    # what 0.4.0 to 0.9.0 left behind: a count without the engine, and a write-ahead log
    (tmp_path / ".built").write_text("5\n")
    (tmp_path / "graph.ladybug").write_text("old")
    (tmp_path / "graph.ladybug.wal").write_text("old")
    monkeypatch.setattr("kg_mcp.ladybug.engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    assert len(built) == 1
    assert (tmp_path / "graph.ladybug").read_text() == "graph"
    assert not (tmp_path / "graph.ladybug.wal").exists()
    assert (tmp_path / ".built").read_text() == f"5 0.16.0 {demo._facts_digest()}\n"


def test_graph_from_older_facts_is_rebuilt(tmp_path, built, monkeypatch):
    # what 0.9.1 left behind: the engine recorded, the facts not
    (tmp_path / ".built").write_text("5 0.16.0\n")
    (tmp_path / "graph.ladybug").write_text("old")
    monkeypatch.setattr("kg_mcp.ladybug.engine_version", lambda: "0.16.0")
    demo._ensure_graph(tmp_path)
    assert len(built) == 1
    assert (tmp_path / "graph.ladybug").read_text() == "graph"


def _edge(fact: str, valid: datetime, invalid: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(fact=fact, valid_at=valid, invalid_at=invalid)


def test_current_facts_come_before_replaced_ones():
    feb, jun = datetime(2026, 2, 1, tzinfo=timezone.utc), datetime(2026, 6, 14, tzinfo=timezone.utc)
    later = datetime.now(timezone.utc) + timedelta(days=30)
    edges = [
        _edge("uses DuckDB", feb, jun),
        _edge("uses PostgreSQL", jun),
        _edge("moves to ClickHouse", jun, later),  # ends in the future: still current
    ]
    current, replaced = demo._split(edges, limit=3)
    assert [x["fact"] for x in current] == ["uses PostgreSQL", "moves to ClickHouse"]
    assert replaced == [
        {"fact": "uses DuckDB", "valid_at": "2026-02-01", "invalid_at": "2026-06-14"}
    ]


def test_shipped_facts_replace_duckdb_with_postgresql():
    import json

    facts = json.loads(demo._facts_path().read_text())["facts"]
    duck = next(f for f in facts if f["fact"] == "Aurora Analytics uses DuckDB as its database.")
    pg = next(f for f in facts if f["fact"] == "Aurora Analytics uses PostgreSQL as its database.")
    assert duck["invalid_at"] == pg["valid_at"]
    assert "invalid_at" not in pg
