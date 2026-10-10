"""The embedded backend: read-only opens, index creation, and reopening on change.

These run against a real Ladybug file and need the FTS and VECTOR extensions on the
host. They skip otherwise; set ``KG_TEST_INSTALL_EXTENSIONS=1`` to let the run install
them (a download).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _extensions_available() -> bool:
    try:
        import ladybug
    except ImportError:
        return False
    database = ladybug.Database(":memory:")
    connection = ladybug.Connection(database)
    for extension in ("FTS", "VECTOR"):
        try:
            connection.execute(f"LOAD EXTENSION {extension}")
        except Exception:
            if os.environ.get("KG_TEST_INSTALL_EXTENSIONS") != "1":
                return False
            try:
                connection.execute(f"INSTALL {extension}")
                connection.execute(f"LOAD EXTENSION {extension}")
            except Exception:
                return False
    return True


pytestmark = pytest.mark.skipif(
    not _extensions_available(), reason="Ladybug search extensions are not installed"
)


def _tables_only(path: Path) -> None:
    """A database as graphiti-core alone would leave it: schema, no full-text indexes."""
    from kg_mcp.ladybug import _alias_kuzu

    ladybug = _alias_kuzu()
    from graphiti_core.driver.kuzu_driver import SCHEMA_QUERIES

    database = ladybug.Database(str(path))
    connection = ladybug.Connection(database)
    connection.execute(SCHEMA_QUERIES)
    connection.close()
    database.close()


def _writer(path: Path):
    import ladybug

    from kg_mcp.ladybug import load_extensions

    database = ladybug.Database(str(path))
    connection = ladybug.Connection(database)
    assert load_extensions(connection) == []
    return database, connection


def _count(driver) -> int:
    rows, _, _ = asyncio.run(driver.execute_query("MATCH (n:Entity) RETURN count(n) AS c"))
    return rows[0]["c"]


def test_write_open_creates_schema_and_every_fulltext_index(tmp_path: Path):
    from kg_mcp.ladybug import build_ladybug_driver, fulltext_indexes, missing_indexes

    driver = build_ladybug_driver(str(tmp_path / "graph.ladybug"), read_only=False)
    try:
        _, connection = _writer(tmp_path / "graph.ladybug")
        assert missing_indexes(connection) == []
        assert len(fulltext_indexes()) == 4
        connection.close()
    finally:
        asyncio.run(driver.close())


def test_read_only_open_refuses_a_missing_database(tmp_path: Path):
    from kg_mcp.ladybug import build_ladybug_driver

    with pytest.raises(RuntimeError, match="kg-ladybug-setup"):
        build_ladybug_driver(str(tmp_path / "missing.ladybug"), read_only=True)


def test_read_only_open_refuses_a_database_without_indexes(tmp_path: Path):
    from kg_mcp.ladybug import build_ladybug_driver, inspect_database

    path = tmp_path / "bare.ladybug"
    _tables_only(path)
    assert set(inspect_database(str(path))["missing_indexes"]) == {
        "episode_content",
        "node_name_and_summary",
        "community_name",
        "edge_name_and_fact",
    }
    with pytest.raises(RuntimeError, match="no full-text index"):
        build_ladybug_driver(str(path), read_only=True)


def test_a_reader_coexists_with_a_writer(tmp_path: Path):
    from kg_mcp.ladybug import build_ladybug_driver

    path = tmp_path / "graph.ladybug"
    asyncio.run(build_ladybug_driver(str(path), read_only=False).close())
    reader = build_ladybug_driver(str(path), read_only=True)
    try:
        assert reader.read_only is True
        database, connection = _writer(path)  # would raise "Could not set lock" against a writer
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: 'Probe', group_id: '', created_at: $t, "
            "labels: ['Entity']})",
            {"u": str(uuid.uuid4()), "t": datetime.now(timezone.utc)},
        )
        connection.close()
        database.close()
    finally:
        asyncio.run(reader.close())


def test_reopen_if_changed_picks_up_another_process_commit(tmp_path: Path):
    from kg_mcp.ladybug import build_ladybug_driver

    path = tmp_path / "graph.ladybug"
    asyncio.run(build_ladybug_driver(str(path), read_only=False).close())
    reader = build_ladybug_driver(str(path), read_only=True)
    try:
        assert _count(reader) == 0
        assert reader.reopen_if_changed() is False, "nothing changed yet"
        database, connection = _writer(path)
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: 'Probe', group_id: '', created_at: $t, "
            "labels: ['Entity']})",
            {"u": str(uuid.uuid4()), "t": datetime.now(timezone.utc)},
        )
        connection.execute("CHECKPOINT")
        connection.close()
        database.close()
        assert _count(reader) == 0, "a read-only handle sees the file as it was opened"
        assert reader.reopen_if_changed() is True
        assert _count(reader) == 1
    finally:
        asyncio.run(reader.close())


def _fake_embedder():
    """Deterministic vectors so the search path runs without a model server."""
    from graphiti_core.embedder.client import EmbedderClient

    class FakeEmbedder(EmbedderClient):
        async def create(self, input_data):
            return [0.5] * 8

        async def create_batch(self, input_data_list):
            return [[0.5] * 8 for _ in input_data_list]

    return FakeEmbedder()


def _graph(path: Path, *, read_only: bool):
    os.environ.setdefault("OPENAI_API_KEY", "test-only")
    from graphiti_core import Graphiti

    from kg_mcp.ladybug import build_ladybug_driver
    from kg_mcp.reranker import PassthroughReranker

    driver = build_ladybug_driver(str(path), read_only=read_only)
    return Graphiti(
        graph_driver=driver, embedder=_fake_embedder(), cross_encoder=PassthroughReranker()
    )


def test_search_on_an_empty_graph_returns_no_facts(tmp_path: Path):
    """Issue #1: this raised a Binder exception because the indexes never existed."""
    path = tmp_path / "graph.ladybug"
    asyncio.run(_graph(path, read_only=False).close())
    graph = _graph(path, read_only=True)
    try:
        assert asyncio.run(graph.search("Who is Ada Lovelace?", num_results=5)) == []
    finally:
        asyncio.run(graph.close())


def test_search_finds_a_fact_written_by_another_process(tmp_path: Path):
    path = tmp_path / "graph.ladybug"
    asyncio.run(_graph(path, read_only=False).close())
    database, connection = _writer(path)
    now = datetime.now(timezone.utc)
    a, b, e = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    for node in (a, b):
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: $n, group_id: '', created_at: $t, "
            "name_embedding: $v, summary: '', labels: ['Entity']})",
            {"u": node, "n": f"node {node[:4]}", "t": now, "v": [0.5] * 8},
        )
    connection.execute(
        "MATCH (x:Entity {uuid: $a}), (y:Entity {uuid: $b}) "
        "CREATE (x)-[:RELATES_TO]->(:RelatesToNode_ {uuid: $e, name: 'WROTE', fact: $f, "
        "group_id: '', created_at: $t, fact_embedding: $v, episodes: []})-[:RELATES_TO]->(y)",
        {
            "a": a,
            "b": b,
            "e": e,
            "f": "Ada Lovelace wrote the first program",
            "t": now,
            "v": [0.5] * 8,
        },
    )
    connection.execute("CHECKPOINT")
    connection.close()
    database.close()
    graph = _graph(path, read_only=True)
    try:
        facts = asyncio.run(graph.search("who wrote the first program", num_results=5))
        assert [fact.fact for fact in facts] == ["Ada Lovelace wrote the first program"]
    finally:
        asyncio.run(graph.close())


def test_collect_sweeps_the_empty_group_ingestion_writes(tmp_path: Path):
    """Ingestion writes group_id None; a scan by configured group must still see it."""
    from types import SimpleNamespace

    from graphiti_core.edges import EntityEdge

    from kg_mcp.queries import collect

    path = tmp_path / "graph.ladybug"
    asyncio.run(_graph(path, read_only=False).close())
    database, connection = _writer(path)
    now = datetime.now(timezone.utc)
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    for node in (a, b):
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: $u, group_id: '', created_at: $t, "
            "name_embedding: $v, summary: '', labels: ['Entity']})",
            {"u": node, "t": now, "v": [0.5] * 8},
        )
    connection.execute(
        "MATCH (x:Entity {uuid: $a}), (y:Entity {uuid: $b}) "
        "CREATE (x)-[:RELATES_TO]->(:RelatesToNode_ {uuid: $e, name: 'USED', fact: 'old', "
        "group_id: '', created_at: $t, invalid_at: $t, fact_embedding: $v, episodes: []})"
        "-[:RELATES_TO]->(y)",
        {"a": a, "b": b, "e": str(uuid.uuid4()), "t": now, "v": [0.5] * 8},
    )
    connection.execute("CHECKPOINT")
    connection.close()
    database.close()
    settings = SimpleNamespace(
        database=SimpleNamespace(provider="ladybug"), graph=SimpleNamespace(groups=["example"])
    )
    graph = _graph(path, read_only=True)
    try:
        edges = asyncio.run(collect(graph, settings, ["example"], EntityEdge.get_by_group_ids))
        assert [edge.fact for edge in edges] == ["old"]
        assert edges[0].invalid_at is not None
    finally:
        asyncio.run(graph.close())


def test_verify_checks_invalidation_on_a_single_ladybug_file(tmp_path: Path, monkeypatch):
    """kg verify must find a superseded fact ingestion wrote, and see it is not served."""
    from datetime import timedelta

    from kg_mcp.verify import check_retrieval

    path = tmp_path / "graph.ladybug"
    asyncio.run(_graph(path, read_only=False).close())
    database, connection = _writer(path)
    now = datetime.now(timezone.utc)
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    for node in (a, b):
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: $u, group_id: '', created_at: $t, "
            "name_embedding: $v, summary: '', labels: ['Entity']})",
            {"u": node, "t": now, "v": [0.5] * 8},
        )
    for fact, invalid_at in (
        ("Aurora uses Postgres", now - timedelta(days=1)),
        ("Aurora uses DuckDB", None),
    ):
        connection.execute(
            "MATCH (x:Entity {uuid: $a}), (y:Entity {uuid: $b}) "
            "CREATE (x)-[:RELATES_TO]->(:RelatesToNode_ {uuid: $e, name: 'USES', fact: $f, "
            "group_id: '', created_at: $t, invalid_at: $i, fact_embedding: $v, episodes: []})"
            "-[:RELATES_TO]->(y)",
            {
                "a": a,
                "b": b,
                "e": str(uuid.uuid4()),
                "f": fact,
                "t": now,
                "i": invalid_at,
                "v": [0.5] * 8,
            },
        )
    connection.execute("CHECKPOINT")
    connection.close()
    database.close()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"graph:\n  groups: [example]\n  workspace_dir: {tmp_path / 'ws'}\n"
        f"database:\n  provider: ladybug\n  ladybug:\n    path: {path}\n"
    )
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setattr(
        "kg_mcp.runtime.build_graphiti", lambda *a, **k: _graph(path, read_only=True)
    )
    checks = {row["check"]: row for row in asyncio.run(check_retrieval("Aurora"))}
    assert checks["retrieval"]["status"] == "ok"
    assert checks["invalidation"] == {
        "check": "invalidation",
        "status": "ok",
        "detail": "1 superseded fact(s); none served",
    }
