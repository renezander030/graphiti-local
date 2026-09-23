"""0.5: one Ladybug file per group, and the library helpers around an embedded graph.

The isolation tests run against real Ladybug files and need the FTS and VECTOR
extensions on the host; they skip otherwise, like the rest of the Ladybug suite. No
model or network is contacted: embeddings come from a deterministic fake.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kg_mcp.config import Settings

HOSTILE_IDS = [
    "../escape",
    "../../etc/passwd",
    "/etc/passwd",
    "C:\\Windows\\System32",
    "a/../../b",
    "nul\x00byte",
    "x" * 5000,
    ".",
    "..",
    "~root",
    "spaces and\nnewlines",
    "\udcff lone surrogate",
]


def _extensions_available() -> bool:
    from kg_mcp.ladybug import extension_status

    try:
        return extension_status()["ok"]
    except Exception:
        return False


needs_extensions = pytest.mark.skipif(
    not _extensions_available(), reason="Ladybug search extensions are not installed"
)


def _use(settings: Settings, tmp_path: Path, monkeypatch) -> Settings:
    """Make ``settings`` the loaded configuration, as the CLI and the workspace read it."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings.model_dump()), encoding="utf-8")
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(path))
    monkeypatch.delenv("KG_WORKSPACE_DIR", raising=False)
    return settings


def _settings(tmp_path: Path, layout: str = "per-group", **server) -> Settings:
    return Settings.model_validate(
        {
            "server": server,
            "graph": {"groups": ["alice", "bob"], "workspace_dir": str(tmp_path / "ws")},
            "database": {
                "provider": "ladybug",
                "ladybug": {
                    "path": str(tmp_path / "graph.ladybug"),
                    "layout": layout,
                    "directory": str(tmp_path / "groups"),
                },
            },
            "embedder": {"model": "nomic-embed-text", "dimensions": 768},
        }
    )


# --- file layout -------------------------------------------------------------------


def test_single_layout_is_the_default_and_unchanged(tmp_path: Path):
    from kg_mcp.config import per_group_ladybug, settings_for_group

    settings = Settings.model_validate(
        {"database": {"provider": "ladybug", "ladybug": {"path": "/data/graph.ladybug"}}}
    )
    assert settings.database.ladybug.layout == "single"
    assert not per_group_ladybug(settings)
    assert settings.database.ladybug.path_for("anyone") == "/data/graph.ladybug"
    assert settings_for_group(settings, "main") is settings


def test_each_group_gets_its_own_file_named_by_sha256(tmp_path: Path):
    import hashlib

    settings = _settings(tmp_path)
    alice = Path(settings.database.ladybug.path_for("alice"))
    bob = Path(settings.database.ladybug.path_for("bob"))
    assert alice != bob
    assert alice.parent == bob.parent == (tmp_path / "groups").resolve()
    assert alice.name == hashlib.sha256(b"alice").hexdigest() + ".ladybug"
    with pytest.raises(ValueError, match="needs a group"):
        settings.database.ladybug.path_for(None)


@pytest.mark.parametrize("hostile", HOSTILE_IDS)
def test_hostile_ids_stay_inside_the_directory(tmp_path: Path, hostile: str):
    from kg_mcp.ladybug import group_database_path

    root = (tmp_path / "groups").resolve()
    target = group_database_path(root, hostile)
    assert target.parent == root
    assert len(target.stem) == 64 and all(c in "0123456789abcdef" for c in target.stem)


def test_an_empty_group_is_refused(tmp_path: Path):
    from kg_mcp.ladybug import group_database_path

    with pytest.raises(ValueError, match="non-empty"):
        group_database_path(tmp_path, "")


def test_binding_narrows_settings_to_one_group_file(tmp_path: Path):
    from kg_mcp.config import per_group_ladybug, settings_for_group

    settings = _settings(tmp_path)
    bound = settings_for_group(settings, "alice")
    assert bound.graph.groups == ["alice"]
    assert bound.database.ladybug.path == settings.database.ladybug.path_for("alice")
    assert not per_group_ladybug(bound)
    with pytest.raises(ValueError, match="not allowed"):
        settings_for_group(settings, "mallory")


def test_per_group_layout_accepts_a_partial_token_scope(tmp_path: Path):
    grants = {"auth": {"tokens": [{"token": "alice-token-0123456", "groups": ["alice"]}]}}
    settings = _settings(tmp_path, transport="streamable-http", **grants)
    assert settings.server.auth.grants()[0].groups == ["alice"]
    with pytest.raises(ValueError, match="per-group"):
        _settings(tmp_path, layout="single", transport="streamable-http", **grants)


# --- isolation on real files -------------------------------------------------------


def _fake_embedder_class():
    from graphiti_core.embedder.client import EmbedderClient

    class FakeEmbedder(EmbedderClient):
        def __init__(self, config=None):
            del config

        async def create(self, input_data):
            return [0.5] * 8

        async def create_batch(self, input_data_list):
            return [[0.5] * 8 for _ in input_data_list]

    return FakeEmbedder


def _write_fact(path: str, fact: str) -> None:
    """Write one fact straight into a file, as a drain would leave it."""
    import ladybug

    from kg_mcp.ladybug import build_ladybug_driver, load_extensions

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(build_ladybug_driver(path, read_only=False).close())
    database = ladybug.Database(path)
    connection = ladybug.Connection(database)
    assert load_extensions(connection) == []
    now = datetime.now(timezone.utc)
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    for node in (a, b):
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: $n, group_id: '', created_at: $t, "
            "name_embedding: $v, summary: '', labels: ['Entity']})",
            {"u": node, "n": f"node {node[:4]}", "t": now, "v": [0.5] * 8},
        )
    connection.execute(
        "MATCH (x:Entity {uuid: $a}), (y:Entity {uuid: $b}) "
        "CREATE (x)-[:RELATES_TO]->(:RelatesToNode_ {uuid: $e, name: 'NOTE', fact: $f, "
        "group_id: '', created_at: $t, fact_embedding: $v, episodes: []})-[:RELATES_TO]->(y)",
        {"a": a, "b": b, "e": str(uuid.uuid4()), "f": fact, "t": now, "v": [0.5] * 8},
    )
    connection.execute("CHECKPOINT")
    connection.close()
    database.close()


def _raw_facts(path: str) -> list[str]:
    import ladybug

    database = ladybug.Database(path, read_only=True)
    connection = ladybug.Connection(database)
    try:
        rows = connection.execute("MATCH (e:RelatesToNode_) RETURN e.fact AS fact")
        return sorted(row["fact"] for row in rows.rows_as_dict())
    finally:
        connection.close()
        database.close()


@pytest.fixture
def two_users(tmp_path: Path, monkeypatch):
    os.environ.setdefault("OPENAI_API_KEY", "test-only")
    import graphiti_core.embedder.openai as openai_embedder

    monkeypatch.setattr(openai_embedder, "OpenAIEmbedder", _fake_embedder_class())
    settings = _use(_settings(tmp_path), tmp_path, monkeypatch)
    _write_fact(settings.database.ladybug.path_for("alice"), "Alice keeps the invoice ledger")
    _write_fact(settings.database.ladybug.path_for("bob"), "Bob keeps the invoice archive")
    return settings


async def _call(server, name: str, arguments: dict):
    return await server._tool_manager.call_tool(name, arguments)


@needs_extensions
def test_two_users_two_files_each_holding_only_its_own_facts(two_users: Settings):
    ladybug = two_users.database.ladybug
    assert _raw_facts(ladybug.path_for("alice")) == ["Alice keeps the invoice ledger"]
    assert _raw_facts(ladybug.path_for("bob")) == ["Bob keeps the invoice archive"]
    assert not Path(ladybug.path).exists()  # the single-layout file is never created


@needs_extensions
def test_a_token_for_one_user_reads_only_that_users_file(two_users: Settings):
    from kg_mcp.server import _request_group_scope, create_server

    server = create_server(two_users)

    async def scenario():
        marker = _request_group_scope.set(frozenset({"alice"}))
        try:
            async with server.settings.lifespan(server) as state:
                facts = await _call(server, "search_memory_facts", {"query": "invoice"})
                denied = await _call(
                    server, "search_memory_facts", {"query": "invoice", "group_ids": ["bob"]}
                )
                status = await _call(server, "get_status", {})
                opened = sorted(state["graphs"])
            return facts, denied, status, opened
        finally:
            _request_group_scope.reset(marker)

    facts, denied, status, opened = asyncio.run(scenario())
    assert [item["fact"] for item in facts["facts"]] == ["Alice keeps the invoice ledger"]
    assert "not granted" in denied["error"]
    assert status["status"] == "ok" and status["groups"] == ["alice"]
    assert opened == ["alice"]  # bob's file was never opened


@needs_extensions
def test_an_unscoped_reader_must_name_one_group(two_users: Settings):
    from kg_mcp.server import create_server

    server = create_server(two_users)

    async def scenario():
        async with server.settings.lifespan(server):
            ambiguous = await _call(server, "search_memory_facts", {"query": "invoice"})
            bob = await _call(
                server, "search_memory_facts", {"query": "invoice", "group_ids": "bob"}
            )
        return ambiguous, bob

    ambiguous, bob = asyncio.run(scenario())
    assert "one group per call" in ambiguous["error"]
    assert [item["fact"] for item in bob["facts"]] == ["Bob keeps the invoice archive"]


@needs_extensions
def test_kg_ask_reads_the_named_users_file(two_users: Settings):
    from kg_mcp import cli

    args = cli.parser().parse_args(["ask", "invoice", "bob", "--keyword"])
    payload = asyncio.run(cli._dispatch(args))
    assert payload["groups"] == ["bob"]
    assert [item["fact"] for item in payload["facts"]] == ["Bob keeps the invoice archive"]

    ambiguous = cli.parser().parse_args(["ask", "invoice", "--keyword"])
    with pytest.raises(ValueError, match="one group per call"):
        asyncio.run(cli._dispatch(ambiguous))


def test_ingest_routes_each_domain_to_its_own_file(tmp_path: Path, monkeypatch):
    from kg_mcp import ingest, runtime

    settings = _use(_settings(tmp_path), tmp_path, monkeypatch)
    written: dict[str, list[str]] = {}

    class FakeGraph:
        def __init__(self, bound):
            self.path = bound.database.ladybug.path

        async def build_indices_and_constraints(self):
            return None

        async def add_episode(self, *, name, **kwargs):
            written.setdefault(self.path, []).append(name)

        async def close(self):
            return None

    monkeypatch.setattr(runtime, "build_graphiti", lambda bound, read_only: FakeGraph(bound))
    records = [
        {"name": "a1", "body": "Alice fact", "domain": "alice"},
        {"name": "b1", "body": "Bob fact", "domain": "bob"},
        {"name": "a2", "body": "Alice again", "domain": "alice"},
    ]
    result = asyncio.run(
        ingest.ingest_records(records, apply=True, ledger=False, settings=settings)
    )
    assert result["ingested"] == 3 and result["failed"] == []
    alice = settings.database.ladybug.path_for("alice")
    bob = settings.database.ladybug.path_for("bob")
    assert written == {alice: ["a1", "a2"], bob: ["b1"]}

    from kg_mcp.fingerprint import recorded_embedder

    assert recorded_embedder(alice)["model"] == "nomic-embed-text"
    assert recorded_embedder(bob)["dimensions"] == 768


# --- library helpers ---------------------------------------------------------------


def test_embedder_record_is_keyed_by_database_path(tmp_path: Path):
    from kg_mcp.fingerprint import embedder_drift, record_embedder, recorded_embedder

    path = tmp_path / "user.ladybug"
    assert recorded_embedder(path) is None
    assert embedder_drift(path, model="nomic-embed-text", dimensions=768) is None
    entry = record_embedder(path, model="nomic-embed-text:latest", dimensions=768)
    assert entry["model"] == "nomic-embed-text"
    assert (tmp_path / "user.ladybug.embedder.json").is_file()
    assert recorded_embedder(path)["dimensions"] == 768
    assert embedder_drift(path, model="nomic-embed-text:latest", dimensions=768) is None
    message = embedder_drift(path, model="mxbai-embed-large", dimensions=1024)
    assert message and "mixing embedders" in message
    other = tmp_path / "other.ladybug"
    assert recorded_embedder(other) is None


def test_configured_drift_check_sees_a_record_written_as_a_library(tmp_path: Path, monkeypatch):
    from kg_mcp import fingerprint
    from kg_mcp.config import settings_for_group

    _use(_settings(tmp_path), tmp_path, monkeypatch)
    bound = settings_for_group(_settings(tmp_path), "alice")
    fingerprint.record_embedder(
        bound.database.ladybug.path, model="mxbai-embed-large", dimensions=1024
    )
    assert "mixing embedders" in fingerprint.drift(bound)


def test_keyword_search_config_makes_no_model_call():
    from graphiti_core.search.search_config import EdgeSearchMethod, NodeSearchMethod

    from kg_mcp.retrieval import keyword_edge_search_config, keyword_node_search_config

    edges = keyword_edge_search_config(7)
    assert edges.limit == 7
    assert edges.edge_config.search_methods == [EdgeSearchMethod.bm25]
    assert edges.node_config is None
    nodes = keyword_node_search_config()
    assert nodes.limit == 10
    assert nodes.node_config.search_methods == [NodeSearchMethod.bm25]
    assert nodes.edge_config is None


def test_extension_status_names_the_fix_when_something_is_missing(tmp_path: Path, monkeypatch):
    import ladybug

    from kg_mcp import ladybug as adapter

    status = adapter.extension_status()
    assert status["engine_version"] == ladybug.__version__
    assert set(status["installed"]) | set(status["missing"]) == {"FTS", "VECTOR"}
    assert status["ok"] is (not status["missing"])

    monkeypatch.setattr(adapter, "load_extensions", lambda connection: ["VECTOR"])
    target = tmp_path / "user.ladybug"
    missing = adapter.extension_status(target)
    assert missing["ok"] is False
    assert missing["missing"] == ["VECTOR"] and missing["installed"] == ["FTS"]
    assert missing["fix"] == f"kg-ladybug-setup --database {target.resolve()} --apply"


@needs_extensions
def test_export_and_restore_move_facts_between_group_files(two_users: Settings, tmp_path: Path):
    from kg_mcp import cli
    from kg_mcp.restore import restore_snapshot

    snapshot = tmp_path / "alice.jsonl"
    args = cli.parser().parse_args(["export", "alice", "--output", str(snapshot)])
    exported = asyncio.run(cli._dispatch(args))
    assert exported["edges"] == 1

    with pytest.raises(ValueError, match="pass --group"):
        asyncio.run(restore_snapshot(snapshot, apply=False))
    result = asyncio.run(restore_snapshot(snapshot, apply=True, group="bob"))
    assert result["groups"] == ["bob"] and result["failed"] == []
    ladybug = two_users.database.ladybug
    assert _raw_facts(ladybug.path_for("bob")) == [
        "Alice keeps the invoice ledger",
        "Bob keeps the invoice archive",
    ]
    assert _raw_facts(ladybug.path_for("alice")) == ["Alice keeps the invoice ledger"]
