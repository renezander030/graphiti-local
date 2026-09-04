"""The 0.3.0 contract: bounded calls, the embedder record, restore, and CLI ergonomics."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from kg_mcp.config import Settings

CONFIG = """
server:
  transport: stdio
graph:
  groups: [allowed]
  workspace_dir: {workspace}
  query_timeout_seconds: {timeout}
database:
  provider: {provider}
  ladybug:
    path: {workspace}/graph.ladybug
llm:
  model: test-model
embedder:
  model: {embedder}
  dimensions: 8
"""


@pytest.fixture()
def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def apply(*, provider: str = "falkordb", timeout: float = 30, embedder: str = "test-embedder"):
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            CONFIG.format(
                workspace=workspace, timeout=timeout, provider=provider, embedder=embedder
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
        monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace))
        return workspace

    return apply


def run_cli(*argv: str) -> None:
    import sys

    from kg_mcp.cli import main

    original = sys.argv
    sys.argv = ["kg", *argv]
    try:
        main()
    finally:
        sys.argv = original


class SlowGraph:
    """A backend that never answers."""

    driver = None

    async def search(self, *args, **kwargs):
        await asyncio.sleep(5)

    async def close(self):
        return None


# --- versions: the package knows which release it is ---


def test_package_version_matches_pyproject():
    from kg_mcp import __version__

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text("utf-8"), re.M)
    assert declared and __version__ == declared.group(1)


def test_version_flag_names_the_core_library(capsys):
    from kg_mcp import __version__

    with pytest.raises(SystemExit) as exit_info:
        run_cli("--version")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out and "graphiti-core" in out


def test_doctor_reports_installed_versions():
    from kg_mcp.doctor import check_versions

    result = check_versions()
    assert result["status"] == "ok"
    assert "graphiti-core" in result["detail"] and "python" in result["detail"]


# --- the reader flag is accepted on either side of the subcommand ---


def test_human_flag_after_the_subcommand(configure, capsys):
    configure()
    run_cli("propose", "allowed", "a fact", "-H")
    assert capsys.readouterr().out.startswith("proposed proposal-")


def test_human_flag_before_the_subcommand(configure, capsys):
    configure()
    run_cli("-H", "propose", "allowed", "a fact")
    assert capsys.readouterr().out.startswith("proposed proposal-")


def test_json_stays_the_default(configure, capsys):
    configure()
    run_cli("propose", "allowed", "a fact")
    assert json.loads(capsys.readouterr().out)["domain"] == "allowed"


# --- every graph call is bounded ---


def test_bounded_names_the_call_and_the_limit():
    from kg_mcp.runtime import bounded

    async def never():
        await asyncio.sleep(5)

    with pytest.raises(TimeoutError, match=r"ask timed out after 0\.01s"):
        asyncio.run(bounded(never(), 0.01, "ask"))


def test_query_timeout_default_and_bounds():
    assert Settings().graph.query_timeout_seconds == 30.0
    with pytest.raises(ValueError):
        Settings.model_validate({"graph": {"query_timeout_seconds": 0}})


def test_cli_ask_exits_3_when_the_backend_hangs(configure, monkeypatch, capsys):
    configure(timeout=0.05)
    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: SlowGraph())
    with pytest.raises(SystemExit) as exit_info:
        run_cli("ask", "anything")
    assert exit_info.value.code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "timed out after 0.05s" in json.loads(captured.err)["error"]


def test_mcp_tool_reports_a_timeout_instead_of_hanging(configure, monkeypatch):
    from kg_mcp.config import load_config
    from kg_mcp.server import create_server

    configure(timeout=0.05)
    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: SlowGraph())
    server = create_server(load_config())

    async def call():
        async with server._mcp_server.lifespan(server._mcp_server):
            return await server.call_tool("search_memory_facts", {"query": "anything"})

    result = asyncio.run(call())
    payload = result[1] if isinstance(result, tuple) else result
    if not isinstance(payload, dict):
        payload = json.loads(payload[0].text)
    assert "timed out after 0.05s" in payload["error"]


# --- the network transport can name the hosts it answers ---


def test_allowed_hosts_become_a_dns_rebinding_allow_list():
    from kg_mcp.server import transport_security

    settings = Settings.model_validate(
        {
            "server": {
                "transport": "streamable-http",
                "auth": {"token": "a-sufficiently-long-token"},
                "allowed_hosts": ["kg.internal:*"],
            }
        }
    )
    security = transport_security(settings)
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["kg.internal:*"]
    assert "https://kg.internal:*" in security.allowed_origins
    assert transport_security(Settings()) is None, "no list means the SDK default"


def test_server_carries_the_allow_list():
    from kg_mcp.server import create_server

    settings = Settings.model_validate(
        {
            "server": {
                "transport": "streamable-http",
                "auth": {"token": "a-sufficiently-long-token"},
                "allowed_hosts": ["kg.internal:*"],
            }
        }
    )
    assert create_server(settings).settings.transport_security.allowed_hosts == ["kg.internal:*"]


# --- the embedder that wrote the graph is recorded and enforced ---


def _with_embedder(settings: Settings, model: str) -> Settings:
    return settings.model_copy(
        update={"embedder": settings.embedder.model_copy(update={"model": model})}
    )


def test_embedder_record_detects_a_model_change(configure):
    from kg_mcp import fingerprint
    from kg_mcp.config import load_config

    configure()
    settings = load_config()
    assert fingerprint.check(settings)[0] == "warn", "nothing recorded before the first write"
    fingerprint.record(settings)
    assert fingerprint.drift(settings) is None
    assert fingerprint.check(settings)[0] == "ok"

    changed = _with_embedder(settings, "other-embedder")
    message = fingerprint.drift(changed)
    assert message and "produced by test-embedder" in message
    assert fingerprint.check(changed)[0] == "fail"


def test_embedder_record_is_per_database(configure):
    from kg_mcp import fingerprint
    from kg_mcp.config import load_config

    configure()
    settings = load_config()
    fingerprint.record(settings)
    other = settings.model_copy(
        update={"database": settings.database.model_copy(update={"provider": "neo4j"})}
    )
    assert fingerprint.recorded(other) is None


def test_ingest_refuses_an_embedder_change_before_touching_the_graph(configure, monkeypatch):
    from kg_mcp import fingerprint
    from kg_mcp.config import load_config
    from kg_mcp.ingest import ingest_records

    configure()
    fingerprint.record(_with_embedder(load_config(), "other-embedder"))

    def explode(*args, **kwargs):
        raise AssertionError("the graph must not be opened for a refused write")

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", explode)
    records = [{"name": "x", "body": "y", "domain": "allowed"}]
    with pytest.raises(ValueError, match="produced by other-embedder"):
        asyncio.run(ingest_records(records, apply=True))


def test_ingest_records_the_embedder_after_a_write(configure, monkeypatch):
    from kg_mcp import fingerprint
    from kg_mcp.config import load_config
    from kg_mcp.ingest import ingest_records

    configure()

    class FakeGraph:
        driver = None

        async def build_indices_and_constraints(self):
            return None

        async def add_episode(self, **kwargs):
            return None

        async def close(self):
            return None

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: FakeGraph())
    records = [{"name": "x", "body": "y", "domain": "allowed"}]
    result = asyncio.run(ingest_records(records, apply=True, ledger=False))
    assert result["ingested"] == 1
    assert fingerprint.recorded(load_config())["model"] == "test-embedder"


def test_doctor_has_an_embedder_record_check(configure):
    from kg_mcp.config import load_config
    from kg_mcp.doctor import check_embedder_fingerprint

    configure()
    result = check_embedder_fingerprint(load_config())
    assert result["check"] == "embedder-record" and result["status"] == "warn"


# --- restore: the other half of export ---

SNAPSHOT_HEADER = {
    "kind": "export",
    "format_version": 1,
    "created_at": "2026-09-04T00:00:00+00:00",
    "provider": "ladybug",
    "groups": ["example"],
}
SNAPSHOT_RECORDS = [
    {
        "kind": "entity_node",
        "uuid": "n1",
        "name": "Ada Lovelace",
        "group_id": "",
        "labels": ["Entity"],
        "created_at": "2026-09-04T00:00:00+00:00",
        "summary": "mathematician",
        "attributes": {},
    },
    {
        "kind": "episodic_node",
        "uuid": "e1",
        "name": "Ada",
        "group_id": "",
        "source": "text",
        "source_description": "test",
        "content": "Ada Lovelace wrote the first computer program.",
        "valid_at": "2026-09-04T00:00:00+00:00",
        "created_at": "2026-09-04T00:00:00+00:00",
        "entity_edges": ["r1"],
    },
    {
        "kind": "entity_edge",
        "uuid": "r1",
        "group_id": "",
        "source_node_uuid": "n1",
        "target_node_uuid": "n1",
        "name": "WROTE",
        "fact": "Ada Lovelace wrote the first computer program",
        "created_at": "2026-09-04T00:00:00+00:00",
        "episodes": ["e1"],
        "fact_embedding": [0.1, 0.2],
    },
    {
        "kind": "episodic_edge",
        "uuid": "m1",
        "group_id": "",
        "source_node_uuid": "e1",
        "target_node_uuid": "n1",
        "created_at": "2026-09-04T00:00:00+00:00",
    },
]


def _write_snapshot(path: Path, records=SNAPSHOT_RECORDS, header=SNAPSHOT_HEADER) -> Path:
    lines = [json.dumps(header), *(json.dumps(record) for record in records)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_records_points_a_snapshot_at_restore(tmp_path: Path):
    from kg_mcp.ingest import read_records

    with pytest.raises(SystemExit, match="--restore"):
        read_records(_write_snapshot(tmp_path / "snapshot.jsonl"))


def test_read_snapshot_requires_the_export_header(tmp_path: Path):
    from kg_mcp.restore import read_snapshot

    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"name":"a","body":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a kg export snapshot"):
        read_snapshot(plain)
    outdated = {**SNAPSHOT_HEADER, "format_version": 99}
    with pytest.raises(ValueError, match="format 99"):
        read_snapshot(_write_snapshot(tmp_path / "v99.jsonl", header=outdated))
    header, records = read_snapshot(_write_snapshot(tmp_path / "ok.jsonl"))
    assert header["provider"] == "ladybug" and len(records) == 4


def test_build_model_rebuilds_every_kind_without_embeddings():
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    from kg_mcp.restore import build_model

    models = [build_model(record, "target") for record in SNAPSHOT_RECORDS]
    expected = [EntityNode, EpisodicNode, EntityEdge, EpisodicEdge]
    assert [type(model) for model in models] == expected
    assert all(model.group_id == "target" for model in models)
    assert models[1].source == EpisodeType.text
    assert models[2].fact_embedding is None, "vectors are re-created, never restored"


def test_restore_dry_run_on_a_server_backend_needs_a_group(configure, tmp_path: Path):
    from kg_mcp.restore import restore_snapshot

    configure(provider="falkordb")
    snapshot = _write_snapshot(tmp_path / "snapshot.jsonl")
    with pytest.raises(ValueError, match="--group"):
        asyncio.run(restore_snapshot(snapshot, apply=False))
    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(restore_snapshot(snapshot, apply=False, group="nope"))
    result = asyncio.run(restore_snapshot(snapshot, apply=False, group="allowed"))
    assert result["applied"] is False
    assert result["groups"] == ["allowed"]
    assert result["planned"] == {
        "entity_node": 1,
        "episodic_node": 1,
        "entity_edge": 1,
        "episodic_edge": 1,
    }


def test_restore_dry_run_on_ladybug_targets_the_embedded_graph(configure, tmp_path: Path):
    from kg_mcp.restore import restore_snapshot

    configure(provider="ladybug")
    result = asyncio.run(restore_snapshot(_write_snapshot(tmp_path / "s.jsonl"), apply=False))
    assert result["groups"] == [""]


def test_restore_refuses_an_embedder_change(configure, tmp_path: Path, monkeypatch):
    from kg_mcp import fingerprint
    from kg_mcp.config import load_config
    from kg_mcp.restore import restore_snapshot

    configure(provider="ladybug")
    fingerprint.record(_with_embedder(load_config(), "other-embedder"))
    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: SlowGraph())
    with pytest.raises(ValueError, match="produced by other-embedder"):
        asyncio.run(restore_snapshot(_write_snapshot(tmp_path / "s.jsonl"), apply=True))


# --- duplicates: names that only differ in casing or punctuation ---


def test_duplicates_cluster_by_normalized_name():
    from kg_mcp.duplicates import cluster, normalized_name

    class Node:
        def __init__(self, uuid, name, summary=""):
            self.uuid, self.name, self.group_id, self.summary = uuid, name, "g", summary

    assert normalized_name("  ACME Corp. ") == "acme corp"
    clusters = cluster([Node("1", "ACME Corp"), Node("2", "acme corp."), Node("3", "Other")])
    assert len(clusters) == 1
    assert clusters[0]["name"] == "acme corp" and clusters[0]["count"] == 2
    assert {member["uuid"] for member in clusters[0]["members"]} == {"1", "2"}


def test_exit_codes_are_distinct():
    from kg_mcp.output import EXIT_ERROR, EXIT_REJECTED, EXIT_TIMEOUT

    assert len({EXIT_ERROR, EXIT_REJECTED, EXIT_TIMEOUT}) == 3 and EXIT_TIMEOUT == 3
