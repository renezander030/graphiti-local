"""0.8.0: precise retirement, a queue that survives any text, and a stricter MCP surface."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kg_mcp.output import EXIT_REJECTED, EXIT_SECRET_SHAPED, CommandError


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str = "") -> Path:
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"graph:\n  groups: [example]\n  workspace_dir: {workspace}\n{extra}",
        encoding="utf-8",
    )
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace))
    return workspace


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _configure(tmp_path, monkeypatch)


def _approve_all(workspace) -> None:
    workspace._human_set_status({item["id"] for item in workspace.pending_for()}, "approved")


# --- retire the exact facts a reviewer names ---


def test_retire_files_one_reviewable_item_for_the_named_edges(configured):
    from kg_mcp import workspace

    item = workspace.retire_facts(
        "example",
        ["edge-a", "edge-b", "edge-a", " "],
        reason="The pilot ended; Aurora no longer uses DuckDB.",
        source="meeting notes",
        link="file:notes/2026-10-01.md",
        valid_at="2026-10-01",
    )
    assert item["operation"] == "retire"
    assert item["edges"] == ["edge-a", "edge-b"]
    assert item["fact_type"] == "action-record"
    assert item["valid_at"].startswith("2026-10-01")
    assert [entry["id"] for entry in workspace.pending_for()] == [item["id"]]


@pytest.mark.parametrize(
    ("domain", "edges", "reason", "code"),
    [
        ("other", ["edge-a"], "gone", EXIT_REJECTED),
        ("example", [], "gone", EXIT_REJECTED),
        ("example", ["edge-a"], "  ", EXIT_REJECTED),
        ("example", ["edge-a"], "key sk-ant-" + "a" * 30, EXIT_SECRET_SHAPED),
    ],
)
def test_retire_refusals_carry_their_exit_code(configured, domain, edges, reason, code):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.retire_facts(domain, edges, reason=reason)
    assert refused.value.code == code
    assert workspace.pending_for() == []


def test_an_approved_retirement_is_bound_to_its_edges(configured):
    from kg_mcp import workspace

    item = workspace.retire_facts("example", ["edge-a"], reason="superseded")
    _approve_all(workspace)
    queued = workspace._read()
    queued[0]["edges"] = ["edge-b"]
    workspace._write("pending.jsonl", queued)
    planned = asyncio.run(workspace.drain(apply=False))["planned"]
    assert planned[0]["id"] == item["id"]
    assert "approve it again" in planned[0]["binding"]


def test_a_dry_run_plans_the_retirement(configured):
    from kg_mcp import workspace

    item = workspace.retire_facts("example", ["edge-a"], reason="superseded", valid_at="2026-10-01")
    _approve_all(workspace)
    assert asyncio.run(workspace.drain(apply=False))["planned"] == [
        {
            "id": item["id"],
            "domain": "example",
            "body": "retire edge-a: superseded",
            "valid_at": item["valid_at"],
            "binding": "ok",
            "decided_by": "reviewer",
        }
    ]


def test_a_dry_run_lists_retirements_and_facts_in_queue_order(configured):
    from kg_mcp import workspace

    first = workspace.retire_facts("example", ["edge-a"], reason="superseded")
    second = workspace.add_proposal("example", "Aurora ships weekly.")
    _approve_all(workspace)
    planned = asyncio.run(workspace.drain(apply=False))["planned"]
    assert [(entry["id"], entry["body"]) for entry in planned] == [
        (first["id"], "retire edge-a: superseded"),
        (second["id"], "Aurora ships weekly."),
    ]


def test_kg_retire_prints_the_proposal(configured, monkeypatch, capsys):
    from kg_mcp import cli

    monkeypatch.setattr(
        sys, "argv", ["kg", "retire", "example", "--edge", "edge-a", "--reason", "superseded"]
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "retire"
    assert payload["edges"] == ["edge-a"]
    assert payload["status"] == "pending"


def _ladybug_ready() -> bool:
    try:
        import ladybug

        from kg_mcp.ladybug import load_extensions

        connection = ladybug.Connection(ladybug.Database(":memory:"))
        return load_extensions(connection) == []
    except Exception:
        return False


ladybug_only = pytest.mark.skipif(
    not _ladybug_ready(), reason="Ladybug search extensions are not installed"
)


def _ladybug_with_edge(path: Path, fact: str) -> str:
    """One fact edge between two entities, written the way the embedded graph stores it."""
    import ladybug

    from kg_mcp.ladybug import build_ladybug_driver, load_extensions

    asyncio.run(build_ladybug_driver(str(path), read_only=False).close())
    database = ladybug.Database(str(path))
    connection = ladybug.Connection(database)
    load_extensions(connection)
    now = datetime.now(timezone.utc)
    a, b, edge = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    for node in (a, b):
        connection.execute(
            "CREATE (:Entity {uuid: $u, name: $u, group_id: '', created_at: $t, "
            "name_embedding: $v, summary: '', labels: ['Entity']})",
            {"u": node, "t": now, "v": [0.5] * 8},
        )
    connection.execute(
        "MATCH (x:Entity {uuid: $a}), (y:Entity {uuid: $b}) "
        "CREATE (x)-[:RELATES_TO]->(:RelatesToNode_ {uuid: $e, name: 'USES', fact: $f, "
        "group_id: '', created_at: $t, fact_embedding: $v, episodes: []})-[:RELATES_TO]->(y)",
        {"a": a, "b": b, "e": edge, "f": fact, "t": now, "v": [0.5] * 8},
    )
    connection.execute("CHECKPOINT")
    connection.close()
    database.close()
    return edge


def _ladybug_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "graph.ladybug"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    _configure(
        tmp_path,
        monkeypatch,
        f"database:\n  provider: ladybug\n  ladybug:\n    path: {path}\n"
        "embedder:\n  model: test-embedder\n  dimensions: 8\n",
    )
    return path


def _read_edge(path: Path, edge: str):
    from graphiti_core.edges import EntityEdge

    from kg_mcp.ladybug import build_ladybug_driver

    driver = build_ladybug_driver(str(path), read_only=True)
    try:
        found = asyncio.run(EntityEdge.get_by_uuids(driver, [edge]))
        if found:
            asyncio.run(found[0].load_fact_embedding(driver))
        return found[0] if found else None
    finally:
        asyncio.run(driver.close())


@ladybug_only
def test_drain_retires_exactly_the_approved_edges_on_ladybug(tmp_path, monkeypatch):
    from kg_mcp import workspace

    path = _ladybug_config(tmp_path, monkeypatch)
    edge = _ladybug_with_edge(path, "Aurora uses DuckDB")
    kept = _ladybug_with_edge(path, "Aurora runs in Frankfurt")
    item = workspace.retire_facts("example", [edge], reason="pilot ended", valid_at="2026-10-01")
    _approve_all(workspace)

    result = asyncio.run(workspace.drain(apply=True))
    assert (result["ingested"], result["failed"]) == (1, [])
    retired = _read_edge(path, edge)
    assert retired.invalid_at.isoformat().startswith("2026-10-01")
    assert retired.expired_at is not None
    assert retired.fact_embedding == [0.5] * 8, "the vector survives the rewrite"
    assert _read_edge(path, kept).invalid_at is None
    archived = workspace._read("archive.jsonl")[0]
    assert (archived["id"], archived["retired"]) == (item["id"], [edge])
    assert workspace._read() == []


@ladybug_only
def test_a_retirement_naming_a_missing_edge_retires_nothing(tmp_path, monkeypatch):
    from kg_mcp import workspace

    path = _ladybug_config(tmp_path, monkeypatch)
    edge = _ladybug_with_edge(path, "Aurora uses DuckDB")
    workspace.retire_facts("example", [edge, "no-such-edge"], reason="pilot ended")
    _approve_all(workspace)

    result = asyncio.run(workspace.drain(apply=True))
    assert result["ingested"] == 0
    assert "no-such-edge" in result["failed"][0]["error"]
    assert _read_edge(path, edge).invalid_at is None
    assert workspace._read()[0]["status"] == "approved", "kept for a corrected retry"



# --- the queue survives any fact text ---


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
def test_a_fact_with_a_unicode_line_separator_keeps_the_queue_readable(configured, separator):
    from kg_mcp import ingest, workspace

    first = workspace.add_proposal("example", f"Aurora uses Rust{separator} for its pipeline.")
    second = workspace.add_proposal("example", "Aurora ships weekly.")
    assert [item["id"] for item in workspace.pending_for()] == [first["id"], second["id"]]
    assert workspace.pending_for()[0]["text"] == first["text"]

    workspace.append_locked("ingest-ledger.jsonl", {"key": "k1", "name": f"a{separator}b"})
    workspace.append_locked("ingest-ledger.jsonl", {"key": "k2", "name": "c"})
    assert ingest.completed_keys() == {"k1", "k2"}


def test_kg_workspace_reports_an_unreadable_queue_as_an_error(configured, monkeypatch, capsys):
    from kg_mcp import workspace

    configured.mkdir(parents=True, exist_ok=True)
    (configured / "pending.jsonl").write_text("{not json\n", encoding="utf-8")
    for argv in (["list"], ["rejected"], ["approve", "proposal-x"], ["reject", "proposal-x"]):
        monkeypatch.setattr(sys, "argv", ["kg-workspace", *argv])
        with pytest.raises(SystemExit) as exited:
            workspace.main()
        assert exited.value.code == 1
        assert "corrupt queue JSON" in json.loads(capsys.readouterr().err)["error"]


# --- the MCP surface ---


async def test_every_mcp_tool_declares_it_only_reads():
    from kg_mcp.server import create_server

    for tool in await create_server().list_tools():
        hints = tool.annotations
        assert hints is not None, tool.name
        assert (hints.readOnlyHint, hints.destructiveHint, hints.idempotentHint) == (
            True,
            False,
            True,
        )
        assert hints.openWorldHint is False


async def test_an_unknown_mcp_argument_is_refused_by_name(mcp_server):
    async with _client(mcp_server) as client:
        refused = await client.call_tool("search_nodes", {"query": "Aurora", "group_id": "x"})
    assert refused.isError
    message = refused.content[0].text
    assert "group_id" in message
    assert "group_ids" in message, "the refusal names what the tool accepts"


@pytest.fixture
def mcp_server(configured, monkeypatch):
    """An initialized server whose graph answers every read with one canned record."""
    from types import SimpleNamespace

    from kg_mcp import queries
    from kg_mcp.server import create_server

    graph = SimpleNamespace(
        driver=SimpleNamespace(),
        close=lambda: asyncio.sleep(0),
        retrieve_episodes=lambda **_: asyncio.sleep(0, result=[]),
    )
    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *_a, **_k: graph)

    async def no_nodes(*_a, **_k):
        return [], {"returned": 0}

    async def no_facts(*_a, **_k):
        return [], {"returned": 0, "suppressed": 0}

    monkeypatch.setattr(queries, "search_nodes", no_nodes)
    monkeypatch.setattr(queries, "search_facts", no_facts)
    return create_server()


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_nodes", {"query": "Aurora"}),
        ("search_memory_facts", {"query": "Aurora"}),
        ("get_episodes", {}),
    ],
)
async def test_an_mcp_read_reports_the_groups_it_searched(mcp_server, tool, arguments):
    async with _client(mcp_server) as client:
        result = await client.call_tool(tool, arguments)
    assert not result.isError
    assert result.structuredContent["groups"] == ["example"]


def _client(server):
    """A client session connected to the server in memory, as an MCP client sees it."""
    from mcp.shared.memory import create_connected_server_and_client_session

    return create_connected_server_and_client_session(server)


# --- the opt-in OpenAI reranker keeps working after gpt-4.1-nano is retired ---


def test_the_openai_reranker_defaults_to_a_model_that_is_still_served(configured, monkeypatch):
    from kg_mcp.config import load_config
    from kg_mcp.runtime import OPENAI_RERANKER_MODEL, build_reranker

    settings = load_config().model_copy(deep=True)
    settings.reranker.provider, settings.reranker.api_key = "openai", "test-only"
    assert build_reranker(settings).config.model == OPENAI_RERANKER_MODEL == "gpt-4.1-mini"
    settings.reranker.model = "my-model"
    assert build_reranker(settings).config.model == "my-model"


# --- kg doctor checks the workspace the commands use ---


def test_doctor_checks_the_workspace_named_by_kg_workspace_dir(tmp_path, monkeypatch):
    from kg_mcp.config import load_config
    from kg_mcp.doctor import check_workspace

    _configure(tmp_path, monkeypatch)
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(override))
    result = check_workspace(load_config())
    assert result["status"] == "ok"
    assert str(override.resolve()) in result["detail"]


# --- repeated copies of one fact ---


def test_fact_duplicates_cluster_current_copies_between_the_same_entities():
    from types import SimpleNamespace

    from kg_mcp.duplicates import cluster_facts

    def edge(uuid_, fact, source="a", target="b", invalid_at=None, day=1):
        return SimpleNamespace(
            uuid=uuid_,
            fact=fact,
            source_node_uuid=source,
            target_node_uuid=target,
            invalid_at=invalid_at,
            group_id="example",
            created_at=datetime(2026, 10, day, tzinfo=timezone.utc),
        )

    past = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clusters = cluster_facts(
        [
            edge("e2", "Aurora uses DuckDB.", day=2),
            edge("e1", "aurora uses duckdb", day=1),
            edge("e3", "Aurora uses DuckDB", source="c"),
            edge("e4", "Aurora uses DuckDB", invalid_at=past),
            edge("e5", "Aurora runs in Frankfurt"),
        ]
    )
    assert [(item["count"], [m["uuid"] for m in item["members"]]) for item in clusters] == [
        (2, ["e1", "e2"])
    ]
