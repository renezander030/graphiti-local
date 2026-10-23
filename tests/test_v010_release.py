"""0.10.0: one drain at a time, retractable proposals, and stricter reads and gates."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from kg_mcp.output import EXIT_REJECTED, EXIT_SECRET_SHAPED, CommandError


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
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _configure(tmp_path, monkeypatch)


class Ingest:
    def __init__(self, edges: int = 1) -> None:
        self.records: list[dict] = []
        self.edges = edges

    async def __call__(self, records, **kwargs):
        self.records.extend(records)
        receipt = {"episode_uuid": f"episode-{len(self.records)}", "edges": self.edges}
        return {"applied": True, "ingested": 1, "failed": [], "receipts": [receipt]}


def _approved(workspace, text: str = "Aurora ships weekly.") -> dict:
    item = workspace.add_proposal("example", text)
    workspace._human_set_status({item["id"]}, "approved")
    return item


def test_release_metadata_is_0_10_0():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "server.json").read_text(encoding="utf-8"))
    assert manifest["version"] == manifest["packages"][0]["version"] == "0.10.0"
    assert 'version = "0.10.0"' in (root / "pyproject.toml").read_text(encoding="utf-8")


# --- one drain at a time ---


def test_a_second_drain_stands_down_while_one_runs(configured, monkeypatch):
    from kg_mcp import workspace

    _approved(workspace)
    ingest = Ingest()
    seen = {}

    async def drain_inside_a_drain(records, **kwargs):
        seen["second"] = await workspace.drain(apply=True)
        return await ingest(records, **kwargs)

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", drain_inside_a_drain)
    first = asyncio.run(workspace.drain(apply=True))
    assert first["ingested"] == 1 and len(ingest.records) == 1
    assert seen["second"] == {
        "applied": False,
        "ingested": 0,
        "failed": [],
        "planned": [],
        "busy": "another drain is running",
    }


def test_kg_workspace_drain_reports_a_busy_drain_and_exits_zero(configured, monkeypatch, capsys):
    from kg_mcp import workspace

    _approved(workspace)
    lock = workspace._drain_lock()
    lock.acquire()
    try:
        monkeypatch.setattr(sys, "argv", ["kg-workspace", "-H", "drain", "--apply"])
        workspace.main()
    finally:
        lock.release()
    assert capsys.readouterr().out.strip() == "another drain is running"
    assert workspace._read()[0]["status"] == "approved"


def test_a_dry_run_does_not_wait_for_a_running_drain(configured):
    from kg_mcp import workspace

    _approved(workspace)
    lock = workspace._drain_lock()
    lock.acquire()
    try:
        planned = asyncio.run(workspace.drain(apply=False))["planned"]
    finally:
        lock.release()
    assert len(planned) == 1


# --- a claim that cannot land goes back to the reviewer ---


def test_a_fact_that_failed_to_land_can_be_rejected(configured, monkeypatch):
    from kg_mcp import workspace

    item = _approved(workspace)
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", Ingest(edges=0))
    result = asyncio.run(workspace.drain(apply=True))
    assert result["failed"][0]["error"] == "extraction produced no facts from this proposal"
    assert "draining_at" not in workspace._read()[0]
    assert workspace._human_set_status({item["id"]}, "rejected")["changed"] == 1


def test_a_claim_left_by_a_stopped_drain_is_released_by_the_next(configured, monkeypatch):
    from kg_mcp import workspace

    item = _approved(workspace)
    assert workspace._claim(workspace._read()[0])  # a drain that stopped after its claim
    assert workspace._human_set_status({item["id"]}, "rejected")["unknown"] == [item["id"]]

    async def down(records, **kwargs):
        return {"applied": True, "ingested": 0, "failed": [{"error": "down"}], "receipts": []}

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", down)
    asyncio.run(workspace.drain(apply=True))
    assert workspace._human_set_status({item["id"]}, "rejected")["changed"] == 1


def test_a_partly_landed_bundle_keeps_its_claim(configured, monkeypatch):
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
        "example", ["Aurora runs in Frankfurt.", "Aurora left Dublin."], provenance="memo"
    )
    workspace._human_set_status({item["id"]}, "approved")
    asyncio.run(workspace.drain(apply=True))
    assert workspace._read()[0]["draining_at"]
    assert workspace._human_set_status({item["id"]}, "rejected")["unknown"] == [item["id"]]


# --- approve and land in one step; see what waits for a drain ---


def test_approve_with_drain_lands_what_was_approved(configured, monkeypatch, capsys):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "Aurora ships weekly.")
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", Ingest())
    monkeypatch.setattr(sys, "argv", ["kg-workspace", "approve", item["id"], "--drain"])
    workspace.main()
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] == 1
    assert (result["drain"]["ingested"], result["drain"]["failed"]) == (1, [])
    assert workspace._read() == []


def test_approve_with_drain_exits_one_when_the_fact_does_not_land(
    configured, monkeypatch, capsys
):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "Aurora ships weekly.")
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", Ingest(edges=0))
    monkeypatch.setattr(sys, "argv", ["kg-workspace", "-H", "approve", item["id"], "--drain"])
    with pytest.raises(SystemExit) as exited:
        workspace.main()
    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "1 proposal(s) approved" in out and "FAILED" in out


def test_list_approved_shows_who_approved(tmp_path, monkeypatch, capsys):
    from kg_mcp import workspace

    _configure(
        tmp_path,
        monkeypatch,
        "workspace:\n  auto_approve:\n    - {fact_type: source-fact, require: source-link}\n",
    )
    workspace.add_proposal(
        "example", "Invoice 42 was paid.", fact_type="source-fact", source="l", link="l:42"
    )
    _approved(workspace)
    workspace.add_proposal("example", "Aurora left Dublin.")
    monkeypatch.setattr(sys, "argv", ["kg-workspace", "-H", "list", "--approved"])
    workspace.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("(approved by policy:source-fact:source-link)")
    assert lines[1].endswith("(approved by reviewer)")
    assert lines[2] == "(2 approved)"


# --- retract what a drained proposal landed ---


def test_only_a_drained_proposal_can_be_retracted(configured):
    from kg_mcp import workspace

    item = workspace.add_proposal("example", "Aurora ships weekly.")
    with pytest.raises(CommandError) as refused:
        workspace.drained(item["id"])
    assert refused.value.code == EXIT_REJECTED


def test_a_retraction_retires_only_facts_no_other_episode_gives():
    from kg_mcp import queries

    edges = [
        SimpleNamespace(uuid="only", episodes=["e1"], group_id="example", invalid_at=None),
        SimpleNamespace(uuid="shared", episodes=["e1", "e9"], group_id="example", invalid_at=None),
        SimpleNamespace(
            uuid="gone",
            episodes=["e1"],
            group_id="example",
            invalid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        SimpleNamespace(uuid="elsewhere", episodes=["e1"], group_id="other", invalid_at=None),
    ]

    async def collect(graph, settings, groups, fetch):
        return edges

    settings = SimpleNamespace(database=SimpleNamespace(provider="neo4j"))
    original = queries.collect
    queries.collect = collect
    try:
        result = asyncio.run(queries.episode_edges(None, settings, ["example"], ["e1"]))
    finally:
        queries.collect = original
    assert result == (["only"], ["shared"])


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


async def _graph_with_episodes(path: Path) -> dict[str, str]:
    """Two episodes on a Ladybug file: one gave two facts, one restated the second."""
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    from kg_mcp.ladybug import build_ladybug_driver

    driver = build_ladybug_driver(str(path), read_only=False)
    now = datetime.now(timezone.utc)
    ids = {name: str(uuid.uuid4()) for name in ("mine", "other", "only", "shared")}
    try:
        nodes = [
            EntityNode(uuid=str(uuid.uuid4()), name=name, group_id="", labels=["Entity"],
                       created_at=now, summary="")
            for name in ("Aurora", "DuckDB")
        ]
        for node in nodes:
            await node.save(driver)
        for name, episodes in (("only", ["mine"]), ("shared", ["mine", "other"])):
            await EntityEdge(
                uuid=ids[name], group_id="", source_node_uuid=nodes[0].uuid,
                target_node_uuid=nodes[1].uuid, name="USES", fact=f"Aurora fact {name}",
                created_at=now, valid_at=now, episodes=[ids[e] for e in episodes],
            ).save(driver)
        for name, edges in (("mine", ["only", "shared"]), ("other", ["shared"])):
            await EpisodicNode(
                uuid=ids[name], name=name, group_id="", labels=[], created_at=now,
                source=EpisodeType.text, source_description="test", content=name,
                valid_at=now, entity_edges=[ids[e] for e in edges],
            ).save(driver)
    finally:
        await driver.close()
    return ids


@ladybug_only
def test_kg_retire_proposal_queues_the_facts_it_alone_landed(tmp_path, monkeypatch, capsys):
    from kg_mcp import cli, workspace
    from kg_mcp.ladybug import setup_database

    path = tmp_path / "graph.ladybug"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    _configure(
        tmp_path,
        monkeypatch,
        f"database:\n  provider: ladybug\n  ladybug:\n    path: {path}\n"
        "embedder:\n  model: test-embedder\n  dimensions: 8\n",
    )
    setup_database(path, quiet=True)
    ids = asyncio.run(_graph_with_episodes(path))
    item = _approved(workspace, "Aurora uses DuckDB.")

    async def landed(records, **kwargs):
        receipt = {"episode_uuid": ids["mine"], "edges": 2}
        return {"applied": True, "ingested": 1, "failed": [], "receipts": [receipt]}

    monkeypatch.setattr("kg_mcp.ingest.ingest_records", landed)
    asyncio.run(workspace.drain(apply=True))

    monkeypatch.setattr(
        sys,
        "argv",
        ["kg", "retire", "example", "--proposal", item["id"], "--reason", "approved by mistake"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["edges"] == [ids["only"]]
    assert payload["kept"] == [ids["shared"]]
    assert payload["retracts"] == item["id"]
    queued = workspace.pending_for()[0]
    assert (queued["operation"], queued["retracts"]) == ("retire", item["id"])


# --- kg edge reads only allow-listed groups, and says when it could not read ---


def _edge_args(uuid_: str) -> SimpleNamespace:
    return SimpleNamespace(command="edge", uuid=uuid_, groups=None)


def test_kg_edge_does_not_show_a_fact_of_another_group(tmp_path, monkeypatch):
    from kg_mcp import cli, queries

    edge = SimpleNamespace(
        fact="secret plan", valid_at=None, invalid_at=None, group_id="other",
        source_node_uuid="a", target_node_uuid="b", episodes=[],
    )

    async def find_edge(drivers, uuid_, timeout, label):
        return edge, object()

    class Graph:
        driver = object()

        async def close(self):
            return None

    _configure(tmp_path, monkeypatch, "database:\n  provider: neo4j\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(queries, "find_edge", find_edge)
    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda settings, read_only: Graph())
    with pytest.raises(CommandError, match="edge not found"):
        asyncio.run(cli._graph_command(_edge_args("e1")))


def test_a_failed_edge_lookup_is_not_reported_as_missing():
    from kg_mcp import queries

    class Driver:
        provider = None
        graph_operations_interface = None

        async def execute_query(self, *args, **kwargs):
            raise ConnectionError("backend unreachable")

    with pytest.raises(RuntimeError, match="edge lookup failed: backend unreachable"):
        asyncio.run(queries.find_edge([Driver()], "e1", 5, "edge"))


def test_an_edge_absent_everywhere_is_reported_as_missing():
    from graphiti_core.errors import EdgeNotFoundError

    from kg_mcp import queries

    class Driver:
        provider = None
        graph_operations_interface = None

        async def execute_query(self, *args, **kwargs):
            raise EdgeNotFoundError("e1")

    assert asyncio.run(queries.find_edge([Driver()], "e1", 5, "edge")) == (None, None)


# --- retry what clears after a wait, never what repeats ---


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - rate limited",
        "429 RESOURCE_EXHAUSTED: quota for this minute",
        "Too Many Requests",
        "the model is overloaded",
    ],
)
def test_quota_and_overload_errors_are_retried(message):
    from kg_mcp.ingest import _retryable

    assert _retryable(RuntimeError(message))


def test_an_answer_cut_off_at_the_token_ceiling_is_not_retried():
    from kg_mcp.ingest import _retryable
    from kg_mcp.runtime import OutputTruncatedError

    assert not _retryable(OutputTruncatedError("cut off"))


def test_a_completion_that_hit_max_tokens_raises_instead_of_parsing():
    from kg_mcp.runtime import OutputTruncatedError, refuse_truncated_output

    def response(reason: str):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason)])

    answers = iter([response("stop"), response("length")])

    async def create(**kwargs):
        return next(answers)

    llm = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    refuse_truncated_output(llm)
    finished = asyncio.run(llm.client.chat.completions.create(max_tokens=50))
    assert finished.choices[0].finish_reason == "stop"
    with pytest.raises(OutputTruncatedError, match=r"max_tokens \(50\)"):
        asyncio.run(llm.client.chat.completions.create(max_tokens=50))


def test_the_generic_client_refuses_truncated_output(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "llm:\n  model: qwen3:8b\n  api_url: http://127.0.0.1:11434/v1\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    from kg_mcp.config import load_config
    from kg_mcp.runtime import build_llm

    llm = build_llm(load_config())
    assert llm.client.chat.completions.create.__name__ == "guarded"


# --- current credential formats are refused ---


@pytest.mark.parametrize(
    "token",
    [
        "glpat-" + "A1b2C3d4E5f6G7h8I9j0",
        "npm_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
        "pypi-" + "AgEIcHlwaS5vcmcCJGE0ZjM5ZDk4LTk1ZmQtNDk2Zi1hYjQ1LTAwMDAwMDAwMDAwMA",
        "sk_live_" + "51A1b2C3d4E5f6G7h8I9j0",
        "rk_live_" + "51A1b2C3d4E5f6G7h8I9j0",
        "ya29." + "a0AfH6SMBxA1b2C3d4E5f6G7h8",
        "xapp-1-" + "A0123456789-0123456789012-abcdef",
        "ASIA" + "0123456789ABCDEF",
        "dop_v1_" + "0123456789abcdef" * 4,
        "SG." + "A1b2C3d4E5f6G7h8I9" + "." + "A1b2C3d4E5f6G7h8I9j0K1",
        "shpat_" + "0123456789abcdef" * 2,
    ],
)
def test_current_credential_formats_are_refused(configured, token):
    from kg_mcp import workspace

    with pytest.raises(CommandError) as refused:
        workspace.add_proposal("example", f"The deploy key is {token} for Aurora.")
    assert refused.value.code == EXIT_SECRET_SHAPED


@pytest.mark.parametrize(
    "fact",
    [
        "Aurora expanded into ASIA and Europe in 2026.",
        "The SG.Region office in Singapore opened.",
        "We published npm_package_lock updates for Aurora.",
        "Aurora uses PostgreSQL 16 as its database.",
    ],
)
def test_ordinary_facts_still_pass_the_secret_gate(configured, fact):
    from kg_mcp import workspace

    assert workspace.add_proposal("example", fact)["status"] == "pending"
