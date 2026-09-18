"""The 0.4.0 contract: scoped reads, safer retrieval, and reviewable writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock

from kg_mcp.config import Settings, TokenGrant


def test_mcp_registry_manifest_matches_the_python_release():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "server.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "io.github.renezander030/graphiti-local"
    assert manifest["version"] == "0.4.1"
    assert manifest["packages"] == [
        {
            "registryType": "pypi",
            "identifier": "graphiti-local",
            "version": "0.4.1",
            "transport": {"type": "stdio"},
        }
    ]
    marker = f"mcp-name: {manifest['name']}"
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert marker in readme
    assert "![Graphiti Local](https://raw.githubusercontent.com/" in readme
    assert "![42-second synthetic memory demo](https://raw.githubusercontent.com/" in readme


def test_rotating_tokens_validate_group_scopes():
    settings = Settings.model_validate(
        {
            "server": {
                "transport": "streamable-http",
                "auth": {
                    "tokens": [
                        {
                            "name": "old",
                            "token": "old-token-long-enough",
                            "groups": ["alpha"],
                        },
                        {
                            "name": "new",
                            "token": "new-token-long-enough",
                            "groups": ["alpha", "beta"],
                        },
                    ]
                },
            },
            "graph": {"groups": ["alpha", "beta"]},
        }
    )
    assert [grant.name for grant in settings.server.auth.grants()] == ["old", "new"]

    with pytest.raises(ValueError, match="unconfigured group"):
        Settings.model_validate(
            {
                "server": {
                    "auth": {
                        "tokens": [{"token": "token-long-enough", "groups": ["not-configured"]}]
                    }
                },
                "graph": {"groups": ["alpha"]},
            }
        )


def test_ladybug_rejects_a_scope_it_cannot_isolate():
    with pytest.raises(ValueError, match="cannot enforce a partial token group scope"):
        Settings.model_validate(
            {
                "server": {
                    "auth": {"tokens": [{"token": "token-long-enough", "groups": ["alpha"]}]}
                },
                "graph": {"groups": ["alpha", "beta"]},
                "database": {"provider": "ladybug"},
            }
        )


def test_middleware_binds_the_matching_token_scope():
    from kg_mcp.server import BearerTokenMiddleware, authorized_groups

    settings = Settings.model_validate({"graph": {"groups": ["alpha", "beta"]}})
    observed = []

    async def downstream(scope, receive, send):
        del scope, receive
        observed.append(authorized_groups(None, settings))
        observed.append(authorized_groups("alpha", settings))
        with pytest.raises(ValueError, match="not granted"):
            authorized_groups("beta", settings)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    app = BearerTokenMiddleware(
        downstream,
        [TokenGrant(token="token-long-enough", groups=["alpha"])],
        settings.graph.groups,
    )
    statuses = []

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    async def receive():
        return {"type": "http.request"}

    asyncio.run(
        app(
            {
                "type": "http",
                "headers": [(b"authorization", b"Bearer token-long-enough")],
            },
            receive,
            send,
        )
    )
    assert statuses == [200]
    assert observed == [["alpha"], ["alpha"]]


def _edge(uuid: str, fact: str, invalid_at=None):
    return SimpleNamespace(uuid=uuid, fact=fact, invalid_at=invalid_at)


def test_current_facts_default_and_identity_safe_reranking():
    from kg_mcp.retrieval import current_edges, rerank_edges

    now = datetime.now(timezone.utc)
    edges = [
        _edge("one", "same fact"),
        _edge("two", "same fact"),
        _edge("old", "old fact", now - timedelta(seconds=1)),
    ]
    current, suppressed = current_edges(edges, include_invalidated=False)
    assert [edge.uuid for edge in current] == ["one", "two"]
    assert suppressed == 1

    class CrossEncoder:
        async def rank(self, query, passages):
            assert query == "query"
            return [(passage, 0.8 - index / 10) for index, passage in enumerate(passages)]

    graph = SimpleNamespace(cross_encoder=CrossEncoder())
    settings = Settings.model_validate({"reranker": {"min_score": 0.5}})
    ranked = asyncio.run(rerank_edges(graph, "query", current, settings, 10))
    assert [edge.uuid for edge, _ in ranked] == ["one", "two"]


def test_falkor_query_compatibility_preserves_real_identifiers():
    from kg_mcp.retrieval import compatible_query

    falkor = Settings()
    assert compatible_query("foo_bar _ baz", falkor) == "foo_bar baz"
    with pytest.raises(ValueError, match="empty"):
        compatible_query(" _ ", falkor)
    neo4j = Settings.model_validate({"database": {"provider": "neo4j"}})
    assert compatible_query("foo_bar _ baz", neo4j) == "foo_bar _ baz"


def test_falkor_multi_label_node_search_fans_out(monkeypatch):
    from kg_mcp.server import create_server

    calls = []

    class Graph:
        driver = SimpleNamespace()

        async def search_(self, query, **kwargs):
            labels = kwargs["search_filter"].node_labels
            calls.append((query, labels))
            label = labels[0]
            node = SimpleNamespace(
                uuid=f"node-{label}",
                name=label,
                labels=[label],
                created_at=None,
                summary="",
                group_id="main",
                attributes={},
            )
            return SimpleNamespace(nodes=[node])

        async def close(self):
            return None

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: Graph())
    server = create_server(Settings())

    async def call():
        async with server._mcp_server.lifespan(server._mcp_server):
            return await server.call_tool(
                "search_nodes",
                {"query": "name _ here", "entity_types": ["Person", "Company"]},
            )

    result = asyncio.run(call())
    payload = result[1] if isinstance(result, tuple) else result
    if not isinstance(payload, dict):
        payload = json.loads(payload[0].text)
    assert [node["uuid"] for node in payload["nodes"]] == ["node-Person", "node-Company"]
    assert calls == [("name here", ["Person"]), ("name here", ["Company"])]


def test_verifiable_snapshot_detects_tampering_and_accepts_v1(tmp_path: Path):
    from kg_mcp.export import FORMAT_VERSION, _atomic_snapshot, _json_line
    from kg_mcp.restore import read_snapshot

    records = [{"kind": "entity_node", "uuid": "n1", "name": "Ada"}]
    digest = hashlib.sha256(b"".join(_json_line(record) for record in records)).hexdigest()
    header = {
        "kind": "export",
        "format_version": FORMAT_VERSION,
        "record_count": 1,
        "sha256": digest,
    }
    snapshot = tmp_path / "snapshot.jsonl"
    _atomic_snapshot(snapshot, header, records)
    assert read_snapshot(snapshot)[1][0]["name"] == "Ada"

    snapshot.write_text(
        snapshot.read_text(encoding="utf-8").replace("Ada", "Grace"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_snapshot(snapshot)

    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps({"kind": "export", "format_version": 1}) + "\n" + json.dumps(records[0]) + "\n",
        encoding="utf-8",
    )
    assert len(read_snapshot(legacy)[1]) == 1


def test_ladybug_writer_lock_times_out_with_a_clear_error(tmp_path: Path):
    from kg_mcp.write_lock import graph_writer_lock, lock_path

    database = tmp_path / "graph.ladybug"
    settings = Settings.model_validate(
        {
            "graph": {"writer_lock_timeout_seconds": 0.01},
            "database": {"provider": "ladybug", "ladybug": {"path": str(database)}},
        }
    )
    with (
        FileLock(str(lock_path(database))),
        pytest.raises(RuntimeError, match="another ingest, restore, drain, or setup"),
        graph_writer_lock(settings),
    ):
        pass


def test_llm_token_budget_caps_an_explicit_upstream_request():
    from kg_mcp.runtime import cap_llm_tokens

    class Client:
        def __init__(self):
            self.max_tokens = 99
            self.received = None

        async def generate_response(self, messages, **kwargs):
            del messages
            self.received = kwargs["max_tokens"]
            return {"ok": True}

    client = cap_llm_tokens(Client(), 2048)
    result = asyncio.run(client.generate_response([], max_tokens=16384))
    assert result == {"ok": True}
    assert client.received == 2048
    assert client.configured_max_tokens == 2048


def test_llm_api_mode_can_override_endpoint_guessing():
    from kg_mcp.runtime import uses_responses_api

    proxy = {"llm": {"model": "test", "api_url": "https://proxy.example/v1"}}
    assert uses_responses_api(Settings.model_validate(proxy)) is False
    proxy["llm"]["api_mode"] = "responses"
    assert uses_responses_api(Settings.model_validate(proxy)) is True
    official = {"llm": {"model": "test", "api_mode": "chat"}}
    assert uses_responses_api(Settings.model_validate(official)) is False


def test_review_stage_never_builds_the_production_backend(tmp_path: Path, monkeypatch):
    from kg_mcp.ingest import stage_extraction

    production = Settings.model_validate(
        {
            "graph": {"groups": ["review"]},
            "database": {"provider": "falkordb"},
        }
    )
    observed = []

    async def fake_ingest(records, **kwargs):
        del records
        observed.append(("ingest", kwargs["settings"].database.provider))
        return {"ingested": 1, "failed": []}

    class Graph:
        async def close(self):
            return None

    def fake_build(settings, *, read_only):
        observed.append(("build", settings.database.provider, read_only))
        return Graph()

    async def fake_export(graph, settings, groups, *, output):
        del graph
        observed.append(("export", settings.database.provider, groups))
        return {"output": output, "record_count": 3}

    monkeypatch.setattr("kg_mcp.ingest.load_config", lambda: production)
    monkeypatch.setattr("kg_mcp.ingest.ingest_records", fake_ingest)
    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", fake_build)
    monkeypatch.setattr("kg_mcp.export.export_graph", fake_export)
    monkeypatch.setattr("kg_mcp.ladybug.setup_database", lambda *a, **k: [])

    result = asyncio.run(
        stage_extraction(
            [{"name": "input", "body": "body", "domain": "review"}],
            tmp_path / "review.jsonl",
        )
    )
    assert result["production_graph_changed"] is False
    assert all("falkordb" not in item for item in observed)
    assert result["promote"].endswith("--group review --apply")
