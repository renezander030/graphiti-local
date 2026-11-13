"""Damaged identity is refused; transient provider status preserves bounded retries."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from kg_mcp import fingerprint, ingest
from kg_mcp.config import Settings
from kg_mcp.doctor import check_embedder_fingerprint


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(tmp_path))
    return Settings.model_validate({
        "graph": {
            "groups": ["example"], "workspace_dir": str(tmp_path),
            "ingest_max_attempts": 2, "ingest_retry_base_seconds": 0,
        },
        "database": {"provider": "ladybug", "ladybug": {"path": str(tmp_path / "graph")}},
        "embedder": {"model": "synthetic-model", "dimensions": 8},
    })


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("payload", [b"{broken", b"[]", b"null", b"\xff"])
def test_corrupt_metadata_blocks_ingest_before_open_and_is_preserved(
    settings, tmp_path, monkeypatch, sidecar, payload,
):
    path = (
        fingerprint.sidecar_path(settings.database.ladybug.path)
        if sidecar else tmp_path / "embedders.json"
    )
    path.write_bytes(payload)

    def opened(*args, **kwargs):
        pytest.fail("corrupt identity must be refused before opening the graph")

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", opened)
    with pytest.raises(ValueError, match="invalid embedder"):
        asyncio.run(ingest.ingest_records(
            [{"name": "synthetic", "body": "Aurora uses a local database."}],
            apply=True, settings=settings,
        ))
    assert path.read_bytes() == payload
    assert check_embedder_fingerprint(settings)["status"] == "fail"
    assert not (tmp_path / "ingest-receipts.jsonl").exists()


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("identity", [None, {}, {"model": "synthetic-model"},
                                      {"model": "synthetic-model", "dimensions": True},
                                      {"model": "", "dimensions": 8}])
def test_existing_invalid_identity_is_not_first_write(settings, tmp_path, sidecar, identity):
    if sidecar:
        path = fingerprint.sidecar_path(settings.database.ladybug.path)
        data = identity
    else:
        path = tmp_path / "embedders.json"
        data = {fingerprint.database_key(settings): identity}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid embedder"):
        fingerprint.drift(settings)
    assert check_embedder_fingerprint(settings)["status"] == "fail"


def test_missing_metadata_remains_a_new_graph(settings):
    assert fingerprint.recorded(settings) is None
    assert fingerprint.drift(settings) is None
    assert check_embedder_fingerprint(settings)["status"] == "warn"
    fingerprint.record(settings)
    assert fingerprint.drift(settings) is None
    assert check_embedder_fingerprint(settings)["status"] == "ok"


@pytest.mark.parametrize("transport", ["sdk", "httpx"])
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 400, 401])
def test_http_status_controls_bounded_ingest_retry(settings, monkeypatch, transport, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://provider.example"))
    error = (
        openai.APIStatusError("provider rejected request", response=response, body=None)
        if transport == "sdk"
        else httpx.HTTPStatusError("provider rejected request", request=response.request,
                                   response=response)
    )
    calls = []
    closed = []

    class Graph:
        async def build_indices_and_constraints(self):
            pass

        async def add_episode(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise error
            return SimpleNamespace(episode=SimpleNamespace(uuid=kwargs["uuid"]),
                                   nodes=[], edges=[object()])

        async def close(self):
            closed.append(True)

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", lambda *a, **k: Graph())
    result = asyncio.run(ingest.ingest_records(
        [{"name": "synthetic", "body": "Aurora uses a local database."}],
        apply=True, settings=settings,
    ))
    if status in (400, 401):
        assert len(calls) == 1
        assert result["ingested"] == 0
        assert result["failed"][0]["retryable"] is False
    else:
        assert len(calls) == 2
        assert calls[0]["uuid"] == calls[1]["uuid"]
        assert result["ingested"] == 1 and not result["failed"]
        assert result["receipts"][0]["attempts"] == 2
    assert closed == [True]


@pytest.mark.parametrize("payload", [b"{broken", b"[]", b"\xff"])
def test_valid_workspace_cannot_hide_corrupt_sidecar(settings, monkeypatch, payload):
    fingerprint.record(settings)
    path = fingerprint.sidecar_path(settings.database.ladybug.path)
    path.write_bytes(payload)

    def opened(*args, **kwargs):
        pytest.fail("workspace identity cannot conceal damaged sidecar evidence")

    monkeypatch.setattr("kg_mcp.runtime.build_graphiti", opened)
    with pytest.raises(ValueError, match="invalid embedder"):
        asyncio.run(ingest.ingest_records(
            [{"name": "synthetic", "body": "Aurora uses a local database."}],
            apply=True, settings=settings,
        ))
    assert path.read_bytes() == payload
    assert check_embedder_fingerprint(settings)["status"] == "fail"


def test_conflicting_workspace_and_sidecar_identities_fail(settings):
    fingerprint.record(settings)
    path = fingerprint.sidecar_path(settings.database.ladybug.path)
    data = json.loads(path.read_text())
    data["model"] = "another-synthetic-model"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="identities disagree"):
        fingerprint.drift(settings)
    assert json.loads(path.read_text())["model"] == "another-synthetic-model"
    assert check_embedder_fingerprint(settings)["status"] == "fail"
