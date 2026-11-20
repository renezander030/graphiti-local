"""A waiting writer must validate the identity left by the preceding writer."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from kg_mcp import fingerprint, runtime, write_lock
from kg_mcp.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(tmp_path / "workspace"))
    return Settings.model_validate({
        "graph": {"groups": ["example"], "writer_lock_timeout_seconds": 2},
        "database": {"provider": "ladybug", "ladybug": {"path": str(tmp_path / "graph")}},
        "embedder": {"model": "waiting-model", "dimensions": 8},
    })


@pytest.mark.parametrize("initial_record", [False, True])
@pytest.mark.parametrize("update", ["different-model", "different-width", "damaged"])
def test_waiting_writer_rechecks_identity_before_open(
    settings, monkeypatch, initial_record, update,
):
    if initial_record:
        fingerprint.record(settings)
    attempting = Event()
    original_lock = write_lock.graph_writer_lock

    @contextmanager
    def waiting_lock(active):
        attempting.set()
        with original_lock(active):
            yield

    opened = []

    def build(*args, **kwargs):
        opened.append(True)
        raise AssertionError("an incompatible identity must be refused before graph construction")

    monkeypatch.setattr(write_lock, "graph_writer_lock", waiting_lock)
    monkeypatch.setattr(runtime, "build_graphiti", build)

    async def write():
        async with runtime.writing(settings):
            pytest.fail("incompatible writer entered the session")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with original_lock(settings):
            future = pool.submit(asyncio.run, write())
            assert attempting.wait(timeout=2), "second writer never reached the lock"
            if update == "damaged":
                path = fingerprint.sidecar_path(settings.database.ladybug.path)
                path.write_bytes(b"{broken")
                expected = "invalid embedder"
            else:
                changes = {"model": "first-model"} if update == "different-model" else {
                    "dimensions": 16,
                }
                previous = settings.model_copy(update={
                    "embedder": settings.embedder.model_copy(update=changes),
                })
                fingerprint.record(previous)
                path = fingerprint.sidecar_path(settings.database.ladybug.path)
                expected = "mixing embedders"
            saved = path.read_bytes()
        with pytest.raises(ValueError, match=expected):
            future.result(timeout=5)
    assert opened == []
    assert path.read_bytes() == saved
    # The refusal released the real writer lock.
    with original_lock(settings):
        pass


@pytest.mark.parametrize("track_fingerprint", [True, False])
def test_write_session_preserves_matching_identity_and_opt_out(
    settings, monkeypatch, track_fingerprint,
):
    events = []
    fingerprint.record(settings)
    if not track_fingerprint:
        fingerprint.sidecar_path(settings.database.ladybug.path).write_bytes(b"{broken")
    path = fingerprint.sidecar_path(settings.database.ladybug.path)
    saved = path.read_bytes()

    class Graph:
        async def build_indices_and_constraints(self):
            events.append("indexes")

        async def close(self):
            events.append("closed")

    monkeypatch.setattr(runtime, "build_graphiti", lambda *a, **k: Graph())

    async def write():
        async with runtime.writing(settings, track_fingerprint=track_fingerprint) as session:
            session.wrote = True
            events.append("wrote")

    asyncio.run(write())
    assert events == ["indexes", "wrote", "closed"]
    if track_fingerprint:
        assert fingerprint.drift(settings) is None
    else:
        assert path.read_bytes() == saved
