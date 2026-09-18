"""Cross-process serialization for the embedded graph's single writer."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from kg_mcp.config import Settings


def lock_path(database_path: str | Path) -> Path:
    target = Path(database_path).expanduser().resolve()
    return target.with_name(target.name + ".writer.lock")


@contextmanager
def ladybug_path_lock(database_path: str | Path, timeout: float) -> Iterator[None]:
    path = lock_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(path), timeout=timeout):
            yield
    except Timeout as exc:
        raise RuntimeError(
            f"timed out after {timeout:g}s waiting for the Ladybug writer lock {path}; "
            "another ingest, restore, drain, or setup command is still writing"
        ) from exc


@contextmanager
def graph_writer_lock(settings: Settings) -> Iterator[None]:
    if settings.database.provider != "ladybug":
        yield
        return
    with ladybug_path_lock(
        settings.database.ladybug.path,
        settings.graph.writer_lock_timeout_seconds,
    ):
        yield
