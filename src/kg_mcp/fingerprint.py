"""Record which embedder produced a graph's vectors, and refuse to mix in another.

The dimension guard catches a width mismatch. A model change at the same width is
worse: every query vector comes from a different space than the stored ones, every
search still returns results, and the ranking is quietly wrong. So the first ingest
records the embedder for the database it wrote to, later writes must match it, and
``kg doctor`` compares the configuration against the record.

For a Ladybug file the record is also kept beside the file, as ``<file>.embedder.json``.
:func:`recorded_embedder`, :func:`record_embedder` and :func:`embedder_drift` read and
write that record from a database path alone, for callers that use the graph as a
library without a configuration file or workspace.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_mcp.config import Settings, embedding_model_base

FILE = "embedders.json"
SIDECAR_SUFFIX = ".embedder.json"


def sidecar_path(database_path: str | Path) -> Path:
    target = Path(database_path).expanduser().resolve()
    return target.with_name(target.name + SIDECAR_SUFFIX)


def recorded_embedder(database_path: str | Path) -> dict[str, Any] | None:
    """The embedder recorded for a Ladybug file, or ``None`` before its first write."""
    path = sidecar_path(database_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def record_embedder(
    database_path: str | Path,
    *,
    model: str,
    dimensions: int,
    api_url: str = "",
) -> dict[str, Any]:
    """Record the embedder that wrote a Ladybug file. Call it after a write lands."""
    from filelock import FileLock

    path = sidecar_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "model": embedding_model_base(model),
        "dimensions": dimensions,
        "api_url": api_url,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with FileLock(str(path) + ".lock"):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    return entry


def _drift_message(previous: dict[str, Any], model: str, dimensions: int) -> str | None:
    if (
        previous.get("model") == embedding_model_base(model)
        and previous.get("dimensions") == dimensions
    ):
        return None
    return (
        f"the stored vectors were produced by {previous.get('model')} "
        f"({previous.get('dimensions')} dimensions) but the embedder is now "
        f"{model} ({dimensions}); mixing embedders corrupts search silently. Restore the "
        "previous embedder, or rebuild the graph under the new one: kg export, an empty "
        "database, then kg-ingest SNAPSHOT --restore --apply"
    )


def embedder_drift(database_path: str | Path, *, model: str, dimensions: int) -> str | None:
    """A refusal message when ``model`` is not the embedder that wrote the file."""
    previous = recorded_embedder(database_path)
    if previous is None:
        return None
    return _drift_message(previous, model, dimensions)


def database_key(settings: Settings) -> str:
    """One graph, one record: keyed by the backend and where it lives."""
    database = settings.database
    if database.provider == "ladybug":
        identity = str(Path(database.ladybug.path).expanduser().resolve())
    elif database.provider == "falkordb":
        identity = f"{database.falkordb.host}:{database.falkordb.port}"
    else:
        identity = database.neo4j.uri
    return f"{database.provider}:{identity}"


def current(settings: Settings) -> dict[str, Any]:
    return {
        "model": embedding_model_base(settings.embedder.model),
        "dimensions": settings.embedder.dimensions,
        "api_url": settings.embedder.api_url,
    }


def _path() -> Path:
    from kg_mcp.workspace import workspace_dir

    return workspace_dir() / FILE


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def recorded(settings: Settings) -> dict[str, Any] | None:
    entry = _load().get(database_key(settings))
    if isinstance(entry, dict):
        return entry
    path = _ladybug_file(settings)
    return recorded_embedder(path) if path else None


def _ladybug_file(settings: Settings) -> str | None:
    """The one Ladybug file ``settings`` addresses, if it addresses exactly one."""
    if settings.database.provider != "ladybug":
        return None
    groups = settings.graph.groups
    try:
        return settings.database.ladybug.path_for(groups[0] if len(groups) == 1 else None)
    except ValueError:
        return None


def record(settings: Settings) -> dict[str, Any]:
    """Stamp the configured embedder for this database. Called after a write lands."""
    from filelock import FileLock

    from kg_mcp.workspace import workspace_dir

    directory = workspace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    entry = {**current(settings), "recorded_at": datetime.now(timezone.utc).isoformat()}
    with FileLock(str(directory / ".lock")):
        data = _load()
        data[database_key(settings)] = entry
        target = _path()
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
    path = _ladybug_file(settings)
    if path:
        record_embedder(
            path,
            model=settings.embedder.model,
            dimensions=settings.embedder.dimensions,
            api_url=settings.embedder.api_url,
        )
    return entry


def drift(settings: Settings) -> str | None:
    """A message when the configured embedder is not the one that wrote the graph."""
    previous = recorded(settings)
    if previous is None:
        return None
    return _drift_message(previous, settings.embedder.model, settings.embedder.dimensions)


def check(settings: Settings) -> tuple[str, str]:
    """``(status, detail)`` for ``kg doctor``."""
    previous = recorded(settings)
    if previous is None:
        return "warn", "no embedder recorded yet; the first ingest or restore records it"
    message = drift(settings)
    if message:
        return "fail", message
    now = current(settings)
    if previous.get("api_url") != now["api_url"]:
        return (
            "warn",
            f"{now['model']} matches the record but is served from {now['api_url']} instead of "
            f"{previous.get('api_url')}; a different host can serve a different build",
        )
    return "ok", f"{now['model']} ({now['dimensions']} dimensions) wrote this graph"
