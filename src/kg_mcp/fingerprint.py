"""Record which embedder produced a graph's vectors, and refuse to mix in another.

The dimension guard catches a width mismatch. A model change at the same width is
worse: every query vector comes from a different space than the stored ones, every
search still returns results, and the ranking is quietly wrong. So the first ingest
records the embedder for the database it wrote to, later writes must match it, and
``kg doctor`` compares the configuration against the record.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_mcp.config import Settings, embedding_model_base

FILE = "embedders.json"


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
    return entry if isinstance(entry, dict) else None


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
    return entry


def drift(settings: Settings) -> str | None:
    """A message when the configured embedder is not the one that wrote the graph."""
    previous = recorded(settings)
    if previous is None:
        return None
    now = current(settings)
    if previous.get("model") == now["model"] and previous.get("dimensions") == now["dimensions"]:
        return None
    return (
        f"the stored vectors were produced by {previous.get('model')} "
        f"({previous.get('dimensions')} dimensions) but embedder.model is "
        f"{settings.embedder.model} ({now['dimensions']}); mixing embedders corrupts search "
        "silently. Restore the previous embedder, or rebuild the graph under the new one: "
        "kg export, an empty database, then kg-ingest SNAPSHOT --restore --apply"
    )


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
