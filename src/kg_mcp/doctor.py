"""Preflight checks: prove the configuration works before a run depends on it.

Every check reports ``ok``, ``warn`` or ``fail`` with a one-line detail. Only ``fail``
sets a non-zero exit code, so a missing optional extra warns without blocking a run.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kg_mcp.config import (
    KNOWN_EMBEDDING_DIMENSIONS,
    Settings,
    config_path,
    embedding_model_base,
    load_config,
)

TIMEOUT = 8.0
OK = "ok"
WARN = "warn"
FAIL = "fail"


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> Any:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, api_key: str) -> int:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return int(response.status)


def check_config(explicit: str | None) -> tuple[list[dict[str, str]], Settings | None]:
    path = config_path(explicit)
    try:
        settings = load_config(explicit)
    except FileNotFoundError as exc:
        return [_check("config", FAIL, str(exc))], None
    except Exception as exc:
        return [_check("config", FAIL, f"{path}: {exc}")], None
    results = [_check("config", OK, f"{path} parsed; groups: {', '.join(settings.graph.groups)}")]
    return results, settings


def check_workspace(settings: Settings) -> dict[str, str]:
    directory = Path(settings.graph.workspace_dir).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".doctor-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return _check("workspace", FAIL, f"{directory} is not writable: {exc}")
    return _check("workspace", OK, f"{directory} is writable")


def check_embedder(settings: Settings) -> dict[str, str]:
    """Probe the real vector width. A silent mismatch here corrupts every stored embedding."""
    url = f"{settings.embedder.api_url.rstrip('/')}/embeddings"
    try:
        payload = _post_json(
            url,
            {"model": settings.embedder.model, "input": "graphiti local preflight"},
            settings.embedder.api_key,
        )
    except urllib.error.HTTPError as exc:
        return _check("embedder", FAIL, f"{url} returned HTTP {exc.code}")
    except Exception as exc:
        return _check("embedder", FAIL, f"{url} unreachable: {exc}")
    try:
        actual = len(payload["data"][0]["embedding"])
    except Exception:
        return _check("embedder", FAIL, f"{url} returned no embedding vector")
    if actual != settings.embedder.dimensions:
        return _check(
            "embedder",
            FAIL,
            f"{settings.embedder.model} returns {actual} dimensions but "
            f"embedder.dimensions is {settings.embedder.dimensions}",
        )
    return _check("embedder", OK, f"{settings.embedder.model} returns {actual} dimensions")


def check_llm(settings: Settings) -> dict[str, str]:
    url = f"{settings.llm.api_url.rstrip('/')}/models"
    if not settings.llm.api_key and "api.openai.com" in settings.llm.api_url:
        return _check("llm", FAIL, "llm.api_key is empty for the official OpenAI endpoint")
    try:
        status = _get(url, settings.llm.api_key)
    except urllib.error.HTTPError as exc:
        detail = "check llm.api_key" if exc.code in (401, 403) else f"HTTP {exc.code}"
        return _check("llm", FAIL, f"{url}: {detail}")
    except Exception as exc:
        return _check("llm", FAIL, f"{url} unreachable: {exc}")
    return _check("llm", OK, f"{settings.llm.model} via {settings.llm.api_url} (HTTP {status})")


def check_database(settings: Settings) -> dict[str, str]:
    provider = settings.database.provider
    if provider == "falkordb":
        endpoint = settings.database.falkordb
        try:
            with socket.create_connection((endpoint.host, endpoint.port), timeout=TIMEOUT):
                pass
        except Exception as exc:
            return _check("database", FAIL, f"falkordb {endpoint.host}:{endpoint.port}: {exc}")
        return _check("database", OK, f"falkordb reachable at {endpoint.host}:{endpoint.port}")
    if provider == "neo4j":
        uri = settings.database.neo4j.uri
        host = uri.split("://", 1)[-1].split("/", 1)[0]
        hostname, _, port = host.partition(":")
        try:
            with socket.create_connection((hostname, int(port or 7687)), timeout=TIMEOUT):
                pass
        except Exception as exc:
            return _check("database", FAIL, f"neo4j {uri}: {exc}")
        return _check("database", OK, f"neo4j reachable at {uri}")
    # Resolve so the message names a real path; a relative path is read from the
    # working directory, which is rarely what the reader assumes.
    path = Path(settings.database.ladybug.path).expanduser().resolve()
    if not path.exists():
        return _check(
            "database",
            WARN,
            f"ladybug database not created yet at {path}; run kg-ladybug-setup",
        )
    return _check("database", OK, f"ladybug database present at {path}")


def check_reranker(settings: Settings) -> dict[str, str]:
    provider = settings.reranker.provider
    if provider == "passthrough":
        return _check("reranker", OK, "passthrough (no extra model call)")
    if provider == "bge":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return _check(
                "reranker",
                FAIL,
                "reranker.provider 'bge' needs the optional extra: uv sync --extra rerank",
            )
        return _check("reranker", OK, "bge cross-encoder available locally")
    if not settings.reranker.api_key:
        return _check("reranker", FAIL, f"reranker.provider '{provider}' has no api_key")
    return _check("reranker", OK, f"{provider} reranker configured")


def check_transport(settings: Settings) -> dict[str, str]:
    server = settings.server
    if server.transport == "stdio":
        return _check("transport", OK, "stdio (no network exposure)")
    if server.host not in {"127.0.0.1", "::1", "localhost"} and not server.auth.token:
        return _check("transport", FAIL, f"streamable-http on {server.host} without a token")
    scope = ", ".join(server.auth.groups) if server.auth.groups else "all configured groups"
    return _check(
        "transport",
        OK,
        f"streamable-http on {server.host}:{server.port}; token grants {scope}",
    )


def check_embedding_model_known(settings: Settings) -> dict[str, str]:
    base = embedding_model_base(settings.embedder.model)
    if base in KNOWN_EMBEDDING_DIMENSIONS:
        return _check("embedder-model", OK, f"{base} is a known model")
    return _check(
        "embedder-model",
        WARN,
        f"{base} is not in the known-dimension table; relying on the live probe",
    )


def run_checks(explicit: str | None = None, *, offline: bool = False) -> list[dict[str, str]]:
    results, settings = check_config(explicit)
    if settings is None:
        return results
    results.append(check_workspace(settings))
    results.append(check_transport(settings))
    results.append(check_reranker(settings))
    results.append(check_embedding_model_known(settings))
    if offline:
        results.append(_check("database", WARN, "skipped (--offline)"))
        results.append(_check("llm", WARN, "skipped (--offline)"))
        results.append(_check("embedder", WARN, "skipped (--offline)"))
        return results
    results.append(check_database(settings))
    results.append(check_llm(settings))
    results.append(check_embedder(settings))
    return results


def worst_status(results: list[dict[str, str]]) -> str:
    if any(item["status"] == FAIL for item in results):
        return FAIL
    if any(item["status"] == WARN for item in results):
        return WARN
    return OK
