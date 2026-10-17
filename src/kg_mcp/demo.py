"""A first answer before anything is installed.

The quickstart in the README asks for uv, Ollama, two model pulls, three exported
variables and a schema setup before the first question can be asked. Most of that
exists to serve the *write* path: extraction needs a language model and hybrid
retrieval embeds the query. Neither is needed to read a graph by keyword.

So this ships a small synthetic graph and answers a question against it using
BM25 alone. Nothing here contacts a model, which is why the generated config
points the language model and embedder at a closed port: if some future change
starts calling them, this demo fails loudly rather than quietly downloading
several gigabytes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QUESTION = "Which database does Aurora Analytics use?"
GROUP = "demo"

# Any call to these is a bug in the demo, so they point somewhere that refuses fast.
_UNREACHABLE = "http://127.0.0.1:1/v1"

_CONFIG = """\
server:
  transport: stdio
graph:
  groups: [demo]
  workspace_dir: {workspace}
  query_timeout_seconds: 30
database:
  provider: ladybug
  ladybug:
    path: {database}
llm:
  model: unused-in-the-demo
  api_url: {unreachable}
  api_key: none
  structured_output_mode: json_schema
  temperature: 0
embedder:
  model: nomic-embed-text
  dimensions: 768
  api_url: {unreachable}
  api_key: none
reranker:
  provider: passthrough
"""


def _demo_home() -> Path:
    """Keep the built graph out of the working tree so the demo is repeatable."""
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "graphiti-local" / "demo"


def _facts_path() -> Path:
    """The shipped facts live inside the package, not beside it.

    An installed wheel contains only the package directory, so anything resolved
    relative to the repository root exists while developing and is missing the moment
    someone installs from git. That is the failure this demo exists to avoid.
    """
    return Path(__file__).with_name("demo_facts.json")


async def _build(database: Path) -> int:
    """Write the shipped facts straight into the graph, skipping extraction.

    ``fact_embedding`` is left unset. The full text indexes BM25 uses are built by
    the Ladybug setup, and the vector column is only read by the similarity half of
    a hybrid search, which the demo never runs.
    """
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    from kg_mcp.ladybug import build_ladybug_driver

    payload = json.loads(_facts_path().read_text())
    driver = build_ladybug_driver(str(database), read_only=False)
    now = datetime.now(timezone.utc)
    try:
        seen: dict[str, EntityNode] = {}

        async def node(name: str) -> EntityNode:
            if name not in seen:
                entity = EntityNode(
                    uuid=str(uuid.uuid4()),
                    name=name,
                    group_id=GROUP,
                    labels=["Entity"],
                    created_at=now,
                    summary="",
                )
                await entity.save(driver)
                seen[name] = entity
            return seen[name]

        for item in payload["facts"]:
            source = await node(item["source"])
            target = await node(item["target"])
            edge = EntityEdge(
                uuid=str(uuid.uuid4()),
                group_id=GROUP,
                source_node_uuid=source.uuid,
                target_node_uuid=target.uuid,
                name=item["name"],
                fact=item["fact"],
                created_at=now,
                valid_at=datetime.fromisoformat(item["valid_at"]),
                invalid_at=datetime.fromisoformat(item["invalid_at"])
                if item.get("invalid_at")
                else None,
                episodes=[],
            )
            await edge.save(driver)
        return len(payload["facts"])
    finally:
        await driver.close()


def _engine_version() -> str:
    from kg_mcp.ladybug import _alias_kuzu

    return str(getattr(_alias_kuzu(), "__version__", "unknown"))


def _facts_digest() -> str:
    """The shipped facts' fingerprint, so a graph cached from older facts is rebuilt."""
    return hashlib.sha256(_facts_path().read_bytes()).hexdigest()[:12]


def _ensure_graph(home: Path) -> Path:
    """Build the graph once per engine version and set of facts, then reuse it.

    Ladybug installs its extensions per engine version, so a graph built by an older
    release finds FTS and VECTOR missing after an upgrade and the read fails. The marker
    records the engine and the facts that built the graph; anything else rebuilds it
    from scratch.
    """
    from kg_mcp.ladybug import setup_database

    database = home / "graph.ladybug"
    marker = home / ".built"
    engine = _engine_version()
    digest = _facts_digest()
    if marker.exists() and marker.read_text().split()[1:] == [engine, digest]:
        return database
    for stale in home.glob("graph.ladybug*"):
        stale.unlink()
    home.mkdir(parents=True, exist_ok=True)
    setup_database(database, quiet=True)
    count = asyncio.run(_build(database))
    marker.write_text(f"{count} {engine} {digest}\n")
    return database


def _day(moment: datetime | None) -> str | None:
    return moment.date().isoformat() if moment else None


def _split(edges: list[Any], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The facts that hold now, then the ones a later fact replaced.

    A replaced fact still matches the question by keyword, which is the point of a
    temporal graph: it answers with what is true now and keeps what used to be.
    """
    from kg_mcp.retrieval import current_edge

    def row(edge: Any) -> dict[str, Any]:
        return {
            "fact": edge.fact,
            "valid_at": _day(edge.valid_at),
            "invalid_at": _day(edge.invalid_at),
        }

    current = [row(edge) for edge in edges if current_edge(edge)][:limit]
    replaced = [row(edge) for edge in edges if not current_edge(edge)][:limit]
    return current, replaced


async def _ask(question: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from kg_mcp.config import load_config
    from kg_mcp.retrieval import keyword_edge_search_config
    from kg_mcp.runtime import build_graphiti

    settings = load_config()
    graph = build_graphiti(settings, read_only=True)
    try:
        # room for the replaced facts beside the current ones
        results = await graph.search_(
            question, config=keyword_edge_search_config(limit * 2), group_ids=[GROUP]
        )
        return _split(list(results.edges), limit)
    finally:
        await graph.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kg-demo",
        description="Answer a question against a shipped synthetic graph, without any model.",
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)

    home = _demo_home()
    database = _ensure_graph(home)

    config = home / "demo.yaml"
    config.write_text(_CONFIG.format(workspace=home, database=database, unreachable=_UNREACHABLE))
    os.environ["GRAPHITI_LOCAL_CONFIG"] = str(config)
    os.environ.setdefault("KG_WORKSPACE_DIR", str(home))

    facts, replaced = asyncio.run(_ask(args.question, args.limit))

    print(f"\n  question  {args.question}")
    if not facts and not replaced:
        print("  no match   keyword search found nothing. Try wording closer to the stored fact.")
    for item in facts:
        print(f"\n  fact      {item['fact']}")
        print(f"  valid     since {item['valid_at'] or 'an unknown day'}")
    for item in replaced:
        print(f"\n  replaced  {item['fact']}")
        print(f"  valid     {item['valid_at']} until {item['invalid_at']}")
    print(f"\n  graph     {database}")
    print("  note      synthetic facts, keyword search only, no model was contacted.\n")
    return 0 if facts else 1


if __name__ == "__main__":
    sys.exit(main())
