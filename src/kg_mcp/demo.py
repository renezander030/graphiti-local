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
import json
import os
import sys
import uuid
from datetime import datetime
from datetime import timezone
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
    return Path(__file__).resolve().parents[2] / "examples" / "demo_facts.json"


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
                episodes=[],
            )
            await edge.save(driver)
        return len(payload["facts"])
    finally:
        await driver.close()


def _ensure_graph(home: Path) -> Path:
    """Build the graph once, then reuse it on later runs."""
    from kg_mcp.ladybug import setup_database

    database = home / "graph.ladybug"
    marker = home / ".built"
    if marker.exists():
        return database
    home.mkdir(parents=True, exist_ok=True)
    setup_database(database, quiet=True)
    count = asyncio.run(_build(database))
    marker.write_text(f"{count}\n")
    return database


async def _ask(question: str, limit: int) -> list[dict[str, Any]]:
    from kg_mcp.cli import _keyword_edge_config
    from kg_mcp.config import load_config
    from kg_mcp.runtime import build_graphiti

    settings = load_config()
    graph = build_graphiti(settings, read_only=True)
    try:
        results = await graph.search_(question, config=_keyword_edge_config(limit), group_ids=[GROUP])
        return [
            {"fact": edge.fact, "valid_at": edge.valid_at.isoformat() if edge.valid_at else None}
            for edge in results.edges[:limit]
        ]
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
    config.write_text(
        _CONFIG.format(workspace=home, database=database, unreachable=_UNREACHABLE)
    )
    os.environ["GRAPHITI_LOCAL_CONFIG"] = str(config)
    os.environ.setdefault("KG_LADYBUG_PATH", str(database))
    os.environ.setdefault("KG_WORKSPACE_DIR", str(home))

    facts = asyncio.run(_ask(args.question, args.limit))

    print(f"\n  question  {args.question}")
    if not facts:
        print("  no match   keyword search found nothing. Try wording closer to the stored fact.")
    for item in facts:
        print(f"\n  fact      {item['fact']}")
        print(f"  valid at  {item['valid_at']}")
    print(f"\n  graph     {database}")
    print("  note      synthetic facts, keyword search only, no model was contacted.\n")
    return 0 if facts else 1


if __name__ == "__main__":
    sys.exit(main())
