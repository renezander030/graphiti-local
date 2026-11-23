"""FalkorDB 6.x refuses graphiti-core's procedure form of node full-text index creation.

https://github.com/getzep/graphiti/issues/1947: on FalkorDB 6 the call
``CALL db.idx.fulltext.createNodeIndex({label, stopwords}, 'a', 'b', ...)`` fails with
"Received 5 arguments to procedure 'db.idx.fulltext.createNodeIndex', expected at most 1",
so ``build_indices_and_constraints`` stops and no node full-text index exists. The fake
executor below answers like a FalkorDB 6 server; no server is contacted.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from kg_mcp import runtime
from kg_mcp.config import Settings


class Falkor6Executor:
    """Accepts DDL, rejects the multi-argument procedure, reports repeats as indexed."""

    def __init__(self) -> None:
        self.created: set[str] = set()
        self.statements: list[str] = []

    async def __call__(self, query: str, **_kwargs):
        self.statements.append(query)
        if "db.idx.fulltext.createNodeIndex" in query:
            after_map = query.rsplit("}", 1)[-1]
            args = 1 + after_map.count("'") // 2
            if args > 1:
                raise RuntimeError(
                    f"Received {args} arguments to procedure "
                    "'db.idx.fulltext.createNodeIndex', expected at most 1"
                )
        if query.startswith("CREATE FULLTEXT INDEX") and "OPTIONS" not in query:
            if query in self.created:
                return None  # graphiti-core swallows "already indexed" the same way
            self.created.add(query)
        return [], [], None


@pytest.fixture(autouse=True)
def no_server(monkeypatch: pytest.MonkeyPatch):
    """The FalkorDB client probes the server on construction; there is none here."""
    monkeypatch.setattr("graphiti_core.driver.falkordb_driver.FalkorDB", lambda **_kwargs: object())


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "graph": {"groups": ["public"]},
            "database": {"provider": "falkordb"},
            "llm": {"model": "test-model", "api_key": "synthetic"},
            "embedder": {"model": "test-embedder", "dimensions": 8, "api_key": "synthetic"},
        }
    )


def _fulltext_fields(statements: list[str]) -> set[tuple[str, str]]:
    fields = set()
    for statement in statements:
        match = re.fullmatch(
            r"CREATE FULLTEXT INDEX FOR (?:\(n:(\w+)\)|\(\)-\[e:(\w+)\]-\(\)) ON \([ne]\.(\w+)\)",
            statement,
        )
        if match:
            fields.add((match.group(1) or match.group(2), match.group(3)))
    return fields


def test_falkordb6_builds_every_fulltext_index_and_is_repeatable():
    graph = runtime.build_graphiti(_settings(), read_only=False)
    executor = Falkor6Executor()
    graph.driver.execute_query = executor

    asyncio.run(graph.build_indices_and_constraints())
    asyncio.run(graph.build_indices_and_constraints())  # a second writer on the same graph

    assert not any("db.idx.fulltext.createNodeIndex" in s for s in executor.statements)
    assert _fulltext_fields(executor.statements) == {
        ("Episodic", "content"),
        ("Episodic", "source"),
        ("Episodic", "source_description"),
        ("Episodic", "group_id"),
        ("Entity", "name"),
        ("Entity", "summary"),
        ("Entity", "group_id"),
        ("Community", "name"),
        ("Community", "group_id"),
        ("RELATES_TO", "name"),
        ("RELATES_TO", "fact"),
        ("RELATES_TO", "group_id"),
    }
    assert any(s.startswith("CREATE INDEX FOR (n:Entity)") for s in executor.statements)
    assert type(graph.driver.clone("other")) is type(graph.driver)


def test_read_only_falkordb_still_builds_nothing():
    graph = runtime.build_graphiti(_settings(), read_only=True)
    executor = Falkor6Executor()
    graph.driver.execute_query = executor
    asyncio.run(graph.build_indices_and_constraints())
    assert executor.statements == []
