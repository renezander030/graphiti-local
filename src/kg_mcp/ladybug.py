"""LadybugDB adapter: read-only opens for readers, full-text indexes for everyone.

Ladybug accepts one read-write process at a time, plus any number of read-only
processes alongside it. Readers (the MCP server, ``kg ask``, ``kg export``) therefore
open the database read-only, so an ingest or a drain can run while the server is up.

Graphiti's Kuzu driver, which Ladybug rides on, declares its full-text indexes but its
``build_indices_and_constraints`` is a no-op, so nothing ever created them and every
hybrid search failed with a Binder exception. They are created here instead: by
``kg-ladybug-setup --apply`` and by any read-write open. A read-only open cannot create
them and refuses to start without them, naming the command that fixes it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

EXTENSIONS = ("FTS", "VECTOR")
SETUP_HINT = "run kg-ladybug-setup --database PATH --apply"
_INDEX_STATEMENT = re.compile(r"CREATE_FTS_INDEX\('([^']+)',\s*'([^']+)'")


def _alias_kuzu() -> Any:
    """Graphiti imports the predecessor package by its old name."""
    import ladybug

    sys.modules.setdefault("kuzu", ladybug)
    return ladybug


def fulltext_indexes() -> list[tuple[str, str, str]]:
    """``(table, index, statement)`` for every full-text index upstream search queries.

    Derived from graphiti's own index list so the two cannot drift apart.
    """
    _alias_kuzu()
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.graph_queries import get_fulltext_indices

    found = []
    for statement in get_fulltext_indices(GraphProvider.KUZU):
        match = _INDEX_STATEMENT.search(statement)
        if match:
            found.append((match.group(1), match.group(2), statement.strip().rstrip(";")))
    return found


def _rows(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [row for item in result for row in item.rows_as_dict()]
    return list(result.rows_as_dict())


def load_extensions(connection: Any) -> list[str]:
    """Load the search extensions; returns the names that are not installed."""
    missing = []
    for extension in EXTENSIONS:
        try:
            connection.execute(f"LOAD EXTENSION {extension}")
        except Exception:
            missing.append(extension)
    return missing


def present_indexes(connection: Any) -> set[tuple[str, str]]:
    rows = _rows(connection.execute("CALL SHOW_INDEXES() RETURN *"))
    return {(row["table_name"], row["index_name"]) for row in rows}


def missing_indexes(connection: Any) -> list[str]:
    present = present_indexes(connection)
    return [index for table, index, _ in fulltext_indexes() if (table, index) not in present]


def ensure_indexes(connection: Any) -> list[str]:
    """Create every full-text index that does not exist yet; returns the names created."""
    present = present_indexes(connection)
    created = []
    for table, index, statement in fulltext_indexes():
        if (table, index) in present:
            continue
        connection.execute(statement)
        created.append(index)
    return created


def open_database(path: str, *, read_only: bool) -> tuple[Any, Any]:
    """Open the file and load the extensions; the messages name the fix, not the symptom."""
    ladybug = _alias_kuzu()
    try:
        database = ladybug.Database(path, read_only=read_only)
    except RuntimeError as exc:
        message = str(exc)
        if read_only and "READ ONLY" in message.upper():
            raise RuntimeError(f"ladybug database not found at {path}; {SETUP_HINT}") from exc
        if "lock" in message.lower():
            raise RuntimeError(
                f"another process holds {path} open for writing (an ingest, a drain or a "
                "server older than 0.3.0); wait for it to finish"
            ) from exc
        raise
    connection = ladybug.Connection(database)
    missing = load_extensions(connection)
    if missing:
        raise RuntimeError(
            f"Ladybug extension(s) not installed: {', '.join(missing)}; {SETUP_HINT}"
        )
    return database, connection


def signature(path: str) -> tuple[Any, ...]:
    """What a read-only handle saw at open time: the file and its write-ahead log."""
    parts: list[Any] = []
    for candidate in (path, f"{path}.wal"):
        try:
            stat = os.stat(candidate)
        except FileNotFoundError:
            parts.append(None)
        else:
            parts.append((stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


def inspect_database(path: str) -> dict[str, Any]:
    """Read-only look at a database for ``kg doctor``: extensions and indexes."""
    ladybug = _alias_kuzu()
    database = ladybug.Database(path, read_only=True)
    connection = ladybug.Connection(database)
    try:
        missing_extensions = load_extensions(connection)
        if missing_extensions:
            return {"missing_extensions": missing_extensions, "missing_indexes": []}
        return {"missing_extensions": [], "missing_indexes": missing_indexes(connection)}
    finally:
        connection.close()
        database.close()


def build_ladybug_driver(path: str, *, read_only: bool = False):
    ladybug = _alias_kuzu()
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    class LadybugDriver(KuzuDriver):
        def __init__(self, database_path: str, *, read_only: bool):
            self.path = os.path.expanduser(database_path)
            self.read_only = read_only
            # KuzuDriver.__init__ wires the driver's operation objects, but it also opens
            # the path read-write and runs schema DDL, which a read-only open refuses. Wire
            # the operations against a throwaway in-memory database, then attach the file.
            super().__init__(db=":memory:")
            self.client.close()
            self._attach()

        def _attach(self) -> None:
            database, connection = open_database(self.path, read_only=self.read_only)
            try:
                if self.read_only:
                    missing = missing_indexes(connection)
                    if missing:
                        raise RuntimeError(
                            f"ladybug database {self.path} has no full-text index for "
                            f"{', '.join(missing)}; {SETUP_HINT}"
                        )
                else:
                    self.db = database
                    self.setup_schema()
                    ensure_indexes(connection)
            finally:
                connection.close()
            self.db = database
            self.opened = signature(self.path)
            self.client = ladybug.AsyncConnection(self.db, max_concurrent_queries=1)

        def reopen_if_changed(self) -> bool:
            """Pick up commits made by another process since this handle was opened.

            A read-only handle sees the database as it was when it was opened; a drain
            that lands facts is invisible to a running server until the file is reopened.
            The previous handle is not closed explicitly: a query still awaiting on it
            keeps it alive, and it is released when that query finishes.
            """
            if not self.read_only:
                return False
            current = signature(self.path)
            if current == self.opened:
                return False
            self._attach()
            return True

        def clone(self, database: str | None = None, **kwargs: Any):
            del database, kwargs
            return self

        async def execute_query(self, cypher_query_: str, **kwargs: Any):
            kwargs.pop("database_", None)
            kwargs.pop("routing_", None)
            results = await self.client.execute(cypher_query_, parameters=kwargs)
            if not results:
                return [], None, None
            return _rows(results), None, None

        async def close(self) -> None:
            # Release the file promptly: a writer waiting on the lock should not have to
            # wait for garbage collection.
            for handle in (self.client, self.db):
                try:
                    handle.close()
                except Exception:
                    continue

    return LadybugDriver(path, read_only=read_only)


def setup_main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a LadybugDB graph: search extensions, schema and full-text indexes"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    target = args.database.expanduser().resolve()
    if not args.apply:
        print(
            "would install the FTS and VECTOR extensions, create the schema and the "
            f"full-text indexes for {target}"
        )
        print("dry run only; add --apply in a trusted network-enabled environment")
        return

    ladybug = _alias_kuzu()
    from graphiti_core.driver.kuzu_driver import SCHEMA_QUERIES

    target.parent.mkdir(parents=True, exist_ok=True)
    database = ladybug.Database(str(target))
    connection = ladybug.Connection(database)
    for extension in EXTENSIONS:
        connection.execute(f"INSTALL {extension}")
        connection.execute(f"LOAD EXTENSION {extension}")
        print(f"{extension}: installed and loaded")
    connection.execute(SCHEMA_QUERIES)
    created = ensure_indexes(connection)
    print("schema: ready")
    print(f"full-text indexes created: {', '.join(created) or 'none (already present)'}")
    connection.close()
    database.close()
