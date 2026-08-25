"""LadybugDB adapter and explicit extension setup command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def build_ladybug_driver(path: str):
    import ladybug

    # Graphiti's embedded driver imports the predecessor package by this name.
    sys.modules.setdefault("kuzu", ladybug)
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    class LadybugDriver(KuzuDriver):
        def __init__(self, database_path: str):
            super().__init__(db=os.path.expanduser(database_path))
            connection = ladybug.Connection(self.db)
            missing: list[str] = []
            for extension in ("FTS", "VECTOR"):
                try:
                    connection.execute(f"LOAD EXTENSION {extension}")
                except Exception:
                    missing.append(extension)
            if missing:
                names = ", ".join(missing)
                raise RuntimeError(
                    f"Ladybug extension(s) not installed: {names}; "
                    "run kg-ladybug-setup --database PATH --apply"
                )

        def clone(self, database: str | None = None, **kwargs: Any):
            del database, kwargs
            return self

        async def execute_query(self, cypher_query_: str, **kwargs: Any):
            kwargs.pop("database_", None)
            kwargs.pop("routing_", None)
            results = await self.client.execute(cypher_query_, parameters=kwargs)
            if not results:
                return [], None, None
            if isinstance(results, list):
                rows = [item for result in results for item in result.rows_as_dict()]
            else:
                rows = list(results.rows_as_dict())
            return rows, None, None

    return LadybugDriver(path)


def setup_main() -> None:
    parser = argparse.ArgumentParser(description="Install LadybugDB search extensions")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    target = args.database.expanduser().resolve()
    if not args.apply:
        print(f"would install FTS and VECTOR extensions for {target}")
        print("dry run only; add --apply in a trusted network-enabled environment")
        return

    import ladybug

    target.parent.mkdir(parents=True, exist_ok=True)
    database = ladybug.Database(str(target))
    connection = ladybug.Connection(database)
    for extension in ("FTS", "VECTOR"):
        connection.execute(f"INSTALL {extension}")
        connection.execute(f"LOAD EXTENSION {extension}")
        print(f"{extension}: installed and loaded")

