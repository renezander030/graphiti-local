"""Check a stdio server's actual MCP handshake and six-tool read interface.

Usage: uv run python scripts/mcp_smoke.py COMMAND [ARG ...]
The default check needs a prepared database, but does not call an LLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {
    "get_entity_edge",
    "get_episode_entities",
    "get_episodes",
    "get_status",
    "search_memory_facts",
    "search_nodes",
}


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: mcp_smoke.py COMMAND [ARG ...]")
    parameters = StdioServerParameters(
        command=sys.argv[1], args=sys.argv[2:], env=dict(os.environ)
    )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}
        if names != EXPECTED:
            raise AssertionError(f"unexpected tools: {sorted(names)}")
        status = await session.call_tool("get_status", {})
        payload = status.structuredContent or {}
        if status.isError or payload.get("status") != "ok":
            raise AssertionError(status)
        print(json.dumps({"tools": sorted(names), "status": status.model_dump(mode="json")}))


if __name__ == "__main__":
    asyncio.run(main())
