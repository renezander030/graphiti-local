import os

import pytest

from kg_mcp.server import create_server

EXPECTED_TOOLS = {
    "get_entity_edge",
    "get_episode_entities",
    "get_episodes",
    "get_status",
    "search_memory_facts",
    "search_nodes",
}


@pytest.mark.asyncio
async def test_registry_contains_only_six_read_tools():
    tools = await create_server().list_tools()
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


def test_telemetry_is_disabled_before_graphiti_import():
    assert os.environ["GRAPHITI_TELEMETRY_ENABLED"] == "false"

