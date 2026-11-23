"""The LLM client never changes the messages its caller passes in.

graphiti-core appends schema, language and retry text to the message list it is
given. Repeating a call with the same list then sends a longer prompt each time,
so identical calls stop being identical requests. These tests drive the real
clients built by ``build_llm`` over a fake HTTP transport; nothing leaves the host.
"""

from __future__ import annotations

import asyncio
import json

import httpx
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from kg_mcp.config import Settings
from kg_mcp.runtime import build_llm


class Answer(BaseModel):
    name: str


def _settings(api_mode: str) -> Settings:
    return Settings.model_validate(
        {
            "graph": {"groups": ["example"]},
            "database": {"provider": "falkordb"},
            "llm": {
                "model": "test-model",
                "api_url": "http://127.0.0.1:9/v1",
                "api_key": "local-test",
                "api_mode": api_mode,
                "structured_output_mode": "json_object",
            },
            "embedder": {"model": "test", "dimensions": 8},
        }
    )


def _chat_reply(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _responses_reply(text: str) -> dict:
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg-test",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _install_transport(llm, replies):
    """Route the client's HTTP traffic to canned replies and record each request body."""
    bodies: list[dict] = []
    answers = iter(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=next(answers))

    llm.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return bodies


def _caller_messages() -> list[Message]:
    return [
        Message(role="system", content="You extract names."),
        Message(role="user", content="Ada wrote the first program."),
    ]


def _snapshot(messages: list[Message]) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in messages]


def test_generic_client_leaves_caller_messages_alone_and_repeats_the_same_request():
    llm = build_llm(_settings("chat"))
    bodies = _install_transport(llm, [_chat_reply('{"name": "Ada"}')] * 2)
    messages = _caller_messages()
    before = _snapshot(messages)

    async def call():
        return await llm.generate_response(
            messages, response_model=Answer, group_id="example", attribute_extraction=True
        )

    assert asyncio.run(call()) == {"name": "Ada"}
    assert _snapshot(messages) == before
    assert asyncio.run(call()) == {"name": "Ada"}
    assert _snapshot(messages) == before
    assert len(bodies) == 2
    assert bodies[0]["messages"] == bodies[1]["messages"]
    # The schema and the extraction framing still reach the model.
    assert "Respond with a JSON object" in bodies[0]["messages"][-1]["content"]
    assert "ATTRIBUTE EXTRACTION" in bodies[0]["messages"][0]["content"]


def test_responses_client_keeps_retry_text_out_of_caller_messages():
    llm = build_llm(_settings("responses"))
    bodies = _install_transport(
        llm,
        [
            _responses_reply("not json"),
            _responses_reply('{"name": "Ada"}'),
            _responses_reply('{"name": "Ada"}'),
        ],
    )
    messages = _caller_messages()
    before = _snapshot(messages)

    async def call():
        return await llm.generate_response(messages, response_model=Answer, group_id="example")

    # The first reply is invalid, so the client retries with a correction appended.
    assert asyncio.run(call()) == {"name": "Ada"}
    assert len(bodies) == 2
    assert len(bodies[1]["input"]) == len(bodies[0]["input"]) + 1
    assert _snapshot(messages) == before

    # The next identical call sends exactly the first request again.
    assert asyncio.run(call()) == {"name": "Ada"}
    assert bodies[2]["input"] == bodies[0]["input"]
    assert _snapshot(messages) == before
