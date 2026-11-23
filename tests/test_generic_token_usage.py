"""Provider-reported usage from the OpenAI-compatible client reaches the token tracker."""

from __future__ import annotations

import asyncio
import json

import httpx
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from kg_mcp.config import Settings


class Answer(BaseModel):
    name: str


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "graph": {"groups": ["example"]},
            "database": {"provider": "falkordb"},
            "llm": {
                "model": "test-model",
                "api_url": "http://127.0.0.1:9/v1",
                "api_key": "test-only",
                "api_mode": "chat",
            },
        }
    )


def _completion(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"name": "Ada"})},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _llm_with_transport(answers: list[dict]):
    from kg_mcp.runtime import build_llm

    llm = build_llm(_settings())
    queue = iter(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=next(queue))

    # The real OpenAI SDK client of the real generic client, sending over a fake transport.
    llm.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return llm


def _ask(llm, prompt_name: str) -> dict:
    messages = [Message(role="system", content="Extract."), Message(role="user", content="Ada")]
    return asyncio.run(
        llm.generate_response(messages, response_model=Answer, prompt_name=prompt_name)
    )


def test_generic_client_usage_reaches_the_tracker_totals():
    llm = _llm_with_transport([_completion(120, 30), _completion(80, 20)])

    assert _ask(llm, "extract_nodes") == {"name": "Ada"}
    assert _ask(llm, "extract_edges") == {"name": "Ada"}

    total = llm.token_tracker.get_total_usage()
    assert (total.input_tokens, total.output_tokens) == (200, 50)
    by_prompt = llm.token_tracker.get_usage()
    assert by_prompt["extract_nodes"].total_input_tokens == 120
    assert by_prompt["extract_nodes"].total_output_tokens == 30
    assert by_prompt["extract_edges"].call_count == 1


def test_concurrent_calls_keep_their_own_usage():
    llm = _llm_with_transport([_completion(10, 1), _completion(10, 1), _completion(10, 1)])
    messages = [Message(role="system", content="Extract."), Message(role="user", content="Ada")]

    async def run():
        await asyncio.gather(
            *(
                llm.generate_response(
                    [m.model_copy() for m in messages], response_model=Answer, prompt_name=name
                )
                for name in ("a", "b", "c")
            )
        )

    asyncio.run(run())
    usage = llm.token_tracker.get_usage()
    assert {name: (u.total_input_tokens, u.call_count) for name, u in usage.items()} == {
        "a": (10, 1),
        "b": (10, 1),
        "c": (10, 1),
    }
