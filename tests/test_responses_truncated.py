"""A Responses answer cut off at max_output_tokens costs one request, as in generic mode."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
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
                "api_mode": "responses",
                "max_tokens": 50,
            },
        }
    )


def _response(text: str, status: str = "completed", reason: str | None = None) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60},
    }


def _llm(answers: list[dict], seen: list[httpx.Request]):
    from kg_mcp.runtime import build_llm

    llm = build_llm(_settings())
    queue = iter(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        seen.append(request)
        return httpx.Response(200, json=next(queue))

    llm.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return llm


def _ask(llm):
    messages = [Message(role="system", content="Extract."), Message(role="user", content="Ada")]
    return asyncio.run(llm.generate_response(messages, response_model=Answer))


def test_a_cut_off_answer_is_one_request_and_never_retried():
    from kg_mcp.ingest import _retryable
    from kg_mcp.runtime import OutputTruncatedError

    seen: list[httpx.Request] = []
    cut = _response('{"name": "Ad', status="incomplete", reason="max_output_tokens")
    llm = _llm([cut, cut, cut], seen)
    with pytest.raises(OutputTruncatedError, match=r"max_output_tokens \(50\)") as raised:
        _ask(llm)
    assert len(seen) == 1, "graphiti_core's JSON repair must not resend a cut-off answer"
    assert not _retryable(raised.value), "the ingest loop takes it as final"


def test_a_complete_answer_still_parses_in_one_request():
    seen: list[httpx.Request] = []
    llm = _llm([_response(json.dumps({"name": "Ada"}))], seen)
    assert _ask(llm) == {"name": "Ada"}
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["max_output_tokens"] == 50


def test_an_answer_incomplete_for_another_reason_keeps_its_old_path():
    seen: list[httpx.Request] = []
    llm = _llm([_response(json.dumps({"name": "Ada"}), status="incomplete", reason="content_filter")], seen)
    assert _ask(llm) == {"name": "Ada"}
    assert len(seen) == 1
