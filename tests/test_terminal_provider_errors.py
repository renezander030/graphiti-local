"""Provider rejections must not become prompts asking the model to repair JSON."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from kg_mcp.config import Settings
from kg_mcp.runtime import build_llm


class Answer(BaseModel):
    answer: str


@pytest.fixture
async def client():
    llm = build_llm(Settings.model_validate({
        "llm": {"api_mode": "responses", "api_key": "test", "model": "test-model"},
    }))
    yield llm
    await llm.client.close()


def status_error(status):
    request = httpx.Request("POST", "https://example.invalid/v1/responses")
    response = httpx.Response(status, request=request)
    error_type = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }[status]
    return error_type("provider rejected the request", response=response, body=None)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
@pytest.mark.parametrize("structured", [True, False])
async def test_terminal_provider_error_is_raised_once(client, monkeypatch, status, structured):
    error = status_error(status)
    calls = []

    async def reject(**kwargs):
        calls.append(kwargs)
        raise error

    if structured:
        monkeypatch.setattr(client.client.responses, "parse", reject)
    else:
        monkeypatch.setattr(client.client.chat.completions, "create", reject)
    messages = [Message(role="user", content="Extract one fact.")]
    with pytest.raises(type(error)) as raised:
        await client.generate_response(messages, response_model=Answer if structured else None)
    assert raised.value is error
    assert raised.value.status_code == status
    assert len(calls) == 1
    assert len(messages) == 1


@pytest.mark.parametrize("recover", [True, False])
async def test_json_feedback_retries_are_preserved(client, monkeypatch, recover):
    import json

    calls = []

    async def reply(**kwargs):
        calls.append(kwargs)
        text = '{"answer":"ok"}' if recover and len(calls) == 2 else "{broken"
        return SimpleNamespace(output_text=text, usage=None)

    monkeypatch.setattr(client.client.responses, "parse", reply)
    messages = [Message(role="user", content="Extract one fact.")]
    if recover:
        assert await client.generate_response(messages, response_model=Answer) == {"answer": "ok"}
        assert len(calls) == 2
    else:
        with pytest.raises(json.JSONDecodeError):
            await client.generate_response(messages, response_model=Answer)
        assert len(calls) == 3
    assert len(messages) == len(calls)


@pytest.mark.parametrize("status", [429, 500])
async def test_transient_errors_keep_the_existing_retry_owner(client, monkeypatch, status):
    error = status_error(status)
    calls = []

    async def reject(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(client.client.responses, "parse", reject)
    messages = [Message(role="user", content="Extract one fact.")]
    expected = RateLimitError if status == 429 else type(error)
    with pytest.raises(expected) as raised:
        await client.generate_response(messages, response_model=Answer)
    assert len(calls) == 1
    assert len(messages) == 1
    if status == 500:
        assert raised.value is error
    else:
        assert raised.value.__cause__ is error


async def test_terminal_error_after_json_failure_preserves_explicit_cause(client, monkeypatch):
    cause = ValueError("original cause")
    error = status_error(400)
    calls = []

    async def reply(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(output_text="{broken", usage=None)
        raise error from cause

    monkeypatch.setattr(client.client.responses, "parse", reply)
    messages = [Message(role="user", content="Extract one fact.")]
    with pytest.raises(openai.BadRequestError) as raised:
        await client.generate_response(messages, response_model=Answer)
    assert raised.value is error
    assert raised.value.__cause__ is cause
    assert len(calls) == len(messages) == 2


async def test_true_refusal_stays_a_refusal(client, monkeypatch):
    from graphiti_core.llm_client.errors import RefusalError

    refusal = RefusalError("refused")
    calls = []

    async def reply(**kwargs):
        calls.append(kwargs)
        raise refusal

    monkeypatch.setattr(client.client.responses, "parse", reply)
    with pytest.raises(RefusalError) as raised:
        await client.generate_response(
            [Message(role="user", content="Fact.")], response_model=Answer,
        )
    assert raised.value is refusal
    assert len(calls) == 1


async def test_concurrent_terminal_and_json_recovery_are_independent(client, monkeypatch):
    import asyncio

    calls = {"reject": 0, "recover": 0}
    arrived = {key: asyncio.Event() for key in calls}
    error = status_error(403)

    async def reply(**kwargs):
        lane = kwargs["input"][0]["content"].split()[0]
        calls[lane] += 1
        if calls[lane] == 1:
            arrived[lane].set()
            other = "reject" if lane == "recover" else "recover"
            await asyncio.wait_for(arrived[other].wait(), timeout=2)
        if lane == "reject":
            raise error
        text = "{broken" if calls[lane] == 1 else '{"answer":"ok"}'
        return SimpleNamespace(output_text=text, usage=None)

    monkeypatch.setattr(client.client.responses, "parse", reply)
    rejected = [Message(role="user", content="reject this")]
    recovered = [Message(role="user", content="recover this")]
    results = await asyncio.gather(
        client.generate_response(rejected, response_model=Answer),
        client.generate_response(recovered, response_model=Answer),
        return_exceptions=True,
    )
    assert results[0] is error
    assert results[1] == {"answer": "ok"}
    assert calls == {"reject": 1, "recover": 2}
    assert len(rejected) == 1
    assert len(recovered) == 2
