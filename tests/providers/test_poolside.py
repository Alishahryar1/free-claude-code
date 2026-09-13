"""Tests for the Poolside AI OpenAI-chat provider profile."""

from types import SimpleNamespace

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import POOLSIDE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "poolside/laguna-s-2.1"


@pytest.fixture
def poolside_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "poolside",
        make_provider_config(
            api_key="test-poolside-key",
            base_url=POOLSIDE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="poolside", max_attempts=1),
    )


def _request(**overrides: JsonValue) -> MessagesRequest:
    payload: JsonObject = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("reasoning", "enabled"),
    [(REASONING_OFF, False), (REASONING_ON, True)],
)
def test_encodes_documented_chat_template_thinking_control(
    poolside_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    enabled: bool,
) -> None:
    body = poolside_provider._chat._build_request_body(
        _request(extra_body={"top_k": 20}),
        reasoning=reasoning,
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": enabled},
    }


def test_omits_thinking_control_for_poolside_default(
    poolside_provider: OpenAIChatProvider,
) -> None:
    body = poolside_provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "extra_body" not in body


@pytest.mark.parametrize(
    "field",
    ["reasoning", "reasoning_effort", "thinking", "enable_thinking"],
)
def test_rejects_caller_reasoning_override(
    poolside_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        poolside_provider._chat._build_request_body(request, reasoning=REASONING_ON)


def test_replays_reasoning_content_with_tool_history(
    poolside_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "Inspect the file."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Read it first."},
                    {"type": "text", "text": "I will inspect it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "example.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "print('hello')",
                    }
                ],
            },
        ]
    )

    body = poolside_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "I will inspect it.",
        "reasoning_content": "Read it first.",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "example.py"}',
                },
            }
        ],
    }
    assert body["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }
    assert (
        poolside_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_thinking_control_reaches_wire_as_top_level_extension(
    poolside_provider: OpenAIChatProvider,
) -> None:
    body = poolside_provider._chat._build_request_body(
        _request(),
        reasoning=REASONING_OFF,
    )

    wire = await capture_openai_chat_wire_body(body)

    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in wire


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    poolside_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": _MODEL,
                        "object": "model",
                        "created": 0,
                        "owned_by": "poolside",
                    }
                ],
            },
        )

    await poolside_provider._client.close()
    poolside_provider._client = AsyncOpenAI(
        api_key="wire-poolside-key",
        base_url=POOLSIDE_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await poolside_provider.list_model_infos()
    finally:
        await poolside_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://inference.poolside.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-poolside-key"
