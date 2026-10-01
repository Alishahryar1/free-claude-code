"""Tests for the FutureInfra OpenAI-chat router profile and catalog."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import FUTUREINFRA_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "openai/gpt-4o-mini"


@pytest.fixture
def futureinfra_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "futureinfra",
        make_provider_config(
            api_key="test-futureinfra-key",
            base_url=FUTUREINFRA_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="futureinfra", max_attempts=1),
    )


def _request(**overrides: JsonValue) -> MessagesRequest:
    payload: JsonObject = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Inspect the file."}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def test_constructs_standard_openai_chat_provider(
    futureinfra_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(futureinfra_provider, OpenAIChatProvider)
    assert futureinfra_provider._provider_name == "FUTUREINFRA"
    assert futureinfra_provider._api_key == "test-futureinfra-key"
    assert futureinfra_provider._base_url == FUTUREINFRA_DEFAULT_BASE


@pytest.mark.parametrize(
    "reasoning",
    [REASONING_DEFAULT, REASONING_ON, REASONING_OFF],
)
def test_preserves_standard_fields_and_omits_reasoning_controls(
    futureinfra_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = futureinfra_provider._chat._build_request_body(
        _request(), reasoning=reasoning
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_tool_history_keeps_reasoning_as_plain_text(
    futureinfra_provider: OpenAIChatProvider,
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

    body = futureinfra_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assistant = body["messages"][1]
    assert (
        assistant["content"]
        == "[Earlier reasoning]\nRead it first.\n\nI will inspect it."
    )
    assert assistant["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "example.py"}',
            },
        }
    ]
    assert "reasoning_content" not in assistant
    assert "reasoning" not in assistant
    assert body["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }


@pytest.mark.asyncio
async def test_model_catalog_keeps_text_models_with_context_length(
    futureinfra_provider: OpenAIChatProvider,
) -> None:
    futureinfra_provider._client.get = AsyncMock(
        return_value={
            "currency": "KRW",
            "data": [
                {"id": _MODEL, "kind": "text", "context_length": 128_000},
                {
                    "id": "deepseek/deepseek-chat",
                    "kind": "text",
                    "context_length": 163_840,
                },
                {"id": "openai/text-embedding-3-small", "kind": "embeddings"},
                {"id": "openai/gpt-image-1", "kind": "image"},
            ],
        }
    )

    model_infos = await futureinfra_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(_MODEL, context_window_tokens=128_000),
            ProviderModelInfo("deepseek/deepseek-chat", context_window_tokens=163_840),
        }
    )
    futureinfra_provider._client.get.assert_awaited_once()
    call = futureinfra_provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{"kind": "text"}]}, "include id"),
        ({"data": []}, "did not include any model ids"),
    ],
)
async def test_rejects_malformed_or_empty_catalog(
    futureinfra_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    futureinfra_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await futureinfra_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    futureinfra_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={"currency": "KRW", "data": [{"id": _MODEL, "kind": "text"}]},
        )

    await futureinfra_provider._client.close()
    futureinfra_provider._client = AsyncOpenAI(
        api_key="wire-futureinfra-key",
        base_url=FUTUREINFRA_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await futureinfra_provider.list_model_infos()
    finally:
        await futureinfra_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://futureinfra.ai/v1/ai/models"
    assert requests[0].headers["authorization"] == "Bearer wire-futureinfra-key"
