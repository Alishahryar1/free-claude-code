"""Tests for the TrustedRouter OpenAI-chat gateway profile and catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import TRUSTEDROUTER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
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

_MODEL = "anthropic/claude-sonnet-4.5"


@pytest.fixture
def trustedrouter_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "trustedrouter",
        make_provider_config(
            api_key="test-trustedrouter-key",
            base_url=TRUSTEDROUTER_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="trustedrouter", max_attempts=1),
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


def _chat_model(model_id: str, **fields: JsonValue) -> JsonObject:
    item: JsonObject = {"id": model_id, "trustedrouter": {"supports_chat": True}}
    item.update(fields)
    return item


def test_constructs_standard_openai_chat_provider(
    trustedrouter_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(trustedrouter_provider, OpenAIChatProvider)
    assert trustedrouter_provider._provider_name == "TRUSTEDROUTER"
    assert trustedrouter_provider._api_key == "test-trustedrouter-key"
    assert trustedrouter_provider._base_url == TRUSTEDROUTER_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_OFF, None),
        (REASONING_ON, "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ],
)
def test_encodes_reasoning_effort(
    trustedrouter_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str | None,
) -> None:
    body = trustedrouter_provider._chat._build_request_body(
        _request(),
        reasoning=reasoning,
    )

    assert body.get("reasoning_effort") == expected
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "thinking" not in body
    assert "extra_body" not in body


def test_replays_reasoning_content_with_tool_history(
    trustedrouter_provider: OpenAIChatProvider,
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

    body = trustedrouter_provider._chat._build_request_body(
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
        trustedrouter_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_maps_thinking_limits_and_skips_non_chat_models(
    trustedrouter_provider: OpenAIChatProvider,
) -> None:
    trustedrouter_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _chat_model(
                    "trustedrouter/auto",
                    context_length=1048576,
                    architecture={"input_modalities": ["text", "image"]},
                    top_provider={"max_completion_tokens": None},
                    supported_parameters=["tools", "tool_choice", "max_tokens"],
                ),
                _chat_model(
                    _MODEL,
                    context_length=1000000,
                    architecture={"input_modalities": ["text", "image"]},
                    top_provider={"max_completion_tokens": 64000},
                    supported_parameters=["tools", "reasoning", "reasoning_effort"],
                ),
                {
                    "id": "openai/text-embedding-3-small",
                    "context_length": 8192,
                    "architecture": {"input_modalities": ["text"]},
                    "trustedrouter": {"supports_chat": False},
                },
            ]
        }
    )

    model_infos = await trustedrouter_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "trustedrouter/auto",
                supports_thinking=False,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=1048576,
            ),
            ProviderModelInfo(
                _MODEL,
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=1000000,
                max_output_tokens=64000,
            ),
        }
    )
    trustedrouter_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{}]}, "include id"),
        ({"data": [{"id": _MODEL}]}, "trustedrouter.supports_chat"),
        ({"data": []}, "did not include any model ids"),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    trustedrouter_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    trustedrouter_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await trustedrouter_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    trustedrouter_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [_chat_model(_MODEL)]})

    await trustedrouter_provider._client.close()
    trustedrouter_provider._client = AsyncOpenAI(
        api_key="wire-trustedrouter-key",
        base_url=TRUSTEDROUTER_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await trustedrouter_provider.list_model_infos()
    finally:
        await trustedrouter_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.trustedrouter.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-trustedrouter-key"
