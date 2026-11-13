"""Tests for the APIKO OpenAI-chat gateway profile and catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import APIKO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
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

_MODEL = "claude-sonnet-5-5"


@pytest.fixture
def apiko_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "apiko",
        make_provider_config(
            api_key="test-apiko-key",
            base_url=APIKO_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="apiko", max_attempts=1),
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
    apiko_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(apiko_provider, OpenAIChatProvider)
    assert apiko_provider._provider_name == "APIKO"
    assert apiko_provider._api_key == "test-apiko-key"
    assert apiko_provider._base_url == APIKO_DEFAULT_BASE


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
def test_encodes_only_documented_reasoning_efforts(
    apiko_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str | None,
) -> None:
    body = apiko_provider._chat._build_request_body(
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
    apiko_provider: OpenAIChatProvider,
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

    body = apiko_provider._chat._build_request_body(
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
        apiko_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_keeps_only_chat_category_models(
    apiko_provider: OpenAIChatProvider,
) -> None:
    apiko_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": _MODEL,
                    "category": "chat",
                    "supported_endpoint_types": ["anthropic", "openai"],
                },
                {
                    "id": "deepseek-v4-pro",
                    "category": "chat",
                    "supported_endpoint_types": ["openai"],
                },
                # Task models are advertised under the openai endpoint type too.
                {
                    "id": "seedance-2.0",
                    "category": "video",
                    "supported_endpoint_types": ["openai"],
                },
                {
                    "id": "gpt-image-2",
                    "category": "image",
                    "supported_endpoint_types": ["image-generation", "openai"],
                },
            ]
        }
    )

    model_infos = await apiko_provider.list_model_infos()

    assert model_infos == frozenset(
        {ProviderModelInfo(_MODEL), ProviderModelInfo("deepseek-v4-pro")}
    )
    apiko_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
        options={"params": {"expand": "parameters"}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{"category": "chat"}]}, "include id"),
        ({"data": [{"id": _MODEL}]}, "include category"),
        (
            {"data": [{"id": "seedance-2.0", "category": "video"}]},
            "did not include any model ids",
        ),
        ({"data": []}, "did not include any model ids"),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    apiko_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    apiko_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await apiko_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    apiko_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL, "category": "chat"}]})

    await apiko_provider._client.close()
    apiko_provider._client = AsyncOpenAI(
        api_key="wire-apiko-key",
        base_url=APIKO_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await apiko_provider.list_model_infos()
    finally:
        await apiko_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.apiko.app/v1/models?expand=parameters"
    assert requests[0].headers["authorization"] == "Bearer wire-apiko-key"
