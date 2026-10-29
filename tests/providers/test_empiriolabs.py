"""Tests for the EmpirioLabs AI OpenAI-chat profile and catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import EMPIRIOLABS_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
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

_MODEL = "qwen3-8-flash"
_CHAT_ENDPOINTS = [
    "POST /v1/chat/completions",
    "POST /v1/responses",
    "POST /v1/messages",
]


@pytest.fixture
def empiriolabs_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "empiriolabs",
        make_provider_config(
            api_key="test-empiriolabs-key",
            base_url=EMPIRIOLABS_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="empiriolabs", max_attempts=1),
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
    empiriolabs_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(empiriolabs_provider, OpenAIChatProvider)
    assert empiriolabs_provider._provider_name == "EMPIRIOLABS"
    assert empiriolabs_provider._api_key == "test-empiriolabs-key"
    assert empiriolabs_provider._base_url == EMPIRIOLABS_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_OFF, "none"),
        (REASONING_ON, "medium"),
        (ReasoningPolicy.on(budget_tokens=8000), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "max"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "max"),
    ],
)
def test_encodes_only_documented_reasoning_efforts(
    empiriolabs_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str | None,
) -> None:
    body = empiriolabs_provider._chat._build_request_body(
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
    empiriolabs_provider: OpenAIChatProvider,
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

    body = empiriolabs_provider._chat._build_request_body(
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
        empiriolabs_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_lists_tool_capable_chat_models_with_limits(
    empiriolabs_provider: OpenAIChatProvider,
) -> None:
    empiriolabs_provider._client.get = AsyncMock(
        return_value={
            "object": "list",
            "data": [
                {
                    "id": _MODEL,
                    "supported_endpoints": _CHAT_ENDPOINTS,
                    "features": ["reasoning", "vision", "function_calling"],
                    "capabilities": {"streaming": True, "reasoning": True},
                    "input_modalities": ["text", "image", "video"],
                    "output_modalities": ["text"],
                    "context_window": 1000000,
                    "max_output_tokens": 131072,
                },
                {
                    "id": "nova-micro-1-0",
                    "supported_endpoints": _CHAT_ENDPOINTS,
                    "features": ["function_calling"],
                    "capabilities": {"streaming": True, "reasoning": False},
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "context_window": 128000,
                    "max_output_tokens": 5000,
                },
                # Chat Completions models without tool calling are not listed.
                {
                    "id": "perplexity-sonar",
                    "supported_endpoints": ["POST /v1/chat/completions"],
                    "features": ["web_search"],
                    "capabilities": {"reasoning": False},
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                # Media and embedding models share the public catalog.
                {
                    "id": "qwen-image-3-0",
                    "supported_endpoints": ["POST /v1/images/generations"],
                    "features": [],
                    "capabilities": {"reasoning": False},
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["image"],
                },
                {
                    "id": "text-embedding-v4",
                    "supported_endpoints": ["POST /v1/embeddings"],
                    "features": ["batch"],
                    "capabilities": {"reasoning": False},
                    "input_modalities": ["text"],
                    "output_modalities": ["embedding"],
                    "context_window": 8192,
                },
                # Entries without endpoint metadata are skipped, not fatal.
                {"id": "legacy-model"},
            ],
        }
    )

    model_infos = await empiriolabs_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=1000000,
                max_output_tokens=131072,
            ),
            ProviderModelInfo(
                "nova-micro-1-0",
                reasoning_capability=ReasoningCapability.NONE,
                supports_thinking=False,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=128000,
                max_output_tokens=5000,
            ),
        }
    )
    empiriolabs_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{}]}, "include id"),
        ({"data": []}, "did not include any model ids"),
        (
            {
                "data": [
                    {
                        "id": "text-embedding-v4",
                        "supported_endpoints": ["POST /v1/embeddings"],
                        "features": [],
                    }
                ]
            },
            "did not include any model ids",
        ),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    empiriolabs_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    empiriolabs_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await empiriolabs_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    empiriolabs_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "id": _MODEL,
                        "supported_endpoints": _CHAT_ENDPOINTS,
                        "features": ["function_calling"],
                    }
                ]
            },
        )

    await empiriolabs_provider._client.close()
    empiriolabs_provider._client = AsyncOpenAI(
        api_key="wire-empiriolabs-key",
        base_url=EMPIRIOLABS_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await empiriolabs_provider.list_model_infos()
    finally:
        await empiriolabs_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.empiriolabs.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-empiriolabs-key"
