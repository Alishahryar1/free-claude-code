"""Tests for the LLM7.io OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import LLM7_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability, ReasoningPolicy
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

_MODEL = "openai/gpt-oss-120b"
_SELECTOR_IDS = frozenset({"default", "fast", "pro"})


@pytest.fixture
def llm7_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "llm7",
        make_provider_config(
            api_key="test-llm7-key",
            base_url=LLM7_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="llm7", max_attempts=1),
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


def _catalog_model(
    model_id: str = _MODEL,
    *,
    model_type: object = "chat",
    stream: object = True,
    tools_calling: object = True,
    reasoning: bool | None = None,
    input_modalities: object = ("text",),
    context_window_tokens: int | None = None,
) -> dict[str, object]:
    model: dict[str, object] = {
        "id": model_id,
        "model_type": model_type,
        "stream": stream,
        "tools_calling": tools_calling,
        "modalities": {"input": input_modalities},
    }
    if reasoning is not None:
        model["reasoning"] = reasoning
    if context_window_tokens is not None:
        model["context_window"] = {
            "tokens": context_window_tokens,
            "characters": context_window_tokens * 4,
        }
    return model


def test_constructs_standard_openai_chat_provider(
    llm7_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(llm7_provider, OpenAIChatProvider)
    assert llm7_provider._provider_name == "LLM7"
    assert llm7_provider._api_key == "test-llm7-key"
    assert llm7_provider._base_url == LLM7_DEFAULT_BASE


@pytest.mark.parametrize(
    "reasoning",
    [REASONING_DEFAULT, REASONING_ON, REASONING_OFF],
)
def test_preserves_provider_reasoning_default_and_standard_request_fields(
    llm7_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = llm7_provider._chat._build_request_body(_request(), reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_replays_reasoning_content_with_tool_history(
    llm7_provider: OpenAIChatProvider,
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

    body = llm7_provider._chat._build_request_body(
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


@pytest.mark.asyncio
async def test_filters_catalog_and_adds_live_authoritative_selectors(
    llm7_provider: OpenAIChatProvider,
) -> None:
    llm7_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _catalog_model(
                    "reasoning-model",
                    reasoning=True,
                    input_modalities=("text", "image"),
                ),
                _catalog_model("plain-model", reasoning=False),
                _catalog_model("unknown-reasoning-model"),
                _catalog_model(
                    "default",
                    reasoning=True,
                    context_window_tokens=114688,
                ),
                _catalog_model("image-generator", model_type="image"),
                _catalog_model("video-generator", model_type="video"),
                _catalog_model("nonstreaming-chat", stream=False),
                _catalog_model("chat-without-tools", tools_calling=False),
            ]
        }
    )

    model_infos = await llm7_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "reasoning-model",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "plain-model",
                supports_thinking=False,
                reasoning_capability=ReasoningCapability.NONE,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo(
                "unknown-reasoning-model",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo(
                "default",
                supports_thinking=True,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo("fast"),
            ProviderModelInfo("pro"),
        }
    )
    assert {info.model_id for info in model_infos} & _SELECTOR_IDS == _SELECTOR_IDS
    assert sum(info.model_id == "default" for info in model_infos) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", None),
        ("model_type", 7),
        ("stream", None),
        ("stream", "true"),
        ("tools_calling", None),
        ("tools_calling", "true"),
    ],
)
@pytest.mark.asyncio
async def test_rejects_missing_or_wrongly_typed_required_catalog_metadata(
    llm7_provider: OpenAIChatProvider,
    field: str,
    value: object,
) -> None:
    item = _catalog_model()
    if value is None:
        item.pop(field)
    else:
        item[field] = value
    llm7_provider._client.get = AsyncMock(return_value={"data": [item]})

    with pytest.raises(ModelListResponseError, match=f"include {field} as"):
        await llm7_provider.list_model_infos()


@pytest.mark.asyncio
async def test_wrongly_typed_optional_reasoning_metadata_is_unknown(
    llm7_provider: OpenAIChatProvider,
) -> None:
    item = _catalog_model()
    item["reasoning"] = "true"
    llm7_provider._client.get = AsyncMock(return_value={"data": [item]})

    assert await llm7_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo("default"),
            ProviderModelInfo("fast"),
            ProviderModelInfo("pro"),
        }
    )


@pytest.mark.asyncio
async def test_selectors_do_not_mask_wholly_ineligible_catalog(
    llm7_provider: OpenAIChatProvider,
) -> None:
    llm7_provider._client.get = AsyncMock(
        return_value={"data": [_catalog_model(model_type="image")]}
    )

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await llm7_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    llm7_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [_catalog_model()]})

    await llm7_provider._client.close()
    llm7_provider._client = AsyncOpenAI(
        api_key="wire-llm7-key",
        base_url=LLM7_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await llm7_provider.list_model_infos()
    finally:
        await llm7_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
        | {ProviderModelInfo(selector) for selector in _SELECTOR_IDS}
    )
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.llm7.io/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-llm7-key"
