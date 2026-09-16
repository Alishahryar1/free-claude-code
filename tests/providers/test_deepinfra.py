"""Tests for the DeepInfra OpenAI-chat profile and model catalog."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import DEEPINFRA_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_CATALOG_URL = "https://api.deepinfra.com/models/list"
_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_MISSING = object()


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _catalog_model(
    model_name: object = "model",
    *,
    reported_type: object = "text-generation",
    deprecated: object = None,
    tags: object = (),
) -> dict[str, object]:
    item: dict[str, object] = {}
    for field, value in (
        ("model_name", model_name),
        ("reported_type", reported_type),
        ("deprecated", deprecated),
        ("tags", tags),
    ):
        if value is not _MISSING:
            item[field] = (
                list(value) if field == "tags" and isinstance(value, tuple) else value
            )
    return item


@pytest.fixture
def deepinfra_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "deepinfra",
        make_provider_config(
            api_key="test-deepinfra-key",
            base_url=DEEPINFRA_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="deepinfra"),
    )


def test_init_uses_openai_chat_provider(
    deepinfra_provider: OpenAIChatProvider,
) -> None:
    assert deepinfra_provider._api_key == "test-deepinfra-key"
    assert deepinfra_provider._base_url == DEEPINFRA_DEFAULT_BASE
    assert deepinfra_provider._provider_name == "DEEPINFRA"


def test_build_request_body_preserves_common_chat_tools_and_images(
    deepinfra_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "system": "Be concise.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aW1hZ2U=",
                            },
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "name": "inspect_image",
                    "description": "Inspect an image",
                    "input_schema": {
                        "type": "object",
                        "properties": {"detail": {"type": "string"}},
                    },
                }
            ],
        }
    )

    body = deepinfra_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Inspect this image."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
    ]
    assert body["tools"][0]["function"]["name"] == "inspect_image"


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_OFF, "none"),
        (REASONING_ON, "high"),
        *[
            (ReasoningPolicy.on(effort=effort), effort.value)
            for effort in ReasoningEffort
        ],
    ],
)
def test_build_request_body_encodes_documented_reasoning_effort(
    deepinfra_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str,
) -> None:
    request = _request()

    body = deepinfra_provider._chat._build_request_body(request, reasoning=reasoning)

    assert body["extra_body"] == {"reasoning_effort": expected}


def test_build_request_body_preserves_extra_body_without_reasoning_override(
    deepinfra_provider: OpenAIChatProvider,
) -> None:
    request = _request(extra_body={"service_tier": "priority"})

    body = deepinfra_provider._chat._build_request_body(request, reasoning=REASONING_ON)

    assert body["extra_body"] == {
        "service_tier": "priority",
        "reasoning_effort": "high",
    }


@pytest.mark.parametrize("field", ["reasoning", "reasoning_effort"])
def test_build_request_body_rejects_caller_reasoning_override(
    deepinfra_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        deepinfra_provider._chat._build_request_body(request, reasoning=REASONING_ON)


def test_build_request_body_replays_documented_reasoning_content(
    deepinfra_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [
                {"role": "user", "content": "Solve it."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Work through it."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    body = deepinfra_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }
    assert (
        deepinfra_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_lists_only_active_text_models_with_thinking_metadata(
    deepinfra_provider: OpenAIChatProvider,
) -> None:
    deepinfra_provider._client.get = AsyncMock(
        return_value=[
            _catalog_model("reasoning-model", tags=("openai", "reasoning", "tools")),
            _catalog_model("plain-model", tags=("openai", "non-reasoning", "tools")),
            _catalog_model("hybrid-model", tags=("reasoning", "non-reasoning")),
            _catalog_model("unknown-model", tags=("openai",)),
            _catalog_model(
                "deprecated-model",
                deprecated=1_700_000_000,
                tags=("reasoning",),
            ),
            _catalog_model("image-model", reported_type="text-to-image"),
        ]
    )

    model_infos = await deepinfra_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo(
                "plain-model",
                supports_thinking=False,
                reasoning_capability=ReasoningCapability.NONE,
            ),
            ProviderModelInfo("hybrid-model", supports_thinking=True),
            ProviderModelInfo("unknown-model"),
        }
    )
    deepinfra_provider._client.get.assert_awaited_once_with(
        _CATALOG_URL,
        cast_to=object,
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_absolute_public_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[_catalog_model(_MODEL, tags=("reasoning", "tools"))],
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def build_client(*args: Any, **kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = http_client
        return AsyncOpenAI(*args, **kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        side_effect=build_client,
    ):
        provider = profiled_provider(
            "deepinfra",
            make_provider_config(
                api_key="wire-deepinfra-key",
                base_url=DEEPINFRA_DEFAULT_BASE,
                rate_limit=10,
                rate_window=60,
            ),
            admission=immediate_admission(provider_name="deepinfra"),
        )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "deepseek-ai/DeepSeek-V4-Flash",
                supports_thinking=True,
            )
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == _CATALOG_URL
    assert requests[0].headers["authorization"] == "Bearer wire-deepinfra-key"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "expected root array"),
        ([_catalog_model(model_name=_MISSING)], "include model_name"),
        ([_catalog_model(reported_type=_MISSING)], "include reported_type as str"),
        ([_catalog_model(reported_type=7)], "include reported_type as str"),
        ([_catalog_model(deprecated=_MISSING)], "include deprecated"),
        ([], "did not include any model ids"),
        (
            [_catalog_model("image", reported_type="text-to-image")],
            "did not include any model ids",
        ),
        (
            [_catalog_model("old", deprecated=1)],
            "did not include any model ids",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_malformed_or_unusable_catalog_atomically(
    deepinfra_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    deepinfra_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await deepinfra_provider.list_model_infos()


@pytest.mark.parametrize("tags", [_MISSING, None, [7], "reasoning"])
@pytest.mark.asyncio
async def test_malformed_optional_tags_leave_reasoning_unknown(
    deepinfra_provider: OpenAIChatProvider,
    tags: object,
) -> None:
    deepinfra_provider._client.get = AsyncMock(return_value=[_catalog_model(tags=tags)])

    assert await deepinfra_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("model")}
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_shared_retry_admission(
    deepinfra_provider: OpenAIChatProvider,
) -> None:
    request = httpx.Request("GET", _CATALOG_URL)
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )
    deepinfra_provider._client.get = AsyncMock(
        side_effect=[
            rate_limit_error,
            [_catalog_model(_MODEL, tags=("reasoning", "tools"))],
        ]
    )

    model_infos = await deepinfra_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "deepseek-ai/DeepSeek-V4-Flash",
                supports_thinking=True,
            )
        }
    )
    assert deepinfra_provider._client.get.await_count == 2
