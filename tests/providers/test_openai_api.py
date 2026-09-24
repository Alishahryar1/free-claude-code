"""Tests for the official OpenAI API provider profile."""

from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import OPENAI_API_DEFAULT_BASE
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)

_MODEL = "gpt-4o"


def _provider(
    *,
    api_key: str = "test-openai-key",
    base_url: str = OPENAI_API_DEFAULT_BASE,
) -> OpenAIChatProvider:
    return profiled_provider(
        "openai_api",
        make_provider_config(
            api_key=api_key,
            base_url=base_url,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="openai_api"),
    )


def test_constructs_openai_api_provider() -> None:
    provider = _provider()
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._provider_name == "OPENAI_API"
    assert provider._api_key == "test-openai-key"
    assert provider._base_url == OPENAI_API_DEFAULT_BASE


def test_init_uses_api_key_and_base_url() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = _provider(api_key="custom-key", base_url="https://api.openai.com/v1")

    assert provider._provider_name == "OPENAI_API"
    assert provider._api_key == "custom-key"
    assert provider._base_url == "https://api.openai.com/v1"
    assert openai_client.call_args.kwargs["api_key"] == "custom-key"
    assert openai_client.call_args.kwargs["base_url"] == "https://api.openai.com/v1"


def test_normalizes_base_url_without_v1() -> None:
    provider = _provider(base_url="https://api.openai.com")
    assert provider._base_url == "https://api.openai.com/v1"


def test_request_uses_modern_token_field_tools_and_reasoning() -> None:
    request = make_messages_request(
        _MODEL,
        temperature=None,
        top_p=None,
        tools=[
            {
                "name": "read_file",
                "description": "Read one file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )

    body = _provider()._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )

    assert body["model"] == _MODEL
    assert body["messages"][0] == {"role": "system", "content": "System prompt"}
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "high"
    assert body["tools"][0]["function"]["name"] == "read_file"


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy.off(), "none"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ],
)
def test_reasoning_uses_openai_supported_vocabulary(
    policy: ReasoningPolicy,
    expected: str,
) -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            _MODEL,
            temperature=None,
            top_p=None,
        ),
        reasoning=policy,
    )

    assert body["reasoning_effort"] == expected


def test_reasoning_history_replays_as_portable_think_tags() -> None:
    request = make_messages_request(
        _MODEL,
        temperature=None,
        top_p=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the repository."},
                    {"type": "text", "text": "I found the file."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ],
    )

    body = _provider()._chat._build_request_body(request)
    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )

    assert assistant["content"] == (
        "<think>\nInspect the repository.\n</think>\n\nI found the file."
    )
    assert "reasoning_content" not in assistant
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_lists_models_from_models_endpoint() -> None:
    provider = _provider()
    provider._client.get = AsyncMock(
        return_value={"data": [{"id": "gpt-4o"}, {"id": "o3-mini"}]}
    )

    model_infos = await provider.list_model_infos()

    assert model_infos == frozenset(
        {ProviderModelInfo("gpt-4o"), ProviderModelInfo("o3-mini")}
    )
    provider._client.get.assert_awaited_once()
    call = provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
async def test_rejects_empty_model_ids() -> None:
    provider = _provider()
    provider._client.get = AsyncMock(return_value={"data": [{"id": ""}]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth() -> None:
    provider = _provider()
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL}]})

    await provider._client.close()
    provider._client = AsyncOpenAI(
        api_key="wire-openai-key",
        base_url=OPENAI_API_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-openai-key"
