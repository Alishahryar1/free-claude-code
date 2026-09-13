"""Tests for Cloudflare Workers AI OpenAI-compatible chat provider."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import CLOUDFLARE_AI_REST_ROOT
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.cloudflare import (
    CloudflareProvider,
    cloudflare_ai_base_url,
)
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    reasoning_for,
)

_ACCOUNT_ID = "account-123"
_BASE_URL = f"{CLOUDFLARE_AI_REST_ROOT}/accounts/{_ACCOUNT_ID}/ai/v1"
_MODEL_SEARCH_URL = f"{CLOUDFLARE_AI_REST_ROOT}/accounts/{_ACCOUNT_ID}/ai/models/search"


@pytest.fixture
def cloudflare_config() -> ProviderConfig:
    return make_provider_config(
        api_key="test-cloudflare-token",
        base_url=CLOUDFLARE_AI_REST_ROOT,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def cloudflare_provider(cloudflare_config: ProviderConfig) -> CloudflareProvider:
    return CloudflareProvider(
        cloudflare_config,
        account_id=_ACCOUNT_ID,
        admission=immediate_admission(),
    )


def _request(model: str = "@cf/moonshotai/kimi-k2.6") -> MessagesRequest:
    return MessagesRequest(
        model=model,
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
    )


def _chunk(delta: SimpleNamespace, *, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=5, prompt_tokens=8),
    )


async def _stream(*chunks: SimpleNamespace) -> AsyncIterator[SimpleNamespace]:
    for chunk in chunks:
        yield chunk


def test_cloudflare_ai_base_url_uses_account_scoped_openai_chat_root() -> None:
    assert cloudflare_ai_base_url(CLOUDFLARE_AI_REST_ROOT, "account/with slash") == (
        f"{CLOUDFLARE_AI_REST_ROOT}/accounts/account%2Fwith%20slash/ai/v1"
    )


def test_missing_account_id_raises_authentication_error(
    cloudflare_config: ProviderConfig,
) -> None:
    with pytest.raises(ApplicationUnavailableError, match="CLOUDFLARE_ACCOUNT_ID"):
        CloudflareProvider(
            cloudflare_config, account_id=" ", admission=immediate_admission()
        )


def test_init_composes_account_scoped_openai_chat_base_url(
    cloudflare_config: ProviderConfig,
) -> None:
    with (
        patch(
            "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
        ) as mock_openai,
        patch("httpx.AsyncClient") as mock_httpx_client,
    ):
        provider = CloudflareProvider(
            cloudflare_config,
            account_id=_ACCOUNT_ID,
            admission=immediate_admission(),
        )

    assert provider._api_key == "test-cloudflare-token"
    assert provider._base_url == _BASE_URL
    assert provider._model_search_url == _MODEL_SEARCH_URL
    assert provider._provider_name == "CLOUDFLARE"
    assert mock_openai.call_args.kwargs["base_url"] == _BASE_URL
    assert mock_openai.call_args.kwargs["api_key"] == "test-cloudflare-token"
    assert mock_httpx_client.called


def test_model_list_headers_use_bearer_auth(
    cloudflare_provider: CloudflareProvider,
) -> None:
    assert cloudflare_provider._model_list_headers() == {
        "Authorization": "Bearer test-cloudflare-token"
    }


def test_build_request_body_preserves_literal_cf_model_id_and_controls_thinking(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "@cf/moonshotai/kimi-k2.6",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = cloudflare_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "@cf/moonshotai/kimi-k2.6"
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["extra_body"]["chat_template_kwargs"]["thinking"] is True


def test_build_request_body_disabled_thinking_sets_cloudflare_template_flag(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "@cf/moonshotai/kimi-k2.6",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
        }
    )

    body = cloudflare_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["chat_template_kwargs"]["thinking"] is False


def test_build_request_body_preserves_user_extra_body_without_overriding_thinking(
    cloudflare_provider: CloudflareProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "@cf/moonshotai/kimi-k2.6",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"chat_template_kwargs": {"thinking": False}},
        }
    )

    body = cloudflare_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["chat_template_kwargs"]["thinking"] is False


@pytest.mark.asyncio
async def test_lists_models_from_cloudflare_model_search_endpoint(
    cloudflare_provider: CloudflareProvider,
) -> None:
    response = httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": "@cf/moonshotai/kimi-k2.6", "object": "model"},
                {"id": "@cf/meta/llama-4-scout-17b-16e-instruct", "object": "model"},
            ],
        },
        request=httpx.Request("GET", _MODEL_SEARCH_URL),
    )
    with patch.object(
        cloudflare_provider._model_list_client,
        "get",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_get:
        assert await cloudflare_provider.list_model_infos() == frozenset(
            {
                ProviderModelInfo("@cf/moonshotai/kimi-k2.6"),
                ProviderModelInfo("@cf/meta/llama-4-scout-17b-16e-instruct"),
            }
        )

    mock_get.assert_awaited_once_with(
        _MODEL_SEARCH_URL,
        params={"format": "openrouter"},
        headers={"Authorization": "Bearer test-cloudflare-token"},
    )


@pytest.mark.asyncio
async def test_stream_uses_openai_chat_completions(
    cloudflare_provider: CloudflareProvider,
) -> None:
    delta = SimpleNamespace(
        content="Hello from Cloudflare",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )

    with patch.object(
        cloudflare_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(_chunk(delta)),
    ) as mock_create:
        events = [
            event async for event in cloudflare_provider.stream_messages(_request())
        ]

    parsed = parse_sse_text("".join(events))
    assert any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("text") == "Hello from Cloudflare"
        for event in parsed
    )
    assert mock_create.call_args.kwargs["model"] == "@cf/moonshotai/kimi-k2.6"
    assert mock_create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_maps_cloudflare_reasoning_delta_to_thinking(
    cloudflare_provider: CloudflareProvider,
) -> None:
    delta = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning="Cloudflare reasoning",
        tool_calls=None,
    )

    with patch.object(
        cloudflare_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(_chunk(delta)),
    ):
        events = [
            event async for event in cloudflare_provider.stream_messages(_request())
        ]

    parsed = parse_sse_text("".join(events))
    assert any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("thinking") == "Cloudflare reasoning"
        for event in parsed
    )


@pytest.mark.asyncio
async def test_stream_maps_openai_tool_calls_to_tool_use(
    cloudflare_provider: CloudflareProvider,
) -> None:
    tool_call = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="echo", arguments='{"value":"x"}'),
    )
    delta = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[tool_call],
    )
    request = MessagesRequest.model_validate(
        {
            "model": "@cf/moonshotai/kimi-k2.6",
            "messages": [{"role": "user", "content": "Use the tool"}],
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo a value",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                }
            ],
        }
    )

    with patch.object(
        cloudflare_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(_chunk(delta, finish_reason="tool_calls")),
    ):
        events = [event async for event in cloudflare_provider.stream_messages(request)]

    parsed = parse_sse_text("".join(events))
    assert any(
        event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "tool_use"
        and event.data.get("content_block", {}).get("name") == "echo"
        for event in parsed
    )
    assert any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("partial_json") == '{"value":"x"}'
        for event in parsed
    )
