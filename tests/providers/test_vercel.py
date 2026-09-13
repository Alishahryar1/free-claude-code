"""Tests for Vercel AI Gateway provider."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import VERCEL_AI_GATEWAY_DEFAULT_BASE
from free_claude_code.core.model_capabilities import ModelInputModality
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


def make_request(**overrides):
    return make_messages_request("openai/gpt-5.5", **overrides)


@pytest.fixture
def vercel_config():
    return make_provider_config(
        api_key="test_vercel_key",
        base_url=VERCEL_AI_GATEWAY_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def vercel_provider(vercel_config):
    return profiled_provider(
        "vercel",
        vercel_config,
        admission=immediate_admission(),
    )


def test_default_base_url_constant():
    assert VERCEL_AI_GATEWAY_DEFAULT_BASE == "https://ai-gateway.vercel.sh/v1"


def test_init_uses_default_base_url_and_api_key(vercel_config):
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "vercel",
            vercel_config,
            admission=immediate_admission(),
        )

    assert provider._api_key == "test_vercel_key"
    assert provider._base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(vercel_config):
    config = replace(vercel_config, base_url=f"{VERCEL_AI_GATEWAY_DEFAULT_BASE}/")

    with patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"):
        provider = profiled_provider(
            "vercel",
            config,
            admission=immediate_admission(),
        )

    assert provider._base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE


@pytest.mark.asyncio
async def test_model_catalog_extracts_modalities_and_exhaustive_reasoning_support(
    vercel_provider,
) -> None:
    vercel_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                {
                    "id": "vision-reasoning",
                    "modalities": {"input": ["text", "image"]},
                    "supported_parameters": ["tools", "reasoning"],
                },
                {
                    "id": "text-only",
                    "modalities": {"input": ["text"]},
                    "supported_parameters": ["tools"],
                },
                {
                    "id": "malformed-optional",
                    "modalities": {"input": "text"},
                    "supported_parameters": "reasoning",
                },
            ]
        )
    )

    assert await vercel_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo(
                "vision-reasoning",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "text-only",
                supports_thinking=False,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo("malformed-optional"),
        }
    )


def test_build_request_body_keeps_max_tokens(vercel_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
        }

        body = vercel_provider._chat._build_request_body(make_request())

    assert body["messages"][0].get("name") == "alice"
    assert body["max_tokens"] == 42
    assert "max_completion_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(vercel_provider):
    req = make_request(extra_body={"providerOptions": {"openai": {"reasoning": "low"}}})

    body = vercel_provider._chat._build_request_body(req)

    assert body["extra_body"] == {"providerOptions": {"openai": {"reasoning": "low"}}}


@pytest.mark.asyncio
async def test_stream_messages_text(vercel_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Vercel",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        vercel_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in vercel_provider.stream_messages(make_request())
        ]

    assert any(
        '"text_delta"' in event and "Hello from Vercel" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_messages_reasoning_content(vercel_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via gateway",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        vercel_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in vercel_provider.stream_messages(make_request())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via gateway" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(vercel_provider):
    vercel_provider._client = AsyncMock()

    await vercel_provider.cleanup()

    vercel_provider._client.close.assert_called_once()
