"""Tests for Cohere Compatibility API provider."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.provider_catalog import COHERE_DEFAULT_BASE
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


def make_request(**overrides):
    return make_messages_request("command-a-plus-05-2026", **overrides)


@pytest.fixture
def cohere_config():
    return make_provider_config(
        api_key="test_cohere_key",
        base_url=COHERE_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def cohere_provider(cohere_config):
    return profiled_provider("cohere", cohere_config, admission=immediate_admission())


def test_default_base_url_constant():
    assert COHERE_DEFAULT_BASE == "https://api.cohere.ai/compatibility/v1"


def test_init_uses_default_base_url_and_api_key(cohere_config):
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "cohere", cohere_config, admission=immediate_admission()
        )

    assert provider._api_key == "test_cohere_key"
    assert provider._base_url == COHERE_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(cohere_config):
    config = replace(cohere_config, base_url=f"{COHERE_DEFAULT_BASE}/")

    with patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"):
        provider = profiled_provider("cohere", config, admission=immediate_admission())

    assert provider._base_url == COHERE_DEFAULT_BASE


def test_build_request_body_sanitizes_documented_unsupported_fields(cohere_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "command-a-plus-05-2026",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
            "store": True,
            "metadata": {"trace": "abc"},
            "logit_bias": {"1": -100},
            "top_logprobs": 2,
            "n": 4,
            "modalities": ["text"],
            "prediction": {"type": "content", "content": "x"},
            "audio": {"voice": "alloy"},
            "service_tier": "auto",
            "parallel_tool_calls": True,
        }

        body = cohere_provider._chat._build_request_body(make_request())

    assert body["messages"][0].get("name") is None
    assert body["max_tokens"] == 42
    assert "max_completion_tokens" not in body
    for key in (
        "audio",
        "logit_bias",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "service_tier",
        "store",
        "top_logprobs",
    ):
        assert key not in body


def test_build_request_body_maps_reasoning_on_to_high(cohere_provider):
    request = make_request()
    body = cohere_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "high"


def test_build_request_body_preserves_replayed_reasoning_content(cohere_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "command-a-plus-05-2026",
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "hidden chain",
                }
            ],
        }

        request = make_request()
        body = cohere_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "hidden chain",
        }
    ]
    assert body["reasoning_effort"] == "high"


def test_build_request_body_maps_reasoning_off_to_none():
    provider = profiled_provider(
        "cohere",
        make_provider_config(
            api_key="test_cohere_key",
            base_url=COHERE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )

    request = make_request(thinking={"type": "disabled"})
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))

    assert body["reasoning_effort"] == "none"


def test_build_request_body_promotes_allowed_extra_body(cohere_provider):
    req = make_request(
        extra_body={
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "response_format": {"type": "json_object"},
            "seed": 123,
        }
    )

    body = cohere_provider._chat._build_request_body(req, reasoning=reasoning_for(req))

    assert body["frequency_penalty"] == 0.1
    assert body["presence_penalty"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 123
    assert "extra_body" not in body


def test_build_request_body_rejects_unsupported_extra_body(cohere_provider):
    req = make_request(extra_body={"documents": [{"text": "x"}]})

    with pytest.raises(InvalidRequestError, match="Unsupported"):
        cohere_provider._chat._build_request_body(req, reasoning=reasoning_for(req))


@pytest.mark.asyncio
async def test_stream_messages_text(cohere_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Cohere",
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
        cohere_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in cohere_provider.stream_messages(make_request())
        ]

    assert any(
        '"text_delta"' in event and "Hello from Cohere" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_messages_tool_call(cohere_provider):
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function = MagicMock()
    mock_tc.function.name = "Read"
    mock_tc.function.arguments = '{"file_path":"a.py"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content=None, tool_calls=[mock_tc]),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        cohere_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in cohere_provider.stream_messages(make_request())
        ]

    assert any(
        '"content_block_start"' in event and '"tool_use"' in event for event in events
    )
    assert any(
        '"input_json_delta"' in event and "file_path" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_messages_reasoning_content(cohere_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via Cohere",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        cohere_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in cohere_provider.stream_messages(make_request())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via Cohere" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(cohere_provider):
    cohere_provider._client = AsyncMock()

    await cohere_provider.cleanup()

    cohere_provider._client.close.assert_called_once()
