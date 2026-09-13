"""Tests for SambaNova Cloud (OpenAI-compatible) provider."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import SAMBANOVA_DEFAULT_BASE
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


def make_request(**overrides):
    return make_messages_request("Meta-Llama-3.3-70B-Instruct", **overrides)


@pytest.fixture
def sambanova_config():
    return make_provider_config(
        api_key="test_sambanova_key",
        base_url=SAMBANOVA_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def sambanova_provider(sambanova_config):
    return profiled_provider(
        "sambanova", sambanova_config, admission=immediate_admission()
    )


def test_default_base_url_constant():
    assert SAMBANOVA_DEFAULT_BASE == "https://api.sambanova.ai/v1"


def test_init_uses_default_base_url_and_api_key(sambanova_config):
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "sambanova", sambanova_config, admission=immediate_admission()
        )

    assert provider._api_key == "test_sambanova_key"
    assert provider._base_url == SAMBANOVA_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(sambanova_config):
    config = replace(sambanova_config, base_url=f"{SAMBANOVA_DEFAULT_BASE}/")

    with patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"):
        provider = profiled_provider(
            "sambanova", config, admission=immediate_admission()
        )

    assert provider._base_url == SAMBANOVA_DEFAULT_BASE


def test_build_request_body_basic(sambanova_provider):
    """Basic request body conversion attaches system message and keeps max_tokens."""
    body = sambanova_provider._chat._build_request_body(make_request())

    assert body["model"] == "Meta-Llama-3.3-70B-Instruct"
    assert body["messages"][0]["role"] == "system"
    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(sambanova_provider):
    req = make_request(extra_body={"metadata": {"user": "u1"}})

    body = sambanova_provider._chat._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    (
        (ReasoningPolicy.provider_default(), None),
        (ReasoningPolicy.off(), None),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(), "medium"),
    ),
)
def test_build_request_body_uses_only_documented_reasoning_efforts(
    sambanova_provider, reasoning, expected
):
    body = sambanova_provider._chat._build_request_body(
        make_request(),
        reasoning=reasoning,
    )

    assert body.get("reasoning_effort") == expected


@pytest.mark.asyncio
async def test_stream_messages_text(sambanova_provider):
    """Text content deltas are emitted as text blocks."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from SambaNova",
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
        sambanova_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in sambanova_provider.stream_messages(make_request())
        ]

    assert any(
        '"text_delta"' in event and "Hello from SambaNova" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_messages_tool_call(sambanova_provider):
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
        sambanova_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in sambanova_provider.stream_messages(make_request())
        ]

    assert any(
        '"content_block_start"' in event and '"tool_use"' in event for event in events
    )
    assert any(
        '"input_json_delta"' in event and "file_path" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_messages_reasoning_content(sambanova_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via SambaNova",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        sambanova_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in sambanova_provider.stream_messages(make_request())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via SambaNova" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(sambanova_provider):
    sambanova_provider._client = AsyncMock()

    await sambanova_provider.cleanup()

    sambanova_provider._client.close.assert_called_once()
