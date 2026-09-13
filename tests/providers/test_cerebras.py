"""Tests for Cerebras Inference (OpenAI-compatible) provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import CEREBRAS_DEFAULT_BASE
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


def make_request(model="llama3.1-8b", **overrides):
    return make_messages_request(model, **overrides)


def make_reasoning_tool_history_request():
    return make_request(
        "zai-glm-4.7",
        messages=[
            {"role": "user", "content": "Inspect the file."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need to read it first."},
                    {"type": "text", "text": "I will inspect the file."},
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
        ],
    )


@pytest.fixture
def cerebras_config():
    return make_provider_config(
        api_key="test_cerebras_key",
        base_url=CEREBRAS_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def cerebras_provider(cerebras_config):
    return profiled_provider(
        "cerebras", cerebras_config, admission=immediate_admission()
    )


def test_init(cerebras_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "cerebras", cerebras_config, admission=immediate_admission()
        )
        assert provider._api_key == "test_cerebras_key"
        assert provider._base_url == CEREBRAS_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert CEREBRAS_DEFAULT_BASE == "https://api.cerebras.ai/v1"


def test_build_request_body_basic(cerebras_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = make_request()
    body = cerebras_provider._chat._build_request_body(
        req, reasoning=reasoning_for(req)
    )

    assert body["model"] == "llama3.1-8b"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body


def test_build_request_body_replays_reasoning_as_tagged_content(cerebras_provider):
    body = cerebras_provider._chat._build_request_body(
        make_reasoning_tool_history_request()
    )

    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )
    assert assistant["content"] == (
        "<think>\nI need to read it first.\n</think>\n\nI will inspect the file."
    )
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert body["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }
    assert all(
        "reasoning_content" not in message and "reasoning" not in message
        for message in body["messages"]
    )


def test_replay_is_independent_of_current_turn_reasoning_control():
    provider = profiled_provider(
        "cerebras",
        make_provider_config(
            api_key="test_cerebras_key",
            base_url=CEREBRAS_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )
    body = provider._chat._build_request_body(
        make_reasoning_tool_history_request(), reasoning=REASONING_OFF
    )

    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )
    assert assistant["content"] == (
        "<think>\nI need to read it first.\n</think>\n\nI will inspect the file."
    )
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert body["reasoning_effort"] == "none"


def test_build_request_body_remaps_max_tokens_preserves_message_name(cerebras_provider):
    """Cerebras does not strip message ``name``; ``max_tokens`` maps to completion field."""
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
        }
        req = make_request()
        body = cerebras_provider._chat._build_request_body(
            req, reasoning=reasoning_for(req)
        )

    assert body["messages"][0].get("name") == "alice"
    assert body.get("max_tokens") is None
    assert body["max_completion_tokens"] == 42


def test_build_request_body_prefers_existing_max_completion_tokens(cerebras_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 77,
            "max_tokens": 999,
        }
        body = cerebras_provider._chat._build_request_body(make_request())

    assert body["max_completion_tokens"] == 77
    assert "max_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(cerebras_provider):
    req = make_request(extra_body={"clear_thinking": False})

    body = cerebras_provider._chat._build_request_body(
        req, reasoning=reasoning_for(req)
    )

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("clear_thinking") is False


@pytest.mark.asyncio
async def test_stream_messages_text(cerebras_provider):
    """Text content deltas are emitted as text blocks."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
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
        cerebras_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in cerebras_provider.stream_messages(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_messages_reasoning(cerebras_provider):
    """Cerebras reasoning deltas are emitted as thinking blocks."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        cerebras_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in cerebras_provider.stream_messages(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(cerebras_provider):
    cerebras_provider._client = AsyncMock()

    await cerebras_provider.cleanup()

    cerebras_provider._client.close.assert_called_once()
