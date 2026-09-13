"""Tests for the local and Cloud Ollama OpenAI-compatible providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import (
    OLLAMA_CLOUD_DEFAULT_BASE,
    OLLAMA_DEFAULT_BASE,
)
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    thinking_content,
)
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_CLOUD_MODEL = "qwen3-coder:480b"


def _provider(base_url: str = OLLAMA_DEFAULT_BASE) -> OpenAIChatProvider:
    return profiled_provider(
        "ollama",
        make_provider_config(api_key="ollama", base_url=base_url),
        admission=immediate_admission(),
    )


def _cloud_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "ollama_cloud",
        make_provider_config(
            api_key="ollama-cloud-key",
            base_url=OLLAMA_CLOUD_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ],
)
def test_init_normalizes_openai_base_url(configured: str, expected: str) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = _provider(configured)

    assert provider._provider_name == "OLLAMA"
    assert provider._base_url == expected
    assert provider._api_key == "ollama"
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_cloud_init_uses_fixed_openai_endpoint_and_api_key() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = _cloud_provider()

    assert provider._provider_name == "OLLAMA_CLOUD"
    assert provider._base_url == OLLAMA_CLOUD_DEFAULT_BASE
    assert provider._api_key == "ollama-cloud-key"
    assert openai_client.call_args.kwargs["base_url"] == OLLAMA_CLOUD_DEFAULT_BASE
    assert openai_client.call_args.kwargs["api_key"] == "ollama-cloud-key"


def test_build_request_body_uses_openai_chat_shape() -> None:
    body = _provider()._chat._build_request_body(make_messages_request(OLLAMA_MODEL))

    assert body["model"] == OLLAMA_MODEL
    assert body["messages"][0]["role"] == "system"
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_cloud_build_request_body_forwards_client_reasoning_effort() -> None:
    request = make_messages_request(
        OLLAMA_CLOUD_MODEL, output_config={"effort": "high"}
    )
    body = _cloud_provider()._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == OLLAMA_CLOUD_MODEL
    assert body["reasoning_effort"] == "high"


def test_cloud_build_request_body_replays_thinking_in_ollama_reasoning_field() -> None:
    request = make_messages_request(
        OLLAMA_CLOUD_MODEL,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the repository."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "contents",
                    }
                ],
            },
        ],
    )

    body = _cloud_provider()._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )
    assert assistant["reasoning"] == "Inspect the repository."
    assert "reasoning_content" not in assistant


@pytest.mark.parametrize("provider", [_provider, _cloud_provider])
def test_replay_is_independent_of_disabled_current_turn_reasoning(provider) -> None:
    request = make_messages_request(
        OLLAMA_CLOUD_MODEL,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hidden plan."},
                    {"type": "text", "text": "Visible answer."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ],
    )

    body = provider()._chat._build_request_body(request, reasoning=REASONING_OFF)
    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )

    assert body["reasoning_effort"] == "none"
    assert assistant["reasoning"] == "Hidden plan."
    assert "reasoning_content" not in assistant
    assert assistant["content"] == "Visible answer."


@pytest.mark.asyncio
async def test_stream_messages_uses_shared_openai_chat_provider() -> None:
    provider = _provider()
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Ollama",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    chunk.usage = MagicMock(prompt_tokens=8, completion_tokens=4)

    async def stream():
        yield chunk

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream(),
    ) as create:
        output = "".join(
            [
                event
                async for event in provider.stream_messages(
                    make_messages_request(OLLAMA_MODEL)
                )
            ]
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == OLLAMA_MODEL
    assert "Hello from Ollama" in output
    assert parse_sse_text(output)[-1].event == "message_stop"


@pytest.mark.asyncio
async def test_cloud_stream_maps_ollama_reasoning_delta_to_anthropic_thinking() -> None:
    client = _cloud_provider()
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Final answer",
                reasoning="Working it out",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    chunk.usage = MagicMock(prompt_tokens=8, completion_tokens=4)

    async def stream():
        yield chunk

    with patch.object(
        client._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream(),
    ):
        output = "".join(
            [
                event
                async for event in client.stream_messages(
                    make_messages_request(OLLAMA_CLOUD_MODEL)
                )
            ]
        )

    events = parse_sse_text(output)
    assert thinking_content(events) == "Working it out"
    assert "Final answer" in output
    assert events[-1].event == "message_stop"
