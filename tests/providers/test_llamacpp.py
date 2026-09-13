"""Tests for the llama.cpp OpenAI-compatible provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

LLAMACPP_MODEL = "llamacpp-community/qwen2.5-7b-instruct"


@pytest.fixture
def provider() -> OpenAIChatProvider:
    return profiled_provider(
        "llamacpp",
        make_provider_config(api_key="llamacpp", base_url="http://localhost:8080/v1"),
        admission=immediate_admission(),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:8080", "http://localhost:8080/v1"),
        ("http://localhost:8080/", "http://localhost:8080/v1"),
        ("http://localhost:8080/v1", "http://localhost:8080/v1"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
    ],
)
def test_init_normalizes_openai_base_url(configured: str, expected: str) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider(
            "llamacpp",
            make_provider_config(api_key="llamacpp", base_url=configured),
            admission=immediate_admission(),
        )

    assert provider._base_url == expected
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_init_uses_openai_chat_client() -> None:
    config = make_provider_config(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1/",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider(
            "llamacpp", config, admission=immediate_admission()
        )

    assert provider._provider_name == "LLAMACPP"
    assert provider._base_url == "http://localhost:8080/v1"
    assert provider._api_key == "llamacpp"
    timeout = openai_client.call_args.kwargs["timeout"]
    assert (timeout.read, timeout.write, timeout.connect) == (600.0, 15.0, 5.0)


def test_build_request_body_uses_openai_chat_shape(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(LLAMACPP_MODEL, max_tokens=None)

    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == LLAMACPP_MODEL
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"][0]["role"] == "system"
    assert "thinking" not in body


def test_replay_is_independent_of_current_turn_reasoning_control(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(
        LLAMACPP_MODEL,
        system=None,
        messages=[
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "s"},
                    {"type": "text", "text": "visible"},
                ],
            },
        ],
    )

    body = provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert body["messages"][1]["content"] == ("<think>\nprivate\n</think>\n\nvisible")
    assert body["extra_body"]["thinking_budget_tokens"] == 0


@pytest.mark.asyncio
async def test_stream_messages_uses_shared_openai_chat_provider(
    provider: OpenAIChatProvider,
) -> None:
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from llama.cpp",
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
                    make_messages_request(LLAMACPP_MODEL)
                )
            ]
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == LLAMACPP_MODEL
    assert "Hello from llama.cpp" in output
    assert parse_sse_text(output)[-1].event == "message_stop"
