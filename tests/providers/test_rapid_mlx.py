"""Tests for the Rapid-MLX OpenAI-compatible local provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import RAPID_MLX_DEFAULT_BASE
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    SDKStreamDouble,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
    sdk_chat_chunk,
)

RAPID_MLX_MODEL = "rapid_mlx/local-model"


@pytest.fixture
def provider() -> OpenAIChatProvider:
    return profiled_provider(
        "rapid_mlx",
        make_provider_config(
            api_key="test-rapid_mlx-key", base_url=RAPID_MLX_DEFAULT_BASE
        ),
        admission=immediate_admission(),
    )


def test_default_base_url_constant() -> None:
    assert RAPID_MLX_DEFAULT_BASE == "http://localhost:8000/v1"


def test_init_uses_openai_chat_client() -> None:
    config = make_provider_config(
        api_key="test-rapid_mlx-key",
        base_url="http://localhost:8000/v1/",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider(
            "rapid_mlx", config, admission=immediate_admission()
        )

    assert provider._provider_name == "Rapid-MLX"
    assert provider._base_url == "http://localhost:8000/v1"
    assert provider._api_key == "test-rapid_mlx-key"
    timeout = openai_client.call_args.kwargs["timeout"]
    assert (timeout.read, timeout.write, timeout.connect) == (600.0, 15.0, 5.0)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1"),
        ("http://localhost:8000/", "http://localhost:8000/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1"),
    ],
)
def test_init_normalizes_openai_base_url(configured: str, expected: str) -> None:
    config = make_provider_config(api_key="test-rapid_mlx-key", base_url=configured)
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider(
            "rapid_mlx", config, admission=immediate_admission()
        )

    assert provider._base_url == expected
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_build_request_body_uses_openai_chat_shape(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(RAPID_MLX_MODEL, max_tokens=None)

    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == RAPID_MLX_MODEL
    assert "max_tokens" not in body
    assert body["messages"][0]["role"] == "system"
    assert "thinking" not in body


def test_replay_preserves_reasoning_history(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(
        RAPID_MLX_MODEL,
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

    assert body["messages"][1]["content"] == "visible"
    assert body["messages"][1]["reasoning_content"] == "private"
    assert "extra_body" not in body


@pytest.mark.asyncio
async def test_stream_messages_uses_shared_openai_chat_provider(
    provider: OpenAIChatProvider,
) -> None:
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Rapid-MLX",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    chunk.usage = MagicMock(prompt_tokens=8, completion_tokens=4)

    async def stream():
        yield sdk_chat_chunk(chunk)

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=SDKStreamDouble(stream()),
    ) as create:
        output = "".join(
            [
                frame
                async for event in provider.stream_messages(
                    make_messages_request(RAPID_MLX_MODEL)
                )
                for frame in event.frames
            ]
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == RAPID_MLX_MODEL
    assert "Hello from Rapid-MLX" in output
    assert parse_sse_text(output)[-1].event == "message_stop"
