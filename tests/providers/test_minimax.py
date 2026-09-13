"""Tests for the MiniMax OpenAI-chat provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import MINIMAX_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest, Tool
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


class AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def minimax_provider():
    return profiled_provider(
        "minimax",
        make_provider_config(
            api_key="test-minimax-key",
            base_url=MINIMAX_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


def test_default_base_url():
    assert MINIMAX_DEFAULT_BASE == "https://api.minimax.io/v1"


def test_init_uses_openai_chat_provider(minimax_provider):
    assert isinstance(minimax_provider, OpenAIChatProvider)
    assert minimax_provider._api_key == "test-minimax-key"
    assert minimax_provider._base_url == MINIMAX_DEFAULT_BASE
    assert minimax_provider._provider_name == "MINIMAX"


def test_build_request_body_requests_split_output_and_max_completion_tokens(
    minimax_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "MiniMax-M3",
            "messages": [Message(role="user", content="Hello")],
            "tools": [
                Tool(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = minimax_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "MiniMax-M3"
    assert body["tools"][0]["function"]["name"] == "echo"
    assert body["max_completion_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert "max_tokens" not in body
    assert body["extra_body"] == {"reasoning_split": True}


def test_build_request_body_does_not_invent_unsupported_compute_control(
    minimax_provider,
):
    request = MessagesRequest(
        model="MiniMax-M3",
        messages=[Message(role="user", content="Hello")],
    )

    body = minimax_provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert body["extra_body"] == {"reasoning_split": True}


@pytest.mark.asyncio
async def test_lists_models_from_openai_models_endpoint(minimax_provider):
    minimax_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id="MiniMax-M3"), SimpleNamespace(id="MiniMax-M2.7")]
        )
    )

    assert await minimax_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("MiniMax-M3"), ProviderModelInfo("MiniMax-M2.7")}
    )


@pytest.mark.asyncio
async def test_stream_preserves_reasoning_content(minimax_provider):
    request = MessagesRequest(
        model="MiniMax-M3",
        messages=[Message(role="user", content="hi")],
    )
    stream = AsyncStream(
        [
            _chunk(reasoning_content="plan"),
            _chunk(content="done", finish_reason="stop"),
        ]
    )

    with patch.object(
        minimax_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ) as create:
        events = [event async for event in minimax_provider.stream_messages(request)]

    parsed = parse_sse_text("".join(events))
    assert thinking_content(parsed) == "plan"
    assert text_content(parsed) == "done"
    assert create.await_args is not None
    assert create.await_args.kwargs["extra_body"]["reasoning_split"] is True
    assert stream.closed


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(minimax_provider):
    minimax_provider._client = MagicMock()
    minimax_provider._client.close = AsyncMock()

    await minimax_provider.cleanup()

    minimax_provider._client.close.assert_awaited_once()
