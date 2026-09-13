"""Tests for the OpenRouter OpenAI-chat provider."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.history_replay import decode_replay
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
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


def make_request(**overrides):
    return make_messages_request("moonshotai/kimi-k2.6:free", **overrides)


@pytest.fixture
def open_router_provider():
    return OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[dict] | None = None,
    finish_reason: str | None = None,
):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=None,
    )
    if reasoning_details is not None:
        delta.reasoning_details = reasoning_details
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_init_uses_openai_chat_provider(open_router_provider):
    assert isinstance(open_router_provider, OpenAIChatProvider)
    assert open_router_provider._api_key == "test_openrouter_key"
    assert open_router_provider._base_url == "https://openrouter.ai/api/v1"


def test_build_request_body_uses_openai_chat_shape(open_router_provider):
    body = open_router_provider._chat._build_request_body(make_request())

    assert body["model"] == "moonshotai/kimi-k2.6:free"
    assert body["temperature"] == 0.5
    assert body["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello"},
    ]
    assert body["max_tokens"] == 100
    assert "extra_body" not in body


def test_build_request_body_default_max_tokens(open_router_provider):
    body = open_router_provider._chat._build_request_body(make_request(max_tokens=None))

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_openrouter_extra_body_rejects_overriding_reserved_fields(
    open_router_provider,
):
    with pytest.raises(InvalidRequestError, match="model"):
        open_router_provider._chat._build_request_body(
            make_request(extra_body={"model": "hijack"})
        )


def test_openrouter_extra_body_allows_provider_keys(open_router_provider):
    body = open_router_provider._chat._build_request_body(
        make_request(extra_body={"transforms": ["no-web"], "plugins": []}),
        reasoning=REASONING_OFF,
    )

    assert body["extra_body"] == {
        "transforms": ["no-web"],
        "plugins": [],
        "reasoning": {"enabled": False},
    }


def test_build_request_body_disables_reasoning_when_client_disables_it(
    open_router_provider,
):
    request = make_request(thinking={"type": "disabled"})
    body = open_router_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["reasoning"] == {"enabled": False}


def test_build_request_body_maps_thinking_budget_to_reasoning_max_tokens(
    open_router_provider,
):
    request = make_request(thinking={"type": "enabled", "budget_tokens": 4096})
    body = open_router_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["reasoning"] == {"max_tokens": 4096}


def test_build_request_body_replays_openrouter_reasoning_details(
    open_router_provider,
):
    detail = {"type": "reasoning.encrypted", "data": "opaque"}
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": '{"type":"reasoning.encrypted","data":"opaque"}',
                        },
                        {"type": "text", "text": "Need a tool."},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )

    body = open_router_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assistant = next(msg for msg in body["messages"] if msg["role"] == "assistant")
    assert assistant["reasoning_details"] == [detail]


def test_reasoning_details_skip_neutral_tool_turn_boundary(open_router_provider):
    first_detail = {"type": "reasoning.encrypted", "data": "first"}
    second_detail = {"type": "reasoning.encrypted", "data": "second"}
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": json.dumps(first_detail),
                        },
                        {
                            "type": "tool_use",
                            "id": "call_read",
                            "name": "Read",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_read",
                            "content": "contents",
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": json.dumps(second_detail),
                        },
                        {"type": "text", "text": "done"},
                    ],
                },
            ],
        }
    )

    body = open_router_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assistants = [
        message for message in body["messages"] if message["role"] == "assistant"
    ]
    assert assistants[0]["reasoning_details"] == [first_detail]
    assert assistants[1] == {"role": "assistant", "content": " "}
    assert assistants[2]["reasoning_details"] == [second_detail]


def test_reasoning_details_preserve_redacted_only_assistant_after_tool(
    open_router_provider,
):
    first_detail = {"type": "reasoning.encrypted", "data": "first"}
    second_detail = {"type": "reasoning.encrypted", "data": "second"}
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": json.dumps(first_detail),
                        },
                        {
                            "type": "tool_use",
                            "id": "call_read",
                            "name": "Read",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_read",
                            "content": "contents",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": json.dumps(second_detail),
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "done"},
            ],
        }
    )

    body = open_router_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assistants = [
        message for message in body["messages"] if message["role"] == "assistant"
    ]
    assert assistants[0]["reasoning_details"] == [first_detail]
    assert assistants[1] == {
        "role": "assistant",
        "content": " ",
        "reasoning_details": [second_detail],
    }
    assert "reasoning_details" not in assistants[2]


@pytest.mark.asyncio
async def test_stream_maps_reasoning_content_and_details(open_router_provider):
    redacted = {"type": "reasoning.encrypted", "data": "opaque"}
    stream = AsyncStream(
        [
            _chunk(reasoning_details=[{"type": "reasoning.text", "text": "plan "}]),
            _chunk(reasoning_content="plan "),
            _chunk(reasoning_details=[redacted]),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    with patch.object(
        open_router_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [
            event
            async for event in open_router_provider.stream_messages(make_request())
        ]

    event_text = "".join(events)
    parsed = parse_sse_text(event_text)
    assert thinking_content(parsed) == "plan "
    carriers = [
        event.data["delta"]["signature"]
        for event in parsed
        if event.data.get("delta", {}).get("type") == "signature_delta"
    ]
    assert len(carriers) == 1
    assert decode_replay(carriers[0]).native["reasoning_details"] == [
        {"type": "reasoning.text", "text": "plan "},
        redacted,
    ]
    assert text_content(parsed) == "done"
    assert stream.closed


@pytest.mark.asyncio
async def test_model_infos_filter_tool_models_and_thinking_metadata(
    open_router_provider,
):
    open_router_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="tool-model",
                    supported_parameters=["tools", "reasoning"],
                    architecture=SimpleNamespace(input_modalities=["text", "image"]),
                    context_length=262144,
                    top_provider=SimpleNamespace(
                        context_length=999999,
                        max_completion_tokens=32768,
                    ),
                ),
                SimpleNamespace(
                    id="tool-model-with-malformed-capabilities",
                    supported_parameters=["tools", 7],
                    architecture=SimpleNamespace(input_modalities=["text"]),
                ),
                SimpleNamespace(id="plain-model", supported_parameters=[]),
            ]
        )
    )

    infos = await open_router_provider.list_model_infos()

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "tool-model",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=262144,
                max_output_tokens=32768,
            ),
            ProviderModelInfo(
                "tool-model-with-malformed-capabilities",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(open_router_provider):
    open_router_provider._client = MagicMock()
    open_router_provider._client.close = AsyncMock()

    await open_router_provider.cleanup()

    open_router_provider._client.close.assert_awaited_once()
