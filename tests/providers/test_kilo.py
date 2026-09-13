"""Tests for the Kilo.ai OpenAI-chat provider."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import KILO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.history_replay import decode_replay
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.kilo import KiloProvider
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    reasoning_for,
)


@pytest.fixture
def kilo_config():
    return make_provider_config(
        api_key="test_kilo_key",
        base_url=KILO_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def kilo_provider(kilo_config):
    return KiloProvider(
        kilo_config,
        admission=immediate_admission(provider_name="kilo"),
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


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    reasoning_details: list[dict] | None = None,
    finish_reason: str | None = None,
):
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        tool_calls=None,
    )
    if reasoning_details is not None:
        delta.reasoning_details = reasoning_details
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_default_base_url():
    assert KILO_DEFAULT_BASE == "https://api.kilo.ai/api/gateway"


def test_init_uses_specialized_openai_chat_provider(kilo_provider):
    assert isinstance(kilo_provider, KiloProvider)
    assert isinstance(kilo_provider, OpenAIChatProvider)
    assert kilo_provider._api_key == "test_kilo_key"
    assert kilo_provider._base_url == KILO_DEFAULT_BASE
    assert kilo_provider._provider_name == "KILO"


def test_build_request_body_openai_shape(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.5",
            "messages": [Message(role="user", content="Hello")],
            "max_tokens": 100,
        }
    )

    body = kilo_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "anthropic/claude-sonnet-4.5"
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100


def test_build_request_body_forwards_caller_extra_body(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"custom_field": "value"},
        }
    )

    body = kilo_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body.get("extra_body", {}).get("custom_field") == "value"


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "messages",
        "tools",
        "stream",
        "max_tokens",
        "reasoning",
    ],
)
def test_extra_body_cannot_override_canonical_request_fields(kilo_provider, field):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {field: "hijack"},
        }
    )

    with pytest.raises(InvalidRequestError, match=field):
        kilo_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_build_request_body_sends_reasoning_object(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = kilo_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body.get("extra_body", {}).get("reasoning") is not None


def test_build_request_body_sends_reasoning_disabled(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    body = kilo_provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.off(),
    )

    assert body["extra_body"]["reasoning"] == {"enabled": False}


def test_build_request_body_replays_opaque_reasoning_details_on_tool_turn(
    kilo_provider,
):
    detail = {
        "type": "reasoning.opaque",
        "data": {"signature": "opaque-value"},
    }
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": json.dumps(detail),
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
            ],
        }
    )

    body = kilo_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assistant = next(msg for msg in body["messages"] if msg["role"] == "assistant")
    assert assistant["reasoning_details"] == [detail]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning_chunks",
    [
        [
            _chunk(
                reasoning="plan ",
                reasoning_details=[{"type": "reasoning.text", "text": "plan "}],
            )
        ],
        [
            _chunk(reasoning="plan "),
            _chunk(reasoning_details=[{"type": "reasoning.text", "text": "plan "}]),
        ],
    ],
    ids=["same-delta", "cross-delta"],
)
async def test_stream_uses_reasoning_field_without_duplicating_plain_details(
    kilo_provider,
    reasoning_chunks,
):
    encrypted = {"type": "reasoning.encrypted", "data": "opaque"}
    stream = AsyncStream(
        [
            *reasoning_chunks,
            _chunk(reasoning_details=[encrypted]),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    request = MessagesRequest.model_validate(
        {
            "model": "kilo-auto/free",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    with patch.object(
        kilo_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        event_text = "".join(
            [event async for event in kilo_provider.stream_messages(request)]
        )

    events = parse_sse_text(event_text)
    assert thinking_content(events) == "plan "
    assert text_content(events) == "done"
    records = [
        decode_replay(event.data["delta"]["signature"]).native
        for event in events
        if event.data.get("delta", {}).get("type") == "signature_delta"
    ]
    assert len(records) == 1
    assert records[0]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "plan "},
        encrypted,
    ]
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_restarts_reasoning_reconciliation_after_early_retry(
    kilo_provider,
):
    abandoned = AsyncStream([_chunk(reasoning="discarded ")])
    recovered = AsyncStream(
        [
            _chunk(reasoning_details=[{"type": "reasoning.text", "text": "plan "}]),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    request = MessagesRequest.model_validate(
        {
            "model": "kilo-auto/free",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    with patch.object(
        kilo_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=[abandoned, recovered],
    ) as create:
        event_text = "".join(
            [event async for event in kilo_provider.stream_messages(request)]
        )

    events = parse_sse_text(event_text)
    assert thinking_content(events) == "plan "
    assert text_content(events) == "done"
    assert create.await_count == 2
    assert abandoned.closed
    assert recovered.closed


@pytest.mark.asyncio
async def test_stream_omits_all_reasoning_representations_when_disabled(
    kilo_provider,
):
    stream = AsyncStream(
        [
            _chunk(
                reasoning="plan ",
                reasoning_details=[
                    {"type": "reasoning.text", "text": "plan "},
                    {"type": "reasoning.encrypted", "data": "opaque"},
                ],
            ),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    request = MessagesRequest.model_validate(
        {
            "model": "kilo-auto/free",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    with patch.object(
        kilo_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        event_text = "".join(
            [
                event
                async for event in kilo_provider.stream_messages(
                    request,
                    reasoning=ReasoningPolicy.off(),
                )
            ]
        )

    events = parse_sse_text(event_text)
    assert thinking_content(events) == ""
    assert text_content(events) == "done"
    assert all(
        event.data.get("content_block", {}).get("type") != "redacted_thinking"
        for event in events
        if event.event == "content_block_start"
    )
    assert stream.closed


@pytest.mark.asyncio
async def test_model_list_filters_to_chat_tool_models_with_capabilities(kilo_provider):
    kilo_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                {
                    "id": "anthropic/tool-reasoning",
                    "supported_parameters": ["tools", "reasoning"],
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "opencode": {"ai_sdk_provider": "anthropic"},
                    "context_length": 200000,
                    "top_provider": {"max_completion_tokens": 16000},
                },
                {
                    "id": "plain-tool",
                    "supported_parameters": ["tool_choice"],
                    "architecture": {"output_modalities": ["text"]},
                    "opencode": {},
                },
                {
                    "id": "missing-optional-metadata",
                    "supported_parameters": ["tools"],
                },
                {
                    "id": "malformed-optional-metadata",
                    "supported_parameters": ["tools", 7],
                    "architecture": {"input_modalities": ["text"]},
                },
                {
                    "id": "chat-only",
                    "supported_parameters": ["reasoning"],
                },
                {
                    "id": "image-output",
                    "supported_parameters": ["tools"],
                    "architecture": {"output_modalities": ["text", "image"]},
                },
                {
                    "id": "responses-only",
                    "supported_parameters": ["tools", "reasoning"],
                    "opencode": {"ai_sdk_provider": "openai"},
                },
                {
                    "id": "missing-tool-metadata",
                    "supported_parameters": None,
                },
            ]
        )
    )

    assert await kilo_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo(
                "anthropic/tool-reasoning",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=200000,
                max_output_tokens=16000,
            ),
            ProviderModelInfo("plain-tool", supports_thinking=False),
            ProviderModelInfo(
                "missing-optional-metadata",
                supports_thinking=False,
            ),
            ProviderModelInfo(
                "malformed-optional-metadata",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )

    kilo_provider._client.models.list.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        SimpleNamespace(data="not-an-array"),
        SimpleNamespace(data=[SimpleNamespace(id="", supported_parameters=["tools"])]),
    ],
)
async def test_model_list_rejects_malformed_payload(kilo_provider, payload):
    kilo_provider._client.models.list = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match="KILO model-list"):
        await kilo_provider.list_model_infos()


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(kilo_provider):
    kilo_provider._client = MagicMock()
    kilo_provider._client.close = AsyncMock()

    await kilo_provider.cleanup()

    kilo_provider._client.close.assert_awaited_once()
