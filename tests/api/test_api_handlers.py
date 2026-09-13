import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from free_claude_code.api.handlers import (
    MessagesHandler,
    ResponsesHandler,
    TokenCountHandler,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
    TokenCountRequest,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy

_LEGACY_CLASSIFIER_SYSTEM = (
    "You are a security monitor. Respond with <block>yes</block> or <block>no</block>."
)
_CURRENT_CLASSIFIER_SYSTEM = (
    "Classify the command's safety. Output <severity>N</severity> where N is the "
    "numeric severity."
)
_CLASSIFIER_USER = (
    "<transcript>\nUser: review the repo\nBash: inspect the requested file\n"
    "</transcript>"
)


class FakeProvider:
    def __init__(self, events: list[str] | None = None) -> None:
        self.requests: list[MessagesRequest] = []
        self.responses_requests: list[OpenAIResponsesRequest] = []
        self.stream_kwargs: list[dict[str, Any]] = []
        self.events = events or [
            'event: message_start\ndata: {"type":"message_start"}\n\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset({ProviderModelInfo("test-model")})

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self.requests.append(request)
        self.stream_kwargs.append(
            {
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        for event in self.events:
            yield event

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        self.responses_requests.append(request)
        self.stream_kwargs.append(
            {
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        yield 'event: response.completed\ndata: {"type":"response.completed"}\n\n'


async def _streaming_body_text(response: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            parts.append(chunk.decode("utf-8"))
        else:
            parts.append(str(chunk))
    return "".join(parts)


def _json_response_content(response: JSONResponse) -> dict[str, Any]:
    content = json.loads(bytes(response.body).decode("utf-8"))
    assert isinstance(content, dict)
    return content


def _trace_events(trace_mock: MagicMock, event: str) -> list[dict[str, Any]]:
    return [
        dict(call.kwargs)
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == event
    ]


@pytest.mark.asyncio
async def test_messages_handler_passes_routed_request_and_stream_metadata() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=True,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)
    assert isinstance(response, StreamingResponse)

    body = await _streaming_body_text(response)
    assert "message_start" in body
    assert provider.requests[0].model == "test-model"
    assert provider.stream_kwargs[0]["input_tokens"] > 0
    assert provider.stream_kwargs[0]["request_id"].startswith("req_")
    assert provider.stream_kwargs[0]["response_model"] == "nvidia_nim/test-model"
    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.provider_default()
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_messages_handler_startup_invalid_request_stays_http_error(
    stream: bool,
) -> None:
    class RejectStartupProvider(FakeProvider):
        def stream_messages(
            self,
            request: MessagesRequest,
            input_tokens: int = 0,
            *,
            reasoning: ReasoningPolicy,
            model_info: ProviderModelInfo | None = None,
            **kwargs: object,
        ) -> AsyncIterator[str]:
            raise InvalidRequestError("bad tool shape")

    provider = RejectStartupProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
        stream=stream,
    )

    if stream:
        response = await handler.create(request)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        assert json.loads(bytes(response.body))["error"]["message"] == "bad tool shape"
    else:
        with pytest.raises(InvalidRequestError, match="bad tool shape"):
            await handler.create(request)


@pytest.mark.asyncio
async def test_messages_handler_aggregates_provider_stream_when_stream_false() -> None:
    provider = FakeProvider(
        [
            format_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "test-model",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 7, "output_tokens": 1},
                    },
                },
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "OK"},
                },
            ),
            format_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 10,
                    },
                },
            ),
            format_sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=False,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)

    assert isinstance(response, JSONResponse)
    assert response.headers["content-type"].startswith("application/json")
    body = _json_response_content(response)
    assert body["id"] == "msg_test"
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "test-model"
    assert body["content"] == [{"type": "text", "text": "OK"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {
        "input_tokens": 20,
        "output_tokens": 2,
        "cache_read_input_tokens": 10,
    }


@pytest.mark.asyncio
async def test_messages_handler_returns_error_json_for_stream_false_sse_error() -> None:
    provider = FakeProvider(
        [
            format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "upstream failed"},
                },
            )
        ]
    )
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=False,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    body = _json_response_content(response)
    assert body["type"] == "error"
    assert body["error"] == {"type": "api_error", "message": "upstream failed"}
    assert body["request_id"].startswith("req_")


@pytest.mark.asyncio
async def test_messages_handler_discards_partial_stream_false_output_on_error() -> None:
    provider = FakeProvider(
        [
            format_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_partial",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "test-model",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "incomplete"},
                },
            ),
            format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "provider overloaded",
                    },
                },
            ),
        ]
    )
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=False,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 529
    assert response.headers["x-should-retry"] == "false"
    body = _json_response_content(response)
    assert body["error"] == {
        "type": "overloaded_error",
        "message": "provider overloaded",
    }
    assert "content" not in body


@pytest.mark.asyncio
async def test_messages_handler_stream_false_provider_exception_keeps_status() -> None:
    class FailingProvider(FakeProvider):
        async def stream_messages(
            self,
            request: Any,
            input_tokens: int = 0,
            *,
            request_id: str | None = None,
            response_model: str | None = None,
            reasoning: ReasoningPolicy,
            request_headers: Mapping[str, str] | None = None,
            model_info: ProviderModelInfo | None = None,
        ) -> AsyncIterator[str]:
            self.requests.append(request)
            self.stream_kwargs.append(
                {
                    "input_tokens": input_tokens,
                    "request_id": request_id,
                    "response_model": response_model,
                    "reasoning": reasoning,
                }
            )
            raise ExecutionFailure(
                kind=FailureKind.RATE_LIMIT,
                status_code=429,
                message="upstream is busy",
                retryable=True,
            )
            yield "unreachable"

    provider = FailingProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=False,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    body = _json_response_content(response)
    assert body["error"] == {
        "type": "rate_limit_error",
        "message": "upstream is busy",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("system", "classifier_stop_sequence"),
    [
        (_CURRENT_CLASSIFIER_SYSTEM, "</severity>"),
        (_LEGACY_CLASSIFIER_SYSTEM, "</block>"),
    ],
)
async def test_messages_handler_normalizes_safety_classifier_policy(
    system: str,
    classifier_stop_sequence: str,
) -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=True,
        system=system,
        stop_sequences=[classifier_stop_sequence],
        messages=[Message(role="user", content=_CLASSIFIER_USER)],
    )

    with patch("free_claude_code.api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.prefer_off()
    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.prefer_off()
    assert provider.requests[0].model == "test-model"
    assert provider.requests[0].system == system
    assert provider.requests[0].stop_sequences is None
    assert provider.requests[0].stop_sequences is None
    assert request.stop_sequences == [classifier_stop_sequence]
    assert _trace_events(
        trace_mock, "free_claude_code.api.route.safety_classifier_policy"
    ) == [
        {
            "stage": "routing",
            "event": "free_claude_code.api.route.safety_classifier_policy",
            "source": "api",
            "model": "nvidia_nim/test-model",
            "classifier_stop_sequence": classifier_stop_sequence,
            "reasoning_changed": True,
            "stop_sequence_removed": True,
        }
    ]


@pytest.mark.asyncio
async def test_messages_handler_preserves_thinking_for_non_classifier() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=True,
        system="Explain XML formats.",
        stop_sequences=["</severity>"],
        messages=[
            Message(
                role="user",
                content=(
                    "Explain <transcript>...</transcript> and a <block> tag "
                    "without making a verdict."
                ),
            )
        ],
    )

    with patch("free_claude_code.api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.provider_default()
    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.provider_default()
    assert provider.requests[0].stop_sequences == ["</severity>"]
    assert (
        _trace_events(
            trace_mock,
            "free_claude_code.api.route.safety_classifier_policy",
        )
        == []
    )


@pytest.mark.asyncio
async def test_messages_handler_tolerates_required_thinking_for_classifier() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="claude-3-freecc-no-thinking/nvidia_nim/test-model",
        max_tokens=100,
        stream=True,
        system=_LEGACY_CLASSIFIER_SYSTEM,
        stop_sequences=["</block>"],
        messages=[Message(role="user", content=_CLASSIFIER_USER)],
    )

    with patch("free_claude_code.api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.prefer_off()
    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.prefer_off()
    assert provider.requests[0].stop_sequences is None
    assert request.stop_sequences == ["</block>"]
    assert _trace_events(
        trace_mock, "free_claude_code.api.route.safety_classifier_policy"
    ) == [
        {
            "stage": "routing",
            "event": "free_claude_code.api.route.safety_classifier_policy",
            "source": "api",
            "model": "claude-3-freecc-no-thinking/nvidia_nim/test-model",
            "classifier_stop_sequence": "</block>",
            "reasoning_changed": True,
            "stop_sequence_removed": True,
        }
    ]


@pytest.mark.asyncio
async def test_messages_handler_prefers_no_thinking_without_classifier_stop_hint() -> (
    None
):
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=True,
        system=_CURRENT_CLASSIFIER_SYSTEM,
        messages=[Message(role="user", content=_CLASSIFIER_USER)],
    )

    with patch("free_claude_code.api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.prefer_off()
    assert provider.requests[0].stop_sequences is None
    assert _trace_events(
        trace_mock, "free_claude_code.api.route.safety_classifier_policy"
    ) == [
        {
            "stage": "routing",
            "event": "free_claude_code.api.route.safety_classifier_policy",
            "source": "api",
            "model": "nvidia_nim/test-model",
            "classifier_stop_sequence": "</severity>",
            "reasoning_changed": True,
            "stop_sequence_removed": False,
        }
    ]


@pytest.mark.asyncio
async def test_messages_handler_preserves_unowned_classifier_stop_sequences() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
    original_stops = ["custom", "</severity>", "custom", "</severity>", "tail"]
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=True,
        system=_CURRENT_CLASSIFIER_SYSTEM,
        stop_sequences=original_stops,
        messages=[Message(role="user", content=_CLASSIFIER_USER)],
    )

    response = await handler.create(request)
    assert isinstance(response, StreamingResponse)
    await _streaming_body_text(response)

    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.prefer_off()
    assert provider.requests[0].stop_sequences == [
        "custom",
        "custom",
        "tail",
    ]
    assert provider.requests[0].stop_sequences == ["custom", "custom", "tail"]
    assert request.stop_sequences == original_stops


@pytest.mark.asyncio
async def test_messages_handler_optimization_intercepts_before_provider_execution() -> (
    None
):
    provider_resolver = MagicMock()
    handler = MessagesHandler(Settings(), provider_resolver=provider_resolver)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        messages=[Message(role="user", content="quota check")],
    )
    optimized = object()

    with patch(
        "free_claude_code.api.handlers.messages.try_optimizations",
        return_value=optimized,
    ):
        assert await handler.create(request) is optimized

    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_responses_handler_bypasses_message_only_optimizations() -> None:
    provider = FakeProvider()
    handler = ResponsesHandler(Settings(), provider_resolver=lambda _: provider)

    with patch(
        "free_claude_code.api.handlers.messages.try_optimizations",
        side_effect=AssertionError("Responses must not use message optimizations"),
    ):
        response = await handler.create(
            OpenAIResponsesRequest(
                model="nvidia_nim/test-model",
                input="quota check",
            )
        )

    assert isinstance(response, StreamingResponse)
    body = await _streaming_body_text(response)
    assert "response.completed" in body
    assert provider.responses_requests[0].input == "quota check"


@pytest.mark.asyncio
async def test_responses_handler_does_not_apply_safety_classifier_policy() -> None:
    provider = FakeProvider()
    handler = ResponsesHandler(Settings(), provider_resolver=lambda _: provider)

    with patch("free_claude_code.api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(
            OpenAIResponsesRequest(
                model="nvidia_nim/test-model",
                input=_CLASSIFIER_USER,
                instructions=_CURRENT_CLASSIFIER_SYSTEM,
            )
        )

        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.provider_default()
    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.provider_default()
    assert (
        _trace_events(
            trace_mock,
            "free_claude_code.api.route.safety_classifier_policy",
        )
        == []
    )


def test_token_count_handler_routes_and_counts_tokens() -> None:
    handler = TokenCountHandler(
        Settings(),
        token_counter=lambda messages, system, tools: len(messages) + 41,
    )

    with patch("free_claude_code.api.handlers.token_count.trace_event") as trace:
        response = handler.count(
            TokenCountRequest(
                model="nvidia_nim/test-model",
                messages=[Message(role="user", content="hi")],
            ),
            request_id="req_ingress",
        )

    assert response.input_tokens == 42
    assert all(
        call.kwargs["request_id"] == "req_ingress" for call in trace.call_args_list
    )
