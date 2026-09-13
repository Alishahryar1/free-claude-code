"""Public commit-boundary behavior for canonical execution failures."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    openai_error_from_failure,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from tests.api.support import create_test_app

_PARTIAL_CONTENT = "PARTIAL_ASSISTANT_CONTENT"


class CanonicalFailureProvider:
    """Provider double that raises one request-correlated canonical failure."""

    def __init__(
        self,
        chunks: list[str],
        *,
        kind: FailureKind,
        status_code: int,
        message: str,
        retryable: bool,
        grouped: bool = False,
    ) -> None:
        self._chunks = chunks
        self._kind = kind
        self._status_code = status_code
        self._message = message
        self._retryable = retryable
        self._grouped = grouped
        self.stream_kwargs: list[dict[str, Any]] = []

    async def stream_messages(
        self,
        _request: object,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.stream_kwargs.append(kwargs)
        for chunk in self._chunks:
            yield chunk
        failure = self._failure(str(kwargs["request_id"]))
        if self._grouped:
            raise self._grouped_failure(failure)
        raise failure

    async def stream_responses(
        self,
        _request: OpenAIResponsesRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.stream_kwargs.append(kwargs)
        failure = self._failure(str(kwargs["request_id"]))
        if not self._chunks:
            if self._grouped:
                raise self._grouped_failure(failure)
            raise failure
        for chunk in _partial_responses_stream(failure):
            yield chunk

    def _failure(self, request_id: str) -> ExecutionFailure:
        return ExecutionFailure(
            kind=self._kind,
            status_code=self._status_code,
            message=f"{self._message}\n\nRequest ID: {request_id}",
            retryable=self._retryable,
        )

    @staticmethod
    def _grouped_failure(failure: ExecutionFailure) -> ExceptionGroup:
        return ExceptionGroup(
            "provider stream and cleanup failed",
            [
                RuntimeError("cleanup failed"),
                ExceptionGroup("provider request failed", [failure]),
            ],
        )


class StalledProvider:
    """Provider double that makes no protocol-visible progress."""

    def __init__(self, *, responses_chunks: tuple[str, ...] = ()) -> None:
        self._responses_chunks = responses_chunks
        self.close_calls = 0

    async def stream_messages(
        self,
        _request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, response_model, reasoning
        try:
            await asyncio.Event().wait()
            yield ""
        finally:
            self.close_calls += 1

    async def stream_responses(
        self,
        _request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, response_model, reasoning
        try:
            for chunk in self._responses_chunks:
                yield chunk
            await asyncio.Event().wait()
            yield ""
        finally:
            self.close_calls += 1


def _messages_payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "nvidia_nim/test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": stream,
    }


def _responses_payload() -> dict[str, object]:
    return {
        "model": "nvidia_nim/test-model",
        "input": "Hello",
        "max_output_tokens": 32,
    }


def _partial_anthropic_stream(*, close_block: bool) -> list[str]:
    chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
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
                "delta": {"type": "text_delta", "text": _PARTIAL_CONTENT},
            },
        ),
    ]
    if close_block:
        chunks.append(
            format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )
        )
    return chunks


def _partial_responses_stream(failure: ExecutionFailure) -> list[str]:
    response_id = "resp_failure_contract"
    message: JsonObject = {
        "id": "msg_failure_contract",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": _PARTIAL_CONTENT,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    created = _responses_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _responses_wire_payload(
                response_id=response_id,
                status="in_progress",
            ),
        },
    )
    item_done = _responses_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": message,
        },
    )
    failed = _responses_event(
        "response.failed",
        {
            "type": "response.failed",
            "sequence_number": 2,
            "response": _responses_wire_payload(
                response_id=response_id,
                status="failed",
                output=[message],
                error=cast(JsonObject, openai_error_from_failure(failure)),
            ),
        },
    )
    return [created, item_done, failed]


def _responses_event(event_type: str, payload: JsonObject) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _responses_wire_payload(
    *,
    response_id: str,
    status: str,
    output: list[JsonObject] | None = None,
    error: JsonObject | None = None,
) -> JsonObject:
    return {
        "id": response_id,
        "object": "response",
        "model": "nvidia_nim/test-model",
        "status": status,
        "output": output or [],
        "error": error,
        "usage": None,
    }


def _client_for(
    provider: CanonicalFailureProvider | StalledProvider,
    *,
    settings: Settings | None = None,
):
    app = create_test_app(settings)
    return (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app),
    )


def _terminal_trace(trace_mock: MagicMock) -> dict[str, Any]:
    return dict(
        next(
            call.kwargs
            for call in trace_mock.call_args_list
            if call.kwargs.get("event")
            == "free_claude_code.api.response.terminal_execution_error"
        )
    )


def _grouped_rate_limit_provider(chunks: list[str]) -> CanonicalFailureProvider:
    return CanonicalFailureProvider(
        chunks,
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
        grouped=True,
    )


def _timeout_provider(chunks: list[str]) -> CanonicalFailureProvider:
    return CanonicalFailureProvider(
        chunks,
        kind=FailureKind.TIMEOUT,
        status_code=504,
        message="Provider execution made no progress for 600 seconds.",
        retryable=False,
    )


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/messages", _messages_payload(stream=True)),
        ("/v1/responses", _responses_payload()),
    ],
)
def test_configured_progress_timeout_reaches_default_handler_executor(
    path: str,
    payload: dict[str, object],
) -> None:
    provider = StalledProvider()
    resolver_patch, client = _client_for(
        provider,
        settings=Settings(provider_progress_timeout=0.02),
    )

    with resolver_patch, client:
        response = client.post(path, json=payload)

    assert response.status_code == 504
    assert response.headers["x-should-retry"] == "false"
    assert "made no progress for 0.02 seconds" in response.json()["error"]["message"]
    assert provider.close_calls == 1


@pytest.mark.parametrize(
    ("path", "payload", "expected_type"),
    [
        ("/v1/messages", _messages_payload(stream=True), "rate_limit_error"),
        ("/v1/responses", _responses_payload(), "rate_limit_error"),
    ],
)
def test_grouped_pre_start_execution_failure_keeps_canonical_wire_error(
    path: str,
    payload: dict[str, object],
    expected_type: str,
) -> None:
    provider = _grouped_rate_limit_provider([])
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == expected_type
    assert error["message"] == f"upstream is busy\n\nRequest ID: {request_id}"
    trace = _terminal_trace(trace_mock)
    assert trace["status_code"] == 429
    assert trace["error_type"] == "rate_limit_error"
    assert trace["exc_type"] == "ExecutionFailure"
    assert trace["failure_kind"] == "rate_limit"


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
def test_grouped_post_start_execution_failure_keeps_canonical_terminal_event(
    path: str,
) -> None:
    provider = _grouped_rate_limit_provider(_partial_anthropic_stream(close_block=True))
    payload = (
        _messages_payload(stream=True)
        if path == "/v1/messages"
        else _responses_payload()
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    if path == "/v1/messages":
        assert events[-1].event == "error"
        error = events[-1].data["error"]
    else:
        assert events[-1].event == "response.failed"
        error = events[-1].data["response"]["error"]
    assert response.status_code == 200
    assert error["type"] == "rate_limit_error"
    assert error["message"] == f"upstream is busy\n\nRequest ID: {request_id}"
    if path == "/v1/messages":
        assert _terminal_trace(trace_mock)["failure_kind"] == "rate_limit"
    else:
        trace_mock.assert_not_called()


def test_grouped_stream_false_execution_failure_discards_partial_content() -> None:
    provider = _grouped_rate_limit_provider(
        _partial_anthropic_stream(close_block=False)
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/messages", json=_messages_payload(stream=False))

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"] == {
        "type": "rate_limit_error",
        "message": f"upstream is busy\n\nRequest ID: {request_id}",
    }
    assert _PARTIAL_CONTENT not in response.text
    trace = _terminal_trace(trace_mock)
    assert trace["status_code"] == 429
    assert trace["error_type"] == "rate_limit_error"
    assert trace["exc_type"] == "ExecutionFailure"
    assert trace["failure_kind"] == "rate_limit"


def test_messages_pre_start_execution_failure_is_correlated_terminal_json() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/messages", json=_messages_payload(stream=True))

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert "x-request-id" not in response.headers
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": f"upstream is busy\n\nRequest ID: {request_id}",
        },
        "request_id": request_id,
    }
    assert provider.stream_kwargs[0]["request_id"] == request_id


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/messages", _messages_payload(stream=True)),
        ("/v1/responses", _responses_payload()),
    ],
)
def test_pre_start_permission_failure_preserves_403_without_client_retry(
    path: str,
    payload: dict[str, object],
) -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.PERMISSION,
        status_code=403,
        message="Provider denied access.",
        retryable=False,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    error = response.json()["error"]
    assert response.status_code == 403
    assert response.headers["x-should-retry"] == "false"
    assert error["type"] == "permission_error"
    assert error["message"] == f"Provider denied access.\n\nRequest ID: {request_id}"
    if path == "/v1/messages":
        assert response.json()["request_id"] == request_id
        assert "x-request-id" not in response.headers
    else:
        assert response.headers["x-request-id"] == request_id
    trace = _terminal_trace(trace_mock)
    assert trace["failure_kind"] == "permission"
    assert trace["status_code"] == 403
    assert trace["error_type"] == "permission_error"
    assert trace["provider_retryable"] is False


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/messages", _messages_payload(stream=True)),
        ("/v1/responses", _responses_payload()),
    ],
)
def test_pre_start_request_too_large_failure_preserves_413(
    path: str,
    payload: dict[str, object],
) -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.INVALID_REQUEST,
        status_code=413,
        message="Provider rejected the request as too large.",
        retryable=False,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    body = response.json()
    assert response.status_code == 413
    assert response.headers["x-should-retry"] == "false"
    assert body["error"]["type"] == "request_too_large"
    assert body["error"]["message"] == (
        f"Provider rejected the request as too large.\n\nRequest ID: {request_id}"
    )
    if path == "/v1/messages":
        assert body["request_id"] == request_id
    else:
        assert response.headers["x-request-id"] == request_id


def test_messages_context_window_failure_triggers_client_compaction() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="Provider input exceeds the model context window.",
        retryable=False,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/messages", json=_messages_payload(stream=True))

    request_id = response.headers["request-id"]
    assert response.status_code == 400
    assert response.headers["x-should-retry"] == "false"
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "prompt is too long\n\n"
                "Provider input exceeds the model context window.\n\n"
                f"Request ID: {request_id}"
            ),
        },
        "request_id": request_id,
    }
    trace = _terminal_trace(trace_mock)
    assert trace["failure_kind"] == "context_window_exceeded"
    assert trace["status_code"] == 400
    assert trace["error_type"] == "invalid_request_error"
    assert trace["provider_retryable"] is False


def test_responses_context_window_failure_triggers_client_compaction() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="Provider input exceeds the model context window.",
        retryable=False,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/responses", json=_responses_payload())

    request_id = response.headers["request-id"]
    assert response.status_code == 400
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"] == {
        "message": (
            "Provider input exceeds the model context window.\n\n"
            f"Request ID: {request_id}"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "context_length_exceeded",
    }


def test_responses_pre_start_execution_failure_is_correlated_terminal_json() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message="provider overloaded",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/responses", json=_responses_payload())

    request_id = response.headers["request-id"]
    assert response.status_code == 529
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"] == request_id
    assert response.json() == {
        "error": {
            "message": f"provider overloaded\n\nRequest ID: {request_id}",
            "type": "overloaded_error",
            "param": None,
            "code": None,
        }
    }
    assert provider.stream_kwargs[0]["request_id"] == request_id


def test_messages_post_start_execution_failure_follows_closed_block() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=True),
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message="provider overloaded",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/messages", json=_messages_payload(stream=True))

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    assert response.status_code == 200
    assert "x-should-retry" not in response.headers
    assert [event.event for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "error",
    ]
    assert events[-1].data["error"] == {
        "type": "overloaded_error",
        "message": f"provider overloaded\n\nRequest ID: {request_id}",
    }
    assert "message_stop" not in response.text
    assert _terminal_trace(trace_mock) == {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": "messages",
        "request_id": request_id,
        "status_code": 529,
        "error_type": "overloaded_error",
        "client_should_retry": False,
        "exc_type": "ExecutionFailure",
        "failure_kind": "overloaded",
        "provider_retryable": True,
    }


def test_responses_post_start_execution_failure_retains_id_after_block_close() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=True),
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/responses", json=_responses_payload())

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    event_names = [event.event for event in events]
    created = events[0].data["response"]
    failed = events[-1].data["response"]
    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
    assert "x-should-retry" not in response.headers
    assert event_names[0] == "response.created"
    assert "response.output_item.done" in event_names
    assert event_names.index("response.output_item.done") < event_names.index(
        "response.failed"
    )
    assert event_names[-1] == "response.failed"
    assert failed["id"] == created["id"]
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "message": f"upstream is busy\n\nRequest ID: {request_id}",
        "type": "rate_limit_error",
        "param": None,
        "code": None,
    }
    trace_mock.assert_not_called()


def test_responses_post_start_context_failure_retains_compaction_code() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=True),
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="Provider input exceeds the model context window.",
        retryable=False,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/responses", json=_responses_payload())

    events = parse_sse_text(response.text)
    assert events[-1].event == "response.failed"
    assert events[-1].data["response"]["error"]["code"] == ("context_length_exceeded")


def test_messages_stream_false_discards_partial_content_on_execution_failure() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=False),
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/messages", json=_messages_payload(stream=False))

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["request_id"] == request_id
    assert response.json()["error"] == {
        "type": "rate_limit_error",
        "message": f"upstream is busy\n\nRequest ID: {request_id}",
    }
    assert _PARTIAL_CONTENT not in response.text


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/messages", _messages_payload(stream=True)),
        ("/v1/responses", _responses_payload()),
    ],
)
def test_pre_start_progress_timeout_is_terminal_504(
    path: str,
    payload: dict[str, object],
) -> None:
    provider = _timeout_provider([])
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    assert response.status_code == 504
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"] == {
        "type": "timeout_error",
        "message": (
            "Provider execution made no progress for 600 seconds.\n\n"
            f"Request ID: {request_id}"
        ),
        **({} if path == "/v1/messages" else {"param": None, "code": None}),
    }
    if path == "/v1/messages":
        assert response.json()["request_id"] == request_id
        assert "x-request-id" not in response.headers
    else:
        assert response.headers["x-request-id"] == request_id
    trace = _terminal_trace(trace_mock)
    assert trace["status_code"] == 504
    assert trace["error_type"] == "timeout_error"
    assert trace["failure_kind"] == "timeout"
    assert trace["provider_retryable"] is False
    assert trace["client_should_retry"] is False


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
def test_post_start_progress_timeout_is_terminal_protocol_event(path: str) -> None:
    provider = _timeout_provider(_partial_anthropic_stream(close_block=True))
    payload = (
        _messages_payload(stream=True)
        if path == "/v1/messages"
        else _responses_payload()
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    assert response.status_code == 200
    assert "x-should-retry" not in response.headers
    if path == "/v1/messages":
        assert events[-1].event == "error"
        error = events[-1].data["error"]
        assert "message_stop" not in response.text
    else:
        assert events[0].event == "response.created"
        assert events[-1].event == "response.failed"
        assert events[-1].data["response"]["id"] == events[0].data["response"]["id"]
        error = events[-1].data["response"]["error"]
    assert error["type"] == "timeout_error"
    assert error["message"] == (
        "Provider execution made no progress for 600 seconds.\n\n"
        f"Request ID: {request_id}"
    )
    if path == "/v1/messages":
        trace = _terminal_trace(trace_mock)
        assert trace["failure_kind"] == "timeout"
        assert trace["provider_retryable"] is False
        assert trace["client_should_retry"] is False
    else:
        trace_mock.assert_not_called()


def test_responses_application_progress_timeout_closes_committed_lifecycle() -> None:
    response_id = "resp_application_timeout"
    provider = StalledProvider(
        responses_chunks=(
            _responses_event(
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": _responses_wire_payload(
                        response_id=response_id,
                        status="in_progress",
                    ),
                },
            ),
        )
    )
    resolver_patch, client = _client_for(
        provider,
        settings=Settings(provider_progress_timeout=0.02),
    )

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/responses", json=_responses_payload())

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    assert response.status_code == 200
    assert [event.event for event in events] == [
        "response.created",
        "response.failed",
    ]
    assert events[-1].data["sequence_number"] == 1
    failed = events[-1].data["response"]
    assert failed["id"] == response_id
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "message": (
            "Provider execution made no progress for 0.02 seconds.\n\n"
            f"Request ID: {request_id}"
        ),
        "type": "timeout_error",
        "param": None,
        "code": None,
    }
    assert provider.close_calls == 1
    assert _terminal_trace(trace_mock) == {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": "responses",
        "request_id": request_id,
        "status_code": 504,
        "error_type": "timeout_error",
        "client_should_retry": False,
        "exc_type": "ExecutionFailure",
        "failure_kind": "timeout",
        "provider_retryable": False,
    }


def test_stream_false_progress_timeout_discards_partial_content() -> None:
    provider = _timeout_provider(_partial_anthropic_stream(close_block=False))
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/messages", json=_messages_payload(stream=False))

    request_id = response.headers["request-id"]
    assert response.status_code == 504
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"] == {
        "type": "timeout_error",
        "message": (
            "Provider execution made no progress for 600 seconds.\n\n"
            f"Request ID: {request_id}"
        ),
    }
    assert _PARTIAL_CONTENT not in response.text
