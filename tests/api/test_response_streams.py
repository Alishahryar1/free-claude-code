"""Tests for public SSE response start gating."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Message, Scope

from free_claude_code.api.request_ids import RequestCorrelationMiddleware
from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    anthropic_sse_streaming_response,
    bind_response_lifetime,
    terminal_execution_error_response,
)
from free_claude_code.core.anthropic import (
    anthropic_error_payload,
    anthropic_failure_payload,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind


async def _body_chunks(chunks: list[str]) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk


async def _body_raises(exc: BaseException) -> AsyncGenerator[str]:
    raise exc
    yield "unreachable"


async def _body_then_raises(
    chunks: list[str], exc: BaseException
) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk
    raise exc


def _json_error(exc: BaseException) -> JSONResponse:
    if isinstance(exc, ExecutionFailure):
        return terminal_execution_error_response(
            status_code=exc.status_code,
            content=anthropic_failure_payload(exc),
        )
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {"type": "api_error", "message": "failed"},
        },
    )


async def _drain(response: StreamingResponse) -> str:
    parts = [
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ]
    return "".join(parts)


def _http_scope() -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"",
            "headers": [],
            "client": None,
            "server": None,
        },
    )


async def _serve(
    response: StreamingResponse,
    *,
    send: Any | None = None,
) -> list[Message]:
    messages: list[Message] = []

    async def receive() -> Message:
        raise AssertionError("ASGI spec 2.4 responses must not read receive")

    async def collect(message: Message) -> None:
        messages.append(message)

    await response(_http_scope(), receive, send or collect)
    return messages


@pytest.mark.asyncio
async def test_anthropic_response_waits_for_first_chunk_before_returning() -> None:
    ready = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        await ready.wait()
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'

    task = asyncio.create_task(
        anthropic_sse_streaming_response(
            body(),
            pre_start_error_response=_json_error,
            request_id="req_test",
        )
    )

    await asyncio.sleep(0)
    assert not task.done()

    ready.set()
    response = await asyncio.wait_for(task, timeout=1)
    assert isinstance(response, StreamingResponse)
    assert "message_start" in await _drain(response)


@pytest.mark.asyncio
async def test_anthropic_pre_start_provider_error_returns_non_200_json() -> None:
    response = await anthropic_sse_streaming_response(
        _body_raises(
            ExecutionFailure(
                kind=FailureKind.RATE_LIMIT,
                status_code=429,
                message="provider says slow down",
                retryable=True,
            )
        ),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["message"] == "provider says slow down"


@pytest.mark.asyncio
async def test_pre_start_failure_closes_body_before_response_release() -> None:
    lifecycle: list[str] = []

    class FailingBody:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise RuntimeError("provider failed")

        async def aclose(self) -> None:
            lifecycle.append("body_closed")

    async def release() -> None:
        lifecycle.append("lease_released")

    response = await anthropic_sse_streaming_response(
        FailingBody(),
        pre_start_error_response=_json_error,
        request_id="req_pre_start_order",
    )
    await bind_response_lifetime(response, release)

    assert lifecycle == ["body_closed", "lease_released"]


@pytest.mark.asyncio
async def test_terminal_execution_error_response_disables_client_retry() -> None:
    response = terminal_execution_error_response(
        status_code=429,
        content=anthropic_error_payload(
            error_type="rate_limit_error",
            message="provider says slow down",
        ),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    body = json.loads(bytes(response.body))
    assert body["error"] == {
        "type": "rate_limit_error",
        "message": "provider says slow down",
    }


@pytest.mark.asyncio
async def test_anthropic_post_start_exception_emits_terminal_error_frame() -> None:
    response = await anthropic_sse_streaming_response(
        _body_then_raises(
            ['event: message_start\ndata: {"type":"message_start"}\n\n'],
            RuntimeError("socket cut"),
        ),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )

    assert isinstance(response, StreamingResponse)
    text = await _drain(response)
    events = parse_sse_text(text)
    assert [event.event for event in events] == ["message_start", "error"]
    assert events[-1].data["error"]["message"] == "socket cut"


@pytest.mark.asyncio
async def test_non_streaming_response_releases_resource_before_return() -> None:
    release = AsyncMock()
    response = JSONResponse({"ok": True})

    result = await bind_response_lifetime(response, release)

    assert result is response
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_unmanaged_stream_is_closed_and_released_before_rejection() -> None:
    release = AsyncMock()
    close = AsyncMock()

    class Body:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            return "unreachable"

        async def aclose(self) -> None:
            await close()

    response = StreamingResponse(Body())

    with pytest.raises(TypeError, match="ManagedStreamingResponse"):
        await bind_response_lifetime(response, release)

    close.assert_awaited_once()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_after_normal_completion() -> None:
    release = AsyncMock()
    response = ManagedStreamingResponse(_body_chunks(["one", "two"]))

    result = await bind_response_lifetime(response, release)

    assert result is response
    release.assert_not_awaited()
    messages = await _serve(response)
    assert b"".join(message.get("body", b"") for message in messages) == b"onetwo"
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_after_body_failure() -> None:
    release = AsyncMock()
    response = ManagedStreamingResponse(
        _body_then_raises(["one"], RuntimeError("stream failed"))
    )
    await bind_response_lifetime(response, release)

    with pytest.raises(RuntimeError, match="stream failed"):
        await _serve(response)

    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_when_consumer_closes_early() -> None:
    release = AsyncMock()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "one"
            yield "two"
        finally:
            source_closed.set()

    response = await anthropic_sse_streaming_response(
        body(),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )
    assert isinstance(response, ManagedStreamingResponse)
    await bind_response_lifetime(response, release)

    await response.aclose()

    assert source_closed.is_set()
    release.assert_awaited_once()

    await response.aclose()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_response_releases_when_consumer_is_cancelled() -> None:
    release = AsyncMock()
    entered = asyncio.Event()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "one"
            entered.set()
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    response = ManagedStreamingResponse(body())
    await bind_response_lifetime(response, release)
    drain_task = asyncio.create_task(_serve(response))
    await entered.wait()

    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    assert source_closed.is_set()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_response_start_send_failure_closes_prefetched_tail_and_releases() -> (
    None
):
    release = AsyncMock()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "prefetched"
            yield "tail"
        finally:
            source_closed.set()

    response = await anthropic_sse_streaming_response(
        body(),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )
    assert isinstance(response, ManagedStreamingResponse)
    await bind_response_lifetime(response, release)

    async def fail_on_start(message: Message) -> None:
        assert message["type"] == "http.response.start"
        raise RuntimeError("send start failed")

    with pytest.raises(RuntimeError, match="send start failed"):
        await _serve(response, send=fail_on_start)

    assert source_closed.is_set()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_body_send_failure_closes_prefetched_tail_and_releases() -> None:
    release = AsyncMock()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "prefetched"
            yield "tail"
        finally:
            source_closed.set()

    response = await anthropic_sse_streaming_response(
        body(),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )
    assert isinstance(response, ManagedStreamingResponse)
    await bind_response_lifetime(response, release)

    async def fail_on_first_body(message: Message) -> None:
        if message["type"] == "http.response.body":
            raise RuntimeError("send body failed")

    with pytest.raises(RuntimeError, match="send body failed"):
        await _serve(response, send=fail_on_first_body)

    assert source_closed.is_set()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_asgi_23_correlation_boundary_preserves_response_start_cleanup() -> None:
    release = AsyncMock()
    source_closed = asyncio.Event()

    async def body() -> AsyncGenerator[str]:
        try:
            yield "prefetched"
            yield "tail"
        finally:
            source_closed.set()

    response = await anthropic_sse_streaming_response(
        body(),
        pre_start_error_response=_json_error,
        request_id="req_test",
    )
    assert isinstance(response, ManagedStreamingResponse)
    await bind_response_lifetime(response, release)

    async def app(scope, receive, send) -> None:
        await response(scope, receive, send)

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def fail_on_start(message: Message) -> None:
        assert message["type"] == "http.response.start"
        raise OSError("client disconnected")

    scope = _http_scope()
    scope["asgi"]["spec_version"] = "2.3"

    with pytest.raises(OSError, match="client disconnected"):
        await RequestCorrelationMiddleware(app)(scope, receive, fail_on_start)

    assert source_closed.is_set()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_body_source_failure_releases_response_lifetime() -> None:
    release = AsyncMock()
    response = ManagedStreamingResponse(_body_raises(RuntimeError("source failed")))
    await bind_response_lifetime(response, release)

    with pytest.raises(RuntimeError, match="source failed"):
        await _serve(response)

    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_close_and_release_completion() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    release_finished = asyncio.Event()

    class GatedBody:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    async def release() -> None:
        release_started.set()
        await allow_release.wait()
        release_finished.set()

    response = ManagedStreamingResponse(GatedBody())
    await bind_response_lifetime(response, release)
    closing = asyncio.create_task(response.aclose())
    await close_started.wait()

    closing.cancel()
    allow_close.set()
    await release_started.wait()
    closing.cancel()
    allow_release.set()

    with pytest.raises(asyncio.CancelledError):
        await closing

    assert close_finished.is_set()
    assert release_finished.is_set()


@pytest.mark.asyncio
async def test_repeated_pre_start_cancellation_waits_for_body_close() -> None:
    iteration_started = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class GatedPreStartBody:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            iteration_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    response_task = asyncio.create_task(
        anthropic_sse_streaming_response(
            GatedPreStartBody(),
            pre_start_error_response=_json_error,
            request_id="req_pre_start_cancel",
        )
    )
    await iteration_started.wait()

    response_task.cancel()
    await close_started.wait()
    response_task.cancel()
    allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await response_task

    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_cancellation_during_pre_start_error_cleanup_waits_for_close() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class FailingPreStartBody:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise RuntimeError("provider failed")

        async def aclose(self) -> None:
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    response_task = asyncio.create_task(
        anthropic_sse_streaming_response(
            FailingPreStartBody(),
            pre_start_error_response=_json_error,
            request_id="req_pre_start_error_cancel",
        )
    )
    await close_started.wait()

    response_task.cancel()
    allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await response_task

    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_cleanup_failures_are_trace_only_and_do_not_replace_success() -> None:
    class CloseFails:
        def __init__(self) -> None:
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            if self._yielded:
                raise StopAsyncIteration
            self._yielded = True
            return "ok"

        async def aclose(self) -> None:
            raise RuntimeError("secret close detail")

    release = AsyncMock(side_effect=RuntimeError("secret release detail"))
    response = ManagedStreamingResponse(CloseFails())
    await bind_response_lifetime(response, release)

    with (
        patch("free_claude_code.core.trace.trace_event") as close_trace,
        patch("free_claude_code.api.response_streams.trace_event") as release_trace,
    ):
        messages = await _serve(response)

    assert b"".join(message.get("body", b"") for message in messages) == b"ok"
    close_trace.assert_called_once()
    assert close_trace.call_args.kwargs["owner"] == "ManagedStreamingResponse"
    assert close_trace.call_args.kwargs["close_exc_type"] == "RuntimeError"
    assert release_trace.call_args.kwargs["operation"] == "release_resource"
    trace_blob = " ".join(
        str(call)
        for call in [*close_trace.call_args_list, *release_trace.call_args_list]
    )
    assert "secret close detail" not in trace_blob
    assert "secret release detail" not in trace_blob


@pytest.mark.asyncio
async def test_body_close_cancellation_propagates_without_releasing() -> None:
    class CloseIsCancelled:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise asyncio.CancelledError

    release = AsyncMock()
    response = ManagedStreamingResponse(CloseIsCancelled())
    await bind_response_lifetime(response, release)

    with pytest.raises(asyncio.CancelledError):
        await response.aclose()

    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_lease_release_cancellation_propagates() -> None:
    release = AsyncMock(side_effect=asyncio.CancelledError)
    response = ManagedStreamingResponse(_body_chunks([]))
    await bind_response_lifetime(response, release)

    with pytest.raises(asyncio.CancelledError):
        await response.aclose()

    release.assert_awaited_once()
