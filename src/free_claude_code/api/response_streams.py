"""FastAPI streaming response wrappers for public API wire formats."""

import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
)
from typing import Literal

from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.responses import ContentStream
from starlette.types import Receive, Scope, Send

from free_claude_code.core.anthropic import anthropic_error_type_for_failure
from free_claude_code.core.anthropic.streaming import (
    ANTHROPIC_SSE_RESPONSE_HEADERS,
    anthropic_terminal_error_frame,
    anthropic_terminal_failure_frame,
)
from free_claude_code.core.async_iterators import try_close_async_iterator
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    committed_response_failure_frame,
    openai_error_type_for_failure,
)
from free_claude_code.core.trace import close_stream_input, trace_event

TERMINAL_EXECUTION_ERROR_HEADERS = {"x-should-retry": "false"}

PreStartErrorResponse = Callable[[BaseException], Response]
TerminalFrameEmitter = Callable[[str, str, BaseException], str]
TerminalFailureObserver = Callable[[BaseException], None]
ReleaseResponseResource = Callable[[], Awaitable[None]]
WireApi = Literal["messages", "responses"]


class EmptyStreamError(RuntimeError):
    """Raised when a public stream ends before emitting any protocol chunk."""


class ManagedStreamingResponse(StreamingResponse):
    """Own body closure and one response-scoped runtime release callback."""

    def __init__(
        self,
        content: ContentStream,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
    ) -> None:
        super().__init__(
            content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )
        self._release: ReleaseResponseResource | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    def bind_release(self, release: ReleaseResponseResource) -> None:
        """Bind the resource retained for this response before ASGI execution."""
        if self._release is not None:
            raise RuntimeError("A response resource release is already bound.")
        if self._cleanup_task is not None:
            raise RuntimeError("Cannot bind a resource after response cleanup started.")
        self._release = release

    async def aclose(self) -> None:
        """Close the body and release its runtime resource exactly once."""
        await self._close(preserved_error=None)

    async def _close(self, *, preserved_error: BaseException | None) -> None:
        task = self._cleanup_task
        if task is None:
            task = asyncio.create_task(
                self._cleanup(preserved_error=preserved_error),
                name="fcc-api-response-cleanup",
            )
            self._cleanup_task = task
        await _wait_for_cleanup(task)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        preserved_error: BaseException | None = None
        try:
            await super().__call__(scope, receive, send)
        except BaseException as exc:
            preserved_error = exc
            raise
        finally:
            await self._close(preserved_error=preserved_error)

    async def _cleanup(self, *, preserved_error: BaseException | None) -> None:
        try:
            await close_stream_input(
                self.body_iterator,
                owner="ManagedStreamingResponse",
                source="api",
                preserved_error=preserved_error,
            )
        except Exception as exc:
            _trace_response_cleanup_failure("close_body", exc)

        release = self._release
        if release is None:
            return
        try:
            await release()
        except Exception as exc:
            _trace_response_cleanup_failure("release_resource", exc)


async def _wait_for_cleanup(task: asyncio.Task[None]) -> None:
    """Wait through repeated caller cancellation, then restore cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc

    # Ordinary defensive failures are trace-only; cancellation remains control flow.
    try:
        task.result()
    except asyncio.CancelledError:
        if cancellation is not None:
            raise cancellation from None
        raise
    except Exception as exc:
        _trace_response_cleanup_failure("cleanup_task", exc)

    if cancellation is not None:
        raise cancellation


def _trace_response_cleanup_failure(operation: str, exc: BaseException) -> None:
    trace_event(
        stage="egress",
        event="free_claude_code.api.response.cleanup_failed",
        source="api",
        operation=operation,
        exc_type=type(exc).__name__,
    )


async def bind_response_lifetime(
    response: object,
    release: ReleaseResponseResource,
) -> object:
    """Retain a runtime resource until a response body is fully consumed."""
    if isinstance(response, ManagedStreamingResponse):
        response.bind_release(release)
        return response
    if isinstance(response, StreamingResponse):
        error = TypeError("Streaming API responses must use ManagedStreamingResponse.")
        try:
            await close_stream_input(
                response.body_iterator,
                owner="bind_response_lifetime",
                source="api",
                preserved_error=error,
            )
        finally:
            await release()
        raise error
    await release()
    return response


def terminal_execution_error_response(
    *, status_code: int, content: JsonObject
) -> JSONResponse:
    """Return a final provider-execution error without enabling client retries."""
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=dict(TERMINAL_EXECUTION_ERROR_HEADERS),
    )


def trace_terminal_execution_error(
    *,
    wire_api: WireApi,
    request_id: str,
    status_code: int,
    error_type: str,
    error: BaseException | None = None,
) -> None:
    """Record one correlated terminal-execution decision at the HTTP boundary."""
    fields: dict[str, object] = {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": wire_api,
        "request_id": request_id,
        "status_code": status_code,
        "error_type": error_type,
        "client_should_retry": False,
    }
    failure = find_execution_failure(error) if error is not None else None
    if error is not None:
        fields["exc_type"] = type(failure or error).__name__
    if failure is not None:
        fields["failure_kind"] = failure.kind.value
        fields["provider_retryable"] = failure.retryable
    trace_event(**fields)


async def _first_chunk_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
    pre_start_error_response: PreStartErrorResponse,
    terminal_frame: TerminalFrameEmitter | None,
    terminal_failure_observer: TerminalFailureObserver | None,
) -> Response:
    try:
        first_chunk = await anext(body)
    except StopAsyncIteration:
        error = EmptyStreamError("Stream ended before emitting a response.")
        await _close_pre_start_body(body, preserved_error=error)
        return pre_start_error_response(error)
    except GeneratorExit as exc:
        await _close_pre_start_body(body, preserved_error=exc)
        raise
    except asyncio.CancelledError as exc:
        await _close_pre_start_body(body, preserved_error=exc)
        raise
    except BaseExceptionGroup as exc:
        await _close_pre_start_body(body, preserved_error=exc)
        return pre_start_error_response(exc)
    except Exception as exc:
        await _close_pre_start_body(body, preserved_error=exc)
        return pre_start_error_response(exc)

    return ManagedStreamingResponse(
        _PrefetchedStream(
            first_chunk,
            body,
            terminal_frame=terminal_frame,
            terminal_failure_observer=terminal_failure_observer,
        ),
        media_type="text/event-stream",
        headers=dict(headers),
    )


async def _close_pre_start_body(
    body: AsyncIterator[str],
    *,
    preserved_error: BaseException,
) -> None:
    task = asyncio.create_task(
        close_stream_input(
            body,
            owner="first_chunk_streaming_response",
            source="api",
            preserved_error=preserved_error,
        ),
        name="fcc-api-pre-start-stream-cleanup",
    )
    await _wait_for_cleanup(task)


class _PrefetchedStream(AsyncIterator[str]):
    """Replay one prefetched frame while retaining ownership of the tail."""

    def __init__(
        self,
        first_chunk: str,
        body: AsyncIterator[str],
        *,
        terminal_frame: TerminalFrameEmitter | None,
        terminal_failure_observer: TerminalFailureObserver | None,
    ) -> None:
        self._first_chunk: str | None = first_chunk
        self._initial_chunk = first_chunk
        self._latest_chunk = first_chunk
        self._body = body
        self._terminal_frame = terminal_frame
        self._terminal_failure_observer = terminal_failure_observer
        self._done = False
        self._closed = False

    def __aiter__(self) -> _PrefetchedStream:
        return self

    async def __anext__(self) -> str:
        if self._closed or self._done:
            raise StopAsyncIteration
        if self._first_chunk is not None:
            first_chunk = self._first_chunk
            self._first_chunk = None
            return first_chunk
        try:
            chunk = await anext(self._body)
            self._latest_chunk = chunk
            return chunk
        except StopAsyncIteration:
            self._done = True
            raise
        except BaseExceptionGroup as exc:
            return self._terminal_chunk(find_execution_failure(exc) or exc)
        except Exception as exc:
            return self._terminal_chunk(exc)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._done = True
        close_error = await try_close_async_iterator(self._body)
        if close_error is not None:
            raise close_error

    def _terminal_chunk(self, exc: BaseException) -> str:
        terminal_frame = self._terminal_frame
        if terminal_frame is None:
            raise exc
        self._done = True
        if self._terminal_failure_observer is not None:
            self._terminal_failure_observer(exc)
        return terminal_frame(self._initial_chunk, self._latest_chunk, exc)


async def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    pre_start_error_response: PreStartErrorResponse,
    request_id: str,
) -> Response:
    """Return a streaming response for Anthropic-style SSE streams."""
    return await _first_chunk_streaming_response(
        body,
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
        pre_start_error_response=pre_start_error_response,
        terminal_frame=_anthropic_terminal_frame,
        terminal_failure_observer=lambda exc: _trace_anthropic_terminal_failure(
            exc,
            request_id=request_id,
        ),
    )


def _anthropic_terminal_frame(
    _first_chunk: str,
    _latest_chunk: str,
    exc: BaseException,
) -> str:
    failure = find_execution_failure(exc)
    if failure is not None:
        return anthropic_terminal_failure_frame(failure)
    return anthropic_terminal_error_frame(safe_exception_message(exc))


def _trace_anthropic_terminal_failure(
    exc: BaseException,
    *,
    request_id: str,
) -> None:
    failure = find_execution_failure(exc)
    trace_terminal_execution_error(
        wire_api="messages",
        request_id=request_id,
        status_code=failure.status_code if failure is not None else 500,
        error_type=(
            anthropic_error_type_for_failure(failure)
            if failure is not None
            else "api_error"
        ),
        error=exc,
    )


async def openai_responses_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
    pre_start_error_response: PreStartErrorResponse,
    request_id: str,
) -> Response:
    """Return a streaming response for OpenAI Responses-style SSE."""
    return await _first_chunk_streaming_response(
        body,
        headers=headers,
        pre_start_error_response=pre_start_error_response,
        terminal_frame=committed_response_failure_frame,
        terminal_failure_observer=lambda exc: _trace_responses_terminal_failure(
            exc,
            request_id=request_id,
        ),
    )


def _trace_responses_terminal_failure(
    exc: BaseException,
    *,
    request_id: str,
) -> None:
    failure = find_execution_failure(exc)
    trace_terminal_execution_error(
        wire_api="responses",
        request_id=request_id,
        status_code=failure.status_code if failure is not None else 500,
        error_type=(
            openai_error_type_for_failure(failure)
            if failure is not None
            else "api_error"
        ),
        error=exc,
    )
