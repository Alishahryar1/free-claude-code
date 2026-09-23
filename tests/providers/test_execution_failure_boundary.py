"""Provider streams raise canonical failures after closing committed blocks."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.http import (
    ProviderAttemptScope,
    close_provider_stream,
)
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    SDKStreamDouble,
    immediate_admission,
    make_provider_config,
)


class _FailingStream(SDKStreamDouble[object]):
    def __init__(
        self,
        chunks: list[object],
        error: Exception | None,
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._close_error = close_error
        self.close_calls = 0
        super().__init__(self._iterate(), close=self.aclose)

    async def _iterate(self) -> AsyncIterator[object]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _chunk(*, content: str | None = None, finish_reason: str | None = None) -> object:
    delta = MagicMock(content=content, tool_calls=None, reasoning_content=None)
    choice = MagicMock(delta=delta, finish_reason=finish_reason)
    return MagicMock(choices=[choice], usage=None)


def _provider() -> NvidiaNimProvider:
    return NvidiaNimProvider(
        make_provider_config(
            api_key="test_key",
            base_url="https://test.api.nvidia.com/v1",
            rate_limit=10,
            rate_window=60,
        ),
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )


@pytest.mark.asyncio
async def test_committed_provider_failure_closes_block_then_raises_canonical_value() -> (
    None
):
    provider = _provider()
    request = make_messages_request(
        "test-model",
        messages=[],
        max_tokens=32,
    )
    # Crossing the recovery buffer's byte threshold makes the content block
    # downstream-visible before the failure, so its close prelude must escape.
    stream = _FailingStream(
        [_chunk(content="x" * 65_536)],
        RuntimeError("connection lost after commit"),
    )
    emitted: deque[str] = deque()

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        async for event in provider.stream_messages(
            request,
            request_id="req_committed_failure",
        ):
            emitted.append(event)

    events = parse_sse_text("".join(emitted))
    assert [event.event for event in events][-1] == "content_block_stop"
    assert not any(event.event in {"error", "message_stop"} for event in events)
    assert exc_info.value.kind is FailureKind.UPSTREAM
    assert exc_info.value.status_code == 502
    assert exc_info.value.retryable is False
    assert "connection lost after commit" in exc_info.value.message
    assert "Request ID: req_committed_failure" in exc_info.value.message


@pytest.mark.asyncio
async def test_openai_stream_close_failure_cannot_mask_execution_failure() -> None:
    provider = _provider()
    request = make_messages_request(
        "test-model",
        messages=[],
        max_tokens=32,
    )
    stream = _FailingStream(
        [],
        RuntimeError("original provider failure"),
        close_error=RuntimeError("cleanup api_key=SECRET"),
    )

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ),
        patch("free_claude_code.providers.http.trace_event") as trace_event,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_messages(
                request,
                request_id="req_close_failure",
            )
        ]

    assert stream.close_calls == 1
    assert exc_info.value.status_code == 502
    assert "original provider failure" in exc_info.value.message
    assert "cleanup" not in exc_info.value.message
    assert "SECRET" not in exc_info.value.message
    trace_event.assert_called_once_with(
        stage="provider",
        event="provider.stream.close_failed",
        source="provider",
        provider="NIM",
        request_id="req_close_failure",
        close_exc_type="RuntimeError",
        preserved_exc_type="ExecutionFailure",
    )
    assert "SECRET" not in repr(trace_event.call_args)


@pytest.mark.asyncio
async def test_stream_close_failure_without_active_error_is_observability_only() -> (
    None
):
    stream = _FailingStream(
        [],
        RuntimeError("unused"),
        close_error=RuntimeError("normal close failed"),
    )

    with patch("free_claude_code.providers.http.trace_event") as trace_event:
        await close_provider_stream(
            stream,
            active_error=None,
            provider_name="TEST",
            request_id="req_normal_close",
        )

    assert stream.close_calls == 1
    trace_event.assert_called_once_with(
        stage="provider",
        event="provider.stream.close_failed",
        source="provider",
        provider="TEST",
        request_id="req_normal_close",
        close_exc_type="RuntimeError",
        preserved_exc_type=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "close_error",
    [
        pytest.param(RuntimeError("cleanup api_key=SECRET"), id="exception"),
        pytest.param(
            asyncio.CancelledError("cleanup api_key=SECRET"),
            id="cleanup-cancellation",
        ),
    ],
)
async def test_attempt_scope_releases_attempt_when_resource_close_fails(
    close_error: BaseException,
) -> None:
    controller = immediate_admission()
    execution = controller.start_execution(request_id="req_scope_close")
    attempt = await execution.open_attempt(ProviderOperationKind.CONTINUATION)
    stream = _FailingStream(
        [],
        None,
        close_error=close_error,
    )
    scope = ProviderAttemptScope(
        attempt,
        provider_name="TEST",
        request_id="req_scope_close",
    )
    assert scope.retain(stream) is stream

    with patch("free_claude_code.providers.http.trace_event") as trace_event:
        await scope.aclose(active_error=ValueError("original"))
        await scope.aclose(active_error=None)

    replacement = await execution.open_attempt(ProviderOperationKind.CONTINUATION)
    await replacement.aclose()
    assert stream.close_calls == 1
    trace_event.assert_called_once_with(
        stage="provider",
        event="provider.stream.close_failed",
        source="provider",
        provider="TEST",
        request_id="req_scope_close",
        close_exc_type=type(close_error).__name__,
        preserved_exc_type="ValueError",
    )
    assert "SECRET" not in repr(trace_event.call_args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "close_error",
    [
        pytest.param(RuntimeError("cleanup api_key=SECRET"), id="exception"),
        pytest.param(
            asyncio.CancelledError("cleanup api_key=SECRET"),
            id="cleanup-cancellation",
        ),
    ],
)
async def test_completed_stream_close_failure_preserves_success_lifecycle(
    close_error: BaseException,
) -> None:
    provider = _provider()
    request = make_messages_request(
        "test-model",
        messages=[],
        max_tokens=32,
    )
    stream = _FailingStream(
        [_chunk(content="complete", finish_reason="stop")],
        None,
        close_error=close_error,
    )

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ),
        patch("free_claude_code.providers.http.trace_event") as trace_event,
    ):
        emitted = [
            event
            async for event in provider.stream_messages(
                request,
                request_id="req_successful_close_failure",
            )
        ]

    events = parse_sse_text("".join(emitted))
    assert events[-1].event == "message_stop"
    assert sum(event.event == "message_stop" for event in events) == 1
    assert not any(event.event == "error" for event in events)
    assert stream.close_calls == 1
    trace_event.assert_called_once_with(
        stage="provider",
        event="provider.stream.close_failed",
        source="provider",
        provider="NIM",
        request_id="req_successful_close_failure",
        close_exc_type=type(close_error).__name__,
        preserved_exc_type=None,
    )
    assert "SECRET" not in repr(trace_event.call_args)


@pytest.mark.asyncio
async def test_caller_cancellation_during_stream_close_propagates() -> None:
    class BlockingClose:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            self.started.set()
            await asyncio.Event().wait()

    stream = BlockingClose()
    with patch("free_claude_code.providers.http.trace_event") as trace_event:
        close_task = asyncio.create_task(
            close_provider_stream(
                stream,
                active_error=None,
                provider_name="TEST",
                request_id="req_cancelled_close",
            )
        )
        await asyncio.wait_for(stream.started.wait(), timeout=1)
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

    assert stream.close_calls == 1
    trace_event.assert_not_called()


@pytest.mark.asyncio
async def test_closing_public_openai_stream_closes_raw_stream_once() -> None:
    provider = _provider()
    request = make_messages_request(
        "test-model",
        messages=[],
        max_tokens=32,
    )
    raw_stream = _FailingStream(
        [
            _chunk(content="x" * 65_536),
            _chunk(content="done", finish_reason="stop"),
        ],
        None,
    )

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=raw_stream,
    ):
        stream = provider.stream_messages(request, request_id="req_early_close")
        await anext(stream)
        assert isinstance(stream, AsyncCloseable)
        await stream.aclose()

    assert raw_stream.close_calls == 1
