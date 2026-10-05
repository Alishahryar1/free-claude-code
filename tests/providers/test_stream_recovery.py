"""Provider stream commit-boundary and recovery policy."""

import asyncio

import pytest

from free_claude_code.core.stream_delivery import StreamDeliveryState
from free_claude_code.providers.stream_recovery import (
    HoldbackSignal,
    RecoveryController,
    RecoveryFailureAction,
    RecoveryHoldbackBuffer,
)


def test_exhausted_hidden_group_cannot_fall_through_to_salvage():
    controller = RecoveryController(StreamDeliveryState())
    controller.push("hidden")
    controller.flush()
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
        attempts_remaining=0,
    )
    assert decision.action == RecoveryFailureAction.FINAL_ERROR


def test_early_retry_discards_uncommitted_holdback() -> None:
    controller = RecoveryController()

    assert controller.push("hidden") == []
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable
    assert decision.has_buffered
    assert not controller.committed
    assert not controller.has_buffered
    assert controller.flush() == []


def test_early_retry_requires_remaining_execution_budget() -> None:
    controller = RecoveryController()
    assert controller.push("hidden") == []

    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable
    assert controller.has_buffered


def test_last_attempt_is_reserved_for_partial_output_recovery() -> None:
    controller = RecoveryController()
    assert controller.push("partial") == []

    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.has_buffered
    assert controller.has_buffered


def test_create_failure_is_owned_by_admission_not_stream_recovery() -> None:
    decision = RecoveryController().advance_failure(
        retryable=True,
        stream_opened=False,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable


def test_statusless_transient_api_error_allows_early_retry() -> None:
    decision = RecoveryController().advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable


def test_committed_output_allows_midstream_recovery() -> None:
    controller = RecoveryController()

    assert controller.push("event: content_block_delta\n\n") == []
    assert controller.flush() == ["event: content_block_delta\n\n"]
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.retryable
    assert decision.committed
    assert controller.flush_uncommitted(decision) == []


def test_uncommitted_complete_tool_can_be_salvaged() -> None:
    controller = RecoveryController()

    assert controller.push("event: content_block_delta\n\n") == []
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert not decision.committed
    assert decision.has_buffered
    assert controller.flush_uncommitted(decision) == ["event: content_block_delta\n\n"]
    assert controller.committed
    assert not controller.has_buffered


def test_non_retryable_error_is_final() -> None:
    decision = RecoveryController().advance_failure(
        retryable=False,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert not decision.retryable


def test_holdback_buffers_until_delay_then_commits() -> None:
    now = [10.0]
    holdback = RecoveryHoldbackBuffer(holdback_seconds=0.75, now=lambda: now[0])

    assert holdback.remaining_delay is None
    assert holdback.push("event: content_block_start\n\n") == []
    now[0] += 0.74
    assert holdback.push("event: content_block_delta\n\n") == []
    assert not holdback.committed
    assert holdback.remaining_delay == pytest.approx(0.01)

    now[0] += 0.01
    assert holdback.push("event: content_block_stop\n\n") == [
        "event: content_block_start\n\n",
        "event: content_block_delta\n\n",
        "event: content_block_stop\n\n",
    ]
    assert holdback.committed
    assert holdback.remaining_delay is None
    assert holdback.push("event: message_stop\n\n") == ["event: message_stop\n\n"]


def test_holdback_flushes_at_internal_buffer_cap() -> None:
    holdback = RecoveryHoldbackBuffer(max_bytes=5, now=lambda: 1.0)

    assert holdback.push("ab") == []
    assert holdback.push("cde") == ["ab", "cde"]
    assert holdback.committed


def test_holdback_discard_drops_uncommitted_events() -> None:
    holdback = RecoveryHoldbackBuffer(now=lambda: 1.0)

    assert holdback.push("hidden") == []
    holdback.discard()

    assert holdback.remaining_delay is None
    assert holdback.flush() == []


def test_new_attempt_deadline_starts_with_its_first_buffered_event(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(now=lambda: now[0]),
    )
    recovery = RecoveryController()
    recovery.push("old")
    now[0] = 10.5
    decision = recovery.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )
    assert decision.action is RecoveryFailureAction.EARLY_RETRY
    assert recovery.remaining_delay is None
    now[0] = 20.0
    recovery.push("new")
    now[0] = 20.7
    assert recovery.remaining_delay == pytest.approx(0.05)
    now[0] = 22.0
    assert recovery.remaining_delay == 0
    assert recovery.flush() == ["new"]
    recovery.discard()
    assert recovery.remaining_delay is None
    assert recovery.push("continuation") == ["continuation"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [None, TimeoutError("upstream"), StopAsyncIteration()]
)
async def test_deadline_preserves_the_pending_read_and_its_outcome(monkeypatch, error):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(holdback_seconds=0.01),
    )
    recovery = RecoveryController()
    recovery.push("buffered")
    release = asyncio.Event()
    reads = []

    class Source:
        async def __anext__(self):
            reads.append("read")
            await release.wait()
            if error is not None:
                raise error
            return "next"

        def __aiter__(self):
            return self

    async with recovery.read_stream(Source()) as stream:
        assert await asyncio.wait_for(anext(stream), 1) is HoldbackSignal.EXPIRED
        assert reads == ["read"]
        assert recovery.flush() == ["buffered"]
        release.set()
        if error is None:
            assert await anext(stream) == "next"
        else:
            with pytest.raises(type(error)) as caught:
                await anext(stream)
            assert caught.value is error
        assert reads == ["read"]


@pytest.mark.asyncio
async def test_events_without_output_cannot_extend_a_buffer_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(now=lambda: now[0]),
    )
    recovery = RecoveryController()
    recovery.push("held")
    reads = []

    async def source():
        reads.append("first")
        now[0] = 1.0
        yield "ignored"
        reads.append("second")
        yield "next"

    upstream = source()
    try:
        async with recovery.read_stream(upstream) as stream:
            assert await anext(stream) == "ignored"
            assert await anext(stream) is HoldbackSignal.EXPIRED
            assert reads == ["first"]
            assert recovery.flush() == ["held"]
            assert await anext(stream) == "next"
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_empty_buffer_does_not_wake_or_read_ahead():
    recovery = RecoveryController()
    entered = asyncio.Event()
    release = asyncio.Event()
    read_count = 0

    async def source():
        nonlocal read_count
        while True:
            read_count += 1
            entered.set()
            await release.wait()
            yield "next"

    upstream = source()
    read = None
    try:
        async with recovery.read_stream(upstream) as stream:
            read = asyncio.create_task(anext(stream))
            await asyncio.wait_for(entered.wait(), 1)
            assert not read.done()
            release.set()
            assert await read == "next"
            assert read_count == 1
    finally:
        release.set()
        if read is not None:
            read.cancel()
            await asyncio.gather(read, return_exceptions=True)
        await upstream.aclose()


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_read_before_leaving_scope():
    recovery = RecoveryController()
    recovery.push("held")
    entered = asyncio.Event()
    closing = asyncio.Event()
    close_release = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closing.set()
            await close_release.wait()
            closed.set()

    async def consume():
        async with recovery.read_stream(source()) as stream:
            await anext(stream)

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        consumer.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        consumer.cancel()
        await asyncio.sleep(0)
        assert not consumer.done()
        assert not closed.is_set()
    finally:
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 1)
    assert closed.is_set()
