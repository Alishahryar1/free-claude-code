"""Provider stream commit-boundary and recovery policy."""

from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
    RecoveryHoldbackBuffer,
)


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

    assert holdback.push("event: content_block_start\n\n") == []
    now[0] += 0.74
    assert holdback.push("event: content_block_delta\n\n") == []
    assert not holdback.committed

    now[0] += 0.01
    assert holdback.push("event: content_block_stop\n\n") == [
        "event: content_block_start\n\n",
        "event: content_block_delta\n\n",
        "event: content_block_stop\n\n",
    ]
    assert holdback.committed
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

    assert holdback.flush() == []
