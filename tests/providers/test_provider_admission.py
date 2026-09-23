"""Provider-owned admission and coordinated recovery contracts."""

import asyncio
import ssl
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import patch

import httpx
import httpx2
import openai
import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.admission import (
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    ProviderAdmissionController,
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderExecutionState,
    ProviderOperationKind,
    _retry_after_seconds,
)
from free_claude_code.providers.failure_policy import ProviderRecoveryExhausted
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError


def _controller(
    *,
    provider_name: str = "TEST",
    rate_limit: int = 1_000_000,
    rate_window: float = 1.0,
    max_concurrency: int = 1_000,
    max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    base_delay: float = 0.0,
    max_delay: float = 0.0,
) -> ProviderAdmissionController:
    return ProviderAdmissionController(
        provider_name=provider_name,
        rate_limit=rate_limit,
        rate_window=rate_window,
        max_concurrency=max_concurrency,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=0.0,
    )


def _status_error(
    status: int,
    *,
    retry_after: str | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else None
    response = httpx.Response(status, request=request, headers=headers)
    return httpx.HTTPStatusError(
        f"upstream returned {status}",
        request=request,
        response=response,
    )


async def _open(execution: ProviderExecution) -> ProviderAttempt:
    return await execution.open_attempt(ProviderOperationKind.GENERATION)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ProviderAdmissionController(provider_name="TEST", rate_limit=0),
            "rate_limit",
        ),
        (
            lambda: ProviderAdmissionController(provider_name="TEST", rate_window=0),
            "rate_window",
        ),
        (
            lambda: ProviderAdmissionController(
                provider_name="TEST", max_concurrency=0
            ),
            "max_concurrency",
        ),
        (
            lambda: ProviderAdmissionController(provider_name="TEST", max_attempts=0),
            "max_attempts",
        ),
        (
            lambda: ProviderAdmissionController(provider_name="TEST", base_delay=-1),
            "base_delay",
        ),
        (
            lambda: ProviderAdmissionController(
                provider_name="TEST", base_delay=2, max_delay=1
            ),
            "max_delay",
        ),
    ],
)
def test_admission_rejects_invalid_configuration(
    factory: Callable[[], ProviderAdmissionController], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.asyncio
async def test_execution_exposes_one_bounded_attempt_budget() -> None:
    execution = _controller(max_attempts=2).start_execution()

    assert execution.max_attempts == 2
    assert execution.attempts_started == 0
    assert execution.attempts_remaining == 2
    assert execution.can_attempt
    first = await execution.open_attempt(ProviderOperationKind.GENERATION)
    await first.aclose()
    second = await execution.open_attempt(ProviderOperationKind.GENERATION)
    await second.aclose()
    assert not execution.can_attempt
    assert execution.attempts_remaining == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        await execution.open_attempt(ProviderOperationKind.GENERATION)


@pytest.mark.asyncio
async def test_execution_allows_only_one_active_physical_attempt() -> None:
    execution = _controller().start_execution()
    attempt = await _open(execution)

    with pytest.raises(RuntimeError, match="active attempt"):
        await _open(execution)

    await attempt.aclose()


@pytest.mark.asyncio
async def test_attempt_outcome_and_close_are_idempotent() -> None:
    controller = _controller(max_concurrency=1)
    execution = controller.start_execution()
    error = _status_error(400)

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        attempt = await _open(execution)
        first = await attempt.fail(error)
        repeated = await attempt.fail(_status_error(503))
        await attempt.accept()
        assert await attempt.correct(error) is ProviderCorrectionAction.FINAL
        await attempt.aclose()
        await attempt.aclose()

    assert not first.retryable
    assert not first.retry_allowed
    assert not repeated.retryable
    assert not repeated.retry_allowed
    resolved = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.attempt.resolved"
    ]
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == "rejected"

    replacement = await asyncio.wait_for(
        _open(controller.start_execution()),
        timeout=1,
    )
    await replacement.accept()
    await replacement.aclose()


@pytest.mark.asyncio
async def test_deterministic_correction_is_typed_and_final_at_exhaustion() -> None:
    execution = _controller(max_attempts=2).start_execution()
    error = _status_error(400)

    first = await _open(execution)
    assert await first.correct(error) is ProviderCorrectionAction.RETRY
    await first.aclose()
    assert execution.state is ProviderExecutionState.ACTIVE
    assert execution.attempts_remaining == 1

    final = await _open(execution)
    assert await final.correct(error) is ProviderCorrectionAction.FINAL
    await final.aclose()
    assert execution.state is ProviderExecutionState.FAILED
    assert execution.last_failure is error
    assert execution.attempts_remaining == 0


@pytest.mark.asyncio
async def test_successful_run_call_is_terminal() -> None:
    execution = _controller().start_execution()

    assert (
        await execution.run_call(
            lambda: asyncio.sleep(0, result="ok"),
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
        )
        == "ok"
    )
    assert execution.state is ProviderExecutionState.SUCCEEDED
    assert execution.attempts_remaining == 0
    with pytest.raises(RuntimeError, match="terminal"):
        await _open(execution)


@pytest.mark.asyncio
async def test_proactive_rate_limit_is_a_strict_rolling_window() -> None:
    controller = _controller(rate_limit=1, rate_window=0.04)

    first_execution = controller.start_execution()
    first = await first_execution.open_attempt(ProviderOperationKind.GENERATION)
    await first.accept()
    await first.aclose()

    started = time.monotonic()
    second_execution = controller.start_execution()
    second = await second_execution.open_attempt(ProviderOperationKind.GENERATION)
    elapsed = time.monotonic() - started
    await second.accept()
    await second.aclose()

    assert elapsed >= 0.03


@pytest.mark.asyncio
async def test_concurrency_bulkhead_limits_active_attempts() -> None:
    controller = _controller(max_concurrency=2)
    first = await controller.start_execution().open_attempt(
        ProviderOperationKind.GENERATION
    )
    second = await controller.start_execution().open_attempt(
        ProviderOperationKind.GENERATION
    )

    third_task = asyncio.create_task(
        controller.start_execution().open_attempt(ProviderOperationKind.GENERATION)
    )
    await asyncio.sleep(0)
    assert not third_task.done()

    await first.accept()
    await first.aclose()
    third = await asyncio.wait_for(third_task, timeout=1)

    await second.accept()
    await second.aclose()
    await third.accept()
    await third.aclose()


@pytest.mark.asyncio
async def test_attempt_cancellation_releases_concurrency() -> None:
    controller = _controller(max_concurrency=1)
    entered = asyncio.Event()

    async def never_finishes() -> None:
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        controller.start_execution().run_call(
            never_finishes,
            operation_kind=ProviderOperationKind.GENERATION,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (
        await asyncio.wait_for(
            controller.start_execution().run_call(
                lambda: asyncio.sleep(0, result="ok"),
                operation_kind=ProviderOperationKind.GENERATION,
            ),
            timeout=1,
        )
        == "ok"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [_status_error(503), ssl.SSLWantReadError("read would block")],
    ids=["http_503", "tls_want_read"],
)
async def test_run_call_uses_one_five_attempt_budget(error: Exception) -> None:
    controller = _controller()
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error)) as exc_info:
        await controller.start_execution().run_call(
            fail,
            operation_kind=ProviderOperationKind.GENERATION,
        )

    assert attempts == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert exc_info.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [_status_error(429), ssl.SSLWantReadError("read would block")],
    ids=["http_429", "tls_want_read"],
)
async def test_run_call_succeeds_without_multiplying_attempts(error: Exception) -> None:
    controller = _controller(max_concurrency=1)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise error
        return "recovered"

    assert (
        await asyncio.wait_for(
            controller.start_execution().run_call(
                recover,
                operation_kind=ProviderOperationKind.GENERATION,
            ),
            timeout=1,
        )
        == "recovered"
    )
    assert attempts == 3


@pytest.mark.asyncio
async def test_recovery_traces_keep_the_logical_request_id() -> None:
    controller = _controller(max_attempts=2)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(503)
        return "recovered"

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        result = await controller.start_execution(request_id="req_trace").run_call(
            recover,
            operation_kind=ProviderOperationKind.GENERATION,
        )

    assert result == "recovered"
    recovery_rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event", "").startswith("provider.recovery")
        or call.kwargs.get("event") == "provider.retry.scheduled"
    ]
    assert {row["event"] for row in recovery_rows} == {
        "provider.recovery.opened",
        "provider.recovery.probe",
        "provider.recovery.closed",
        "provider.retry.scheduled",
    }
    assert all(row["request_id"] == "req_trace" for row in recovery_rows)
    assert recovery_rows[-1]["outcome"] == "success"


@pytest.mark.asyncio
async def test_each_physical_call_has_one_correlated_trace_pair() -> None:
    controller = _controller(max_attempts=2)
    calls = 0

    async def recover() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _status_error(503)
        return "ok"

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        assert (
            await controller.start_execution(request_id="req_attempts").run_call(
                recover,
                operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
            )
            == "ok"
        )

    attempt_rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event", "").startswith("provider.attempt.")
    ]
    assert [row["event"] for row in attempt_rows] == [
        "provider.attempt.started",
        "provider.attempt.resolved",
        "provider.attempt.started",
        "provider.attempt.resolved",
    ]
    assert [row["attempt"] for row in attempt_rows] == [1, 1, 2, 2]
    assert [row.get("outcome") for row in attempt_rows] == [
        None,
        "retryable_failure",
        None,
        "accepted",
    ]
    assert {row["request_id"] for row in attempt_rows} == {"req_attempts"}
    assert {row["operation_kind"] for row in attempt_rows} == {
        ProviderOperationKind.MODEL_DISCOVERY.value
    }
    assert len({row["execution_id"] for row in attempt_rows}) == 1


@pytest.mark.asyncio
async def test_direct_exhaustion_trace_keeps_the_logical_request_id() -> None:
    controller = _controller(max_attempts=1)
    error = _status_error(503)

    async def fail() -> None:
        raise error

    with (
        patch("free_claude_code.providers.admission.trace_event") as trace,
        pytest.raises(httpx.HTTPStatusError),
    ):
        await controller.start_execution(request_id="req_terminal").run_call(
            fail,
            operation_kind=ProviderOperationKind.GENERATION,
        )

    rows = [call.kwargs for call in trace.call_args_list]
    assert {row["event"] for row in rows} == {
        "provider.attempt.started",
        "provider.attempt.resolved",
        "provider.recovery.opened",
        "provider.retry.exhausted",
    }
    assert all(row["request_id"] == "req_terminal" for row in rows)


@pytest.mark.asyncio
async def test_non_retryable_error_is_attempted_once() -> None:
    controller = _controller()
    attempts = 0

    async def reject() -> None:
        nonlocal attempts
        attempts += 1
        raise _status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await controller.start_execution().run_call(
            reject,
            operation_kind=ProviderOperationKind.GENERATION,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_openai_413_does_not_retry_sleep_or_open_recovery() -> None:
    controller = _controller()
    attempts = 0
    request = httpx2.Request("POST", "https://provider.test/chat/completions")
    response = httpx2.Response(
        413,
        request=request,
        headers={"retry-after": "237", "x-should-retry": "false"},
    )
    error = openai.APIStatusError(
        "Request too large for token rate limit",
        response=response,
        body={
            "error": {
                "type": "tokens",
                "code": "rate_limit_exceeded",
                "message": "Request requires 55940 tokens but limit is 8000",
            }
        },
    )

    async def reject() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with (
        patch("free_claude_code.providers.admission.asyncio.sleep") as sleep,
        patch("free_claude_code.providers.admission.trace_event") as trace,
        pytest.raises(openai.APIStatusError),
    ):
        await controller.start_execution(request_id="req_413").run_call(
            reject,
            operation_kind=ProviderOperationKind.GENERATION,
        )

    assert attempts == 1
    sleep.assert_not_awaited()
    assert all(
        call.kwargs.get("event") != "provider.recovery.opened"
        for call in trace.call_args_list
    )


@pytest.mark.asyncio
async def test_pre_response_protocol_failure_opens_coordinated_recovery() -> None:
    controller = _controller()
    execution = controller.start_execution()
    attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)

    decision = await attempt.fail(
        TruncatedProviderStreamError("stream ended before its first chunk")
    )
    assert decision.retryable
    assert decision.retry_allowed
    await attempt.aclose()

    probe = await execution.open_attempt(ProviderOperationKind.GENERATION)
    await probe.accept()
    await probe.aclose()


@pytest.mark.asyncio
async def test_provider_override_can_classify_retryable_semantics() -> None:
    controller = _controller()
    attempts = 0

    async def degraded() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(400)
        return "healthy"

    def classify(error: Exception) -> ExecutionFailure | None:
        del error
        return ExecutionFailure(
            kind=FailureKind.OVERLOADED,
            status_code=529,
            message="temporarily degraded",
            retryable=True,
        )

    assert (
        await controller.start_execution().run_call(
            degraded,
            operation_kind=ProviderOperationKind.GENERATION,
            provider_failure_override=classify,
        )
        == "healthy"
    )
    assert attempts == 2


@pytest.mark.asyncio
async def test_discovery_and_generation_share_provider_recovery_health() -> None:
    controller = _controller()
    discovery = controller.start_execution()
    discovery_attempt = await discovery.open_attempt(
        ProviderOperationKind.MODEL_DISCOVERY
    )

    assert (await discovery_attempt.fail(_status_error(503))).retry_allowed
    await discovery_attempt.aclose()

    generation = controller.start_execution()
    generation_wait = asyncio.create_task(
        generation.open_attempt(ProviderOperationKind.GENERATION)
    )
    await asyncio.sleep(0)
    assert not generation_wait.done()

    probe = await discovery.open_attempt(ProviderOperationKind.MODEL_DISCOVERY)
    await probe.accept()
    await probe.aclose()

    generation_attempt = await asyncio.wait_for(generation_wait, timeout=1)
    await generation_attempt.accept()
    await generation_attempt.aclose()


@pytest.mark.asyncio
async def test_one_leader_backs_off_while_followers_coalesce() -> None:
    controller = _controller(base_delay=2.0, max_delay=60.0)
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)
    follower = await _open(follower_execution)
    error = _status_error(429)

    assert (await leader.fail(error)).retry_allowed
    assert (await follower.fail(error)).retry_allowed
    await leader.aclose()
    await follower.aclose()

    real_sleep = asyncio.sleep
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    delays: list[float] = []

    async def controlled_sleep(delay: float) -> None:
        delays.append(delay)
        sleep_started.set()
        await release_sleep.wait()

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        side_effect=controlled_sleep,
    ):
        leader_probe_task = asyncio.create_task(_open(leader_execution))
        await sleep_started.wait()
        follower_attempt_task = asyncio.create_task(_open(follower_execution))
        await real_sleep(0)

        assert len(delays) == 1
        assert 1.9 <= delays[0] <= 2.0
        assert not follower_attempt_task.done()

        release_sleep.set()
        leader_probe = await asyncio.wait_for(leader_probe_task, timeout=1)
        await real_sleep(0)
        assert not follower_attempt_task.done()

        await leader_probe.accept()
        await leader_probe.aclose()
        resumed_follower = await asyncio.wait_for(follower_attempt_task, timeout=1)

    await resumed_follower.accept()
    await resumed_follower.aclose()


@pytest.mark.asyncio
async def test_cancelled_follower_leaves_recovery_episode() -> None:
    controller = _controller(base_delay=1.0, max_delay=1.0)
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)
    follower = await _open(follower_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    assert (await follower.fail(_status_error(503))).retry_allowed
    await leader.aclose()
    await follower.aclose()

    follower_wait = asyncio.create_task(_open(follower_execution))
    await asyncio.sleep(0)
    episode = controller._episode
    assert episode is not None
    assert follower_execution in episode.waiters

    follower_wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower_wait

    assert follower_execution not in episode.waiters


@pytest.mark.asyncio
async def test_cancelled_backoff_leader_transfers_to_a_waiter() -> None:
    controller = _controller(base_delay=2.0, max_delay=60.0)
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)
    follower = await _open(follower_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    assert (await follower.fail(_status_error(503))).retry_allowed
    await leader.aclose()
    await follower.aclose()

    real_sleep = asyncio.sleep
    first_sleep_started = asyncio.Event()
    second_sleep_started = asyncio.Event()
    release_second_sleep = asyncio.Event()
    sleep_calls = 0

    async def controlled_sleep(delay: float) -> None:
        nonlocal sleep_calls
        assert 1.9 <= delay <= 2.0
        sleep_calls += 1
        if sleep_calls == 1:
            first_sleep_started.set()
            await asyncio.Event().wait()
        second_sleep_started.set()
        await release_second_sleep.wait()

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        side_effect=controlled_sleep,
    ):
        leader_task = asyncio.create_task(_open(leader_execution))
        await first_sleep_started.wait()
        follower_task = asyncio.create_task(_open(follower_execution))
        await real_sleep(0)

        leader_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader_task

        await second_sleep_started.wait()
        release_second_sleep.set()
        probe = await asyncio.wait_for(follower_task, timeout=1)

    await probe.accept()
    await probe.aclose()


@pytest.mark.asyncio
async def test_stale_in_flight_success_cannot_close_recovery_episode() -> None:
    controller = _controller()
    leader_execution = controller.start_execution()
    stale_execution = controller.start_execution()
    leader = await _open(leader_execution)
    stale = await _open(stale_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    await leader.aclose()
    await stale.accept()
    await stale.aclose()

    waiting = asyncio.create_task(_open(controller.start_execution()))
    await asyncio.sleep(0)
    assert not waiting.done()

    probe = await _open(leader_execution)
    await probe.accept()
    await probe.aclose()
    resumed = await asyncio.wait_for(waiting, timeout=1)
    await resumed.accept()
    await resumed.aclose()


@pytest.mark.asyncio
async def test_abandoned_probe_transfers_leadership() -> None:
    controller = _controller()
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)
    follower = await _open(follower_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    assert (await follower.fail(_status_error(503))).retry_allowed
    await leader.aclose()
    await follower.aclose()

    abandoned_probe = await _open(leader_execution)
    follower_probe_task = asyncio.create_task(_open(follower_execution))
    await asyncio.sleep(0)
    assert not follower_probe_task.done()

    await abandoned_probe.aclose()
    follower_probe = await asyncio.wait_for(follower_probe_task, timeout=1)
    await follower_probe.accept()
    await follower_probe.aclose()


@pytest.mark.asyncio
async def test_cancelled_probe_resolution_remains_abandonable() -> None:
    controller = _controller(max_concurrency=1)
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    await leader.aclose()

    probe = await _open(leader_execution)
    await controller._condition.acquire()
    resolution = asyncio.create_task(probe.accept())
    await asyncio.sleep(0)
    resolution.cancel()
    controller._condition.release()
    with pytest.raises(asyncio.CancelledError):
        await resolution

    follower = asyncio.create_task(_open(follower_execution))
    await probe.aclose()
    replacement = await asyncio.wait_for(follower, timeout=1)
    await replacement.accept()
    await replacement.aclose()


@pytest.mark.asyncio
async def test_cancelled_probe_close_releases_slot_and_transfers_leadership() -> None:
    controller = _controller(max_concurrency=1)
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    await leader.aclose()

    probe = await _open(leader_execution)
    await controller._condition.acquire()
    close = asyncio.create_task(probe.aclose())
    await asyncio.sleep(0)
    close.cancel()
    controller._condition.release()
    with pytest.raises(asyncio.CancelledError):
        await close

    replacement = await asyncio.wait_for(
        _open(follower_execution),
        timeout=1,
    )
    await replacement.accept()
    await replacement.aclose()


@pytest.mark.asyncio
async def test_non_retryable_probe_closes_episode_and_only_rejects_leader() -> None:
    controller = _controller()
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)
    follower = await _open(follower_execution)

    assert (await leader.fail(_status_error(503))).retry_allowed
    assert (await follower.fail(_status_error(503))).retry_allowed
    await leader.aclose()
    await follower.aclose()

    probe = await _open(leader_execution)
    follower_task = asyncio.create_task(_open(follower_execution))
    await asyncio.sleep(0)
    assert not follower_task.done()

    decision = await probe.fail(_status_error(400))
    assert not decision.retryable
    assert not decision.retry_allowed
    await probe.aclose()
    resumed = await asyncio.wait_for(follower_task, timeout=1)
    await resumed.accept()
    await resumed.aclose()


@pytest.mark.asyncio
async def test_probe_exhaustion_fails_waiters_and_opens_after_cooldown() -> None:
    controller = _controller(
        max_attempts=2,
        base_delay=0.02,
        max_delay=0.02,
    )
    leader_execution = controller.start_execution()
    follower_execution = controller.start_execution()
    leader = await _open(leader_execution)
    follower = await _open(follower_execution)
    error = _status_error(429)

    assert (await leader.fail(error)).retry_allowed
    assert (await follower.fail(error)).retry_allowed
    await leader.aclose()
    await follower.aclose()

    follower_wait = asyncio.create_task(_open(follower_execution))
    probe = await _open(leader_execution)
    assert not (await probe.fail(error)).retry_allowed
    await probe.aclose()

    with pytest.raises(ProviderRecoveryExhausted) as waiter_error:
        await follower_wait
    assert waiter_error.value.last_error is error

    with pytest.raises(ProviderRecoveryExhausted):
        await _open(controller.start_execution())

    await asyncio.sleep(0.03)
    recovery = await _open(controller.start_execution())
    await recovery.accept()
    await recovery.aclose()


@pytest.mark.asyncio
async def test_exhausted_waiter_cannot_join_a_later_recovery_generation() -> None:
    controller = _controller(
        max_attempts=2,
        base_delay=0.01,
        max_delay=0.01,
    )
    leader_execution = controller.start_execution()
    delayed_waiter_execution = controller.start_execution()
    leader = await _open(leader_execution)
    delayed_waiter = await _open(delayed_waiter_execution)
    error = _status_error(503)

    assert (await leader.fail(error)).retry_allowed
    assert (await delayed_waiter.fail(error)).retry_allowed
    await leader.aclose()
    await delayed_waiter.aclose()

    probe = await _open(leader_execution)
    assert not (await probe.fail(error)).retry_allowed
    await probe.aclose()

    await asyncio.sleep(0.02)
    later = await _open(controller.start_execution())
    await later.accept()
    await later.aclose()

    with pytest.raises(ProviderRecoveryExhausted) as exc_info:
        await _open(delayed_waiter_execution)
    assert exc_info.value.last_error is error


@pytest.mark.asyncio
async def test_late_in_flight_failure_keeps_exhausted_generation_outcome() -> None:
    controller = _controller(
        max_attempts=2,
        base_delay=0.01,
        max_delay=0.01,
    )
    leader_execution = controller.start_execution()
    stale_execution = controller.start_execution()
    leader = await _open(leader_execution)
    stale = await _open(stale_execution)
    error = _status_error(503)

    assert (await leader.fail(error)).retry_allowed
    await leader.aclose()
    probe = await _open(leader_execution)
    assert not (await probe.fail(error)).retry_allowed
    await probe.aclose()

    stale_decision = await stale.fail(_status_error(503))
    assert stale_decision.retryable
    assert not stale_decision.retry_allowed
    await stale.aclose()
    await asyncio.sleep(0.02)

    with pytest.raises(ProviderRecoveryExhausted) as exc_info:
        await _open(stale_execution)
    assert exc_info.value.last_error is error


@pytest.mark.asyncio
async def test_retry_after_is_a_minimum_backoff() -> None:
    controller = _controller(max_attempts=2)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(429, retry_after="7")
        return "ok"

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        return_value=None,
    ) as sleep:
        assert (
            await controller.start_execution().run_call(
                recover,
                operation_kind=ProviderOperationKind.GENERATION,
            )
            == "ok"
        )

    sleep.assert_awaited_once()
    await_args = sleep.await_args
    assert await_args is not None
    assert 6.9 <= await_args.args[0] <= 7.0


def test_retry_after_accepts_http_date_and_rejects_invalid_values() -> None:
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=10), usegmt=True)

    parsed = _retry_after_seconds(_status_error(429, retry_after=future))

    assert parsed is not None
    assert 8 <= parsed <= 10
    assert _retry_after_seconds(_status_error(429, retry_after="invalid")) is None
    assert _retry_after_seconds(_status_error(429, retry_after="nan")) is None
    assert _retry_after_seconds(_status_error(429, retry_after="inf")) is None


@pytest.mark.asyncio
async def test_provider_controllers_do_not_share_recovery_state() -> None:
    first = _controller(provider_name="FIRST")
    second = _controller(provider_name="SECOND")
    first_execution = first.start_execution()
    first_attempt = await _open(first_execution)

    assert (await first_attempt.fail(_status_error(503))).retry_allowed
    await first_attempt.aclose()

    independent = await asyncio.wait_for(
        _open(second.start_execution()),
        timeout=1,
    )
    await independent.accept()
    await independent.aclose()

    probe = await _open(first_execution)
    await probe.accept()
    await probe.aclose()
