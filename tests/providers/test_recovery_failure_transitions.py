"""What one retryable failure does to its model's recovery episode."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from httpx import HTTPStatusError

from free_claude_code.providers import admission
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderAttempt,
    ProviderExecution,
    ProviderExecutionState,
    ProviderOperationKind,
)
from free_claude_code.providers.recovery_errors import ProviderRecoveryExhausted
from tests.providers.test_provider_admission import _status_error

_RECOVERY_EVENTS = {
    "provider.recovery.opened",
    "provider.recovery.cooldown",
    "provider.retry.scheduled",
    "provider.retry.coalesced",
    "provider.retry.exhausted",
}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    now = SimpleNamespace(value=100.0)
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now.value))
    return now


def _controller(*, max_attempts: int = 3) -> ProviderAdmissionController:
    return ProviderAdmissionController(
        provider_name="TEST",
        max_attempts=max_attempts,
        base_delay=2.0,
        max_delay=60.0,
        jitter=0.0,
    )


async def _attempt(execution: ProviderExecution):
    return await execution.open_attempt(ProviderOperationKind.GENERATION)


async def _fail(attempt: ProviderAttempt, error: HTTPStatusError):
    """Fail one attempt and return its decision and the recovery traces it made."""
    with patch.object(admission, "trace_event") as trace:
        decision = await attempt.fail(error)
    await attempt.aclose()
    rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs["event"] in _RECOVERY_EVENTS
    ]
    return decision, rows


@pytest.mark.asyncio
async def test_first_failure_opens_an_episode_led_by_the_failing_execution(
    clock: SimpleNamespace,
):
    controller = _controller()
    execution = controller.start_execution(model="m")
    error = _status_error(503)

    decision, rows = await _fail(await _attempt(execution), error)

    assert decision.retry_allowed
    assert [row["event"] for row in rows] == [
        "provider.recovery.opened",
        "provider.retry.scheduled",
    ]
    assert rows[1]["delay_s"] == 2.0
    episode = controller._episodes["m"]
    assert episode.leader is execution
    assert episode.ready_at == 102.0
    assert episode.last_error is error
    assert not episode.probe_active
    assert episode.terminal_until is None


@pytest.mark.asyncio
async def test_failed_probe_schedules_the_next_probe_with_more_backoff(
    clock: SimpleNamespace,
):
    controller = _controller()
    execution = controller.start_execution(model="m")
    await _fail(await _attempt(execution), _status_error(503))
    generation = controller._episodes["m"].generation
    clock.value = 102.0
    probe = await _attempt(execution)
    error = _status_error(429)

    decision, rows = await _fail(probe, error)

    assert decision.retry_allowed
    assert [row["event"] for row in rows] == ["provider.retry.scheduled"]
    assert rows[0]["delay_s"] == 4.0
    episode = controller._episodes["m"]
    assert episode.generation == generation
    assert episode.leader is execution
    assert episode.ready_at == 106.0
    assert episode.last_error is error
    assert not episode.probe_active


@pytest.mark.asyncio
async def test_failure_outside_the_probe_waits_for_the_leader(clock: SimpleNamespace):
    controller = _controller()
    leader = controller.start_execution(model="m")
    follower = controller.start_execution(model="m")
    first = await _attempt(leader)
    second = await _attempt(follower)
    await _fail(first, _status_error(503))

    decision, rows = await _fail(second, _status_error(502))

    assert decision.retry_allowed
    assert [row["event"] for row in rows] == ["provider.retry.coalesced"]
    episode = controller._episodes["m"]
    assert episode.leader is leader
    assert episode.waiters == {follower}
    assert episode.ready_at == 102.0


@pytest.mark.asyncio
async def test_last_attempt_without_an_episode_opens_an_exhausted_one(
    clock: SimpleNamespace,
):
    controller = _controller(max_attempts=1)
    execution = controller.start_execution(model="m", request_id="req")
    error = _status_error(503)

    decision, rows = await _fail(await _attempt(execution), error)

    assert not decision.retry_allowed
    assert decision.error is error
    assert [row["event"] for row in rows] == [
        "provider.recovery.opened",
        "provider.retry.exhausted",
    ]
    assert rows[0]["request_id"] == "req"
    assert rows[1]["episode_exhausted"] is True
    episode = controller._episodes["m"]
    assert episode.leader is None
    assert episode.terminal_until == 102.0
    assert not episode.reports_cooldown
    assert execution.state is ProviderExecutionState.FAILED


@pytest.mark.asyncio
async def test_last_probe_exhausts_the_episode_and_fails_its_waiters(
    clock: SimpleNamespace,
):
    controller = _controller(max_attempts=2)
    leader = controller.start_execution(model="m")
    follower = controller.start_execution(model="m")
    first = await _attempt(leader)
    second = await _attempt(follower)
    await _fail(first, _status_error(503))
    await _fail(second, _status_error(503))
    clock.value = 102.0
    probe = await _attempt(leader)
    error = _status_error(429)

    decision, rows = await _fail(probe, error)

    assert not decision.retry_allowed
    assert [row["event"] for row in rows] == ["provider.retry.exhausted"]
    assert rows[0]["episode_exhausted"] is True
    episode = controller._episodes["m"]
    assert episode.terminal_until == 106.0
    assert episode.last_error is error
    assert episode.leader is None
    assert not episode.waiters
    assert isinstance(follower.last_failure, ProviderRecoveryExhausted)
    assert follower.last_failure.last_error is error
    assert follower.last_failure.retry_not_before is None


@pytest.mark.asyncio
async def test_last_attempt_outside_the_probe_leaves_the_episode_alone(
    clock: SimpleNamespace,
):
    controller = _controller(max_attempts=1)
    leader = controller.start_execution(model="m")
    other = controller.start_execution(model="m")
    first = await _attempt(leader)
    second = await _attempt(other)
    leader_error = _status_error(503)
    await _fail(first, leader_error)
    error = _status_error(502)
    episode = controller._episodes["m"]
    episode.terminal_until = None
    episode.leader = leader

    decision, rows = await _fail(second, error)

    assert not decision.retry_allowed
    assert decision.error is error
    assert [row["event"] for row in rows] == ["provider.retry.exhausted"]
    assert rows[0]["episode_exhausted"] is False
    assert controller._episodes["m"] is episode
    assert episode.leader is leader
    assert episode.terminal_until is None
    assert episode.last_error is leader_error


@pytest.mark.asyncio
async def test_failure_after_the_episode_closed_takes_the_episode_failure(
    clock: SimpleNamespace,
):
    controller = _controller(max_attempts=2)
    leader = controller.start_execution(model="m")
    stale = controller.start_execution(model="m")
    first = await _attempt(leader)
    late = await _attempt(stale)
    error = _status_error(503)
    await _fail(first, error)
    clock.value = 102.0
    await _fail(await _attempt(leader), error)

    decision, rows = await _fail(late, _status_error(502, retry_after="1"))

    assert not decision.retry_allowed
    assert rows == []
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.last_error is error


@pytest.mark.asyncio
async def test_long_retry_after_starts_a_reported_cooldown(clock: SimpleNamespace):
    controller = _controller()
    execution = controller.start_execution(model="m", request_id="req")
    error = _status_error(429, retry_after="3600")

    decision, rows = await _fail(await _attempt(execution), error)

    assert not decision.retry_allowed
    assert [row["event"] for row in rows] == [
        "provider.recovery.opened",
        "provider.recovery.cooldown",
    ]
    assert rows[0]["request_id"] == "req"
    assert rows[1]["delay_s"] == 3600.0
    episode = controller._episodes["m"]
    assert episode.terminal_until == 3700.0
    assert episode.reports_cooldown
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.retry_not_before == 3700.0


@pytest.mark.asyncio
async def test_retry_after_at_the_ceiling_still_backs_off(clock: SimpleNamespace):
    controller = _controller()
    execution = controller.start_execution(model="m")

    decision, rows = await _fail(
        await _attempt(execution), _status_error(429, retry_after="60")
    )

    assert decision.retry_allowed
    assert rows[-1]["event"] == "provider.retry.scheduled"
    assert rows[-1]["delay_s"] == 60.0


@pytest.mark.asyncio
@pytest.mark.parametrize("leader_in_flight", [False, True])
async def test_cooldown_over_an_open_episode_fails_only_an_idle_leader(
    clock: SimpleNamespace, leader_in_flight: bool
):
    controller = _controller()
    leader = controller.start_execution(model="m")
    other = controller.start_execution(model="m")
    first = await _attempt(leader)
    source = await _attempt(other)
    await _fail(first, _status_error(503))
    generation = controller._episodes["m"].generation
    probe = None
    if leader_in_flight:
        clock.value = 102.0
        probe = await _attempt(leader)
    error = _status_error(429, retry_after="600")

    decision, rows = await _fail(source, error)

    assert [row["event"] for row in rows] == ["provider.recovery.cooldown"]
    episode = controller._episodes["m"]
    assert episode.generation == generation
    assert episode.terminal_until == clock.value + 600
    assert episode.reports_cooldown
    assert episode.leader is None
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.retry_not_before == clock.value + 600
    if probe is not None:
        assert leader.state is ProviderExecutionState.ACTIVE
        await probe.accept()
        await probe.aclose()
    else:
        assert isinstance(leader.last_failure, ProviderRecoveryExhausted)
        assert leader.last_failure.last_error is error


@pytest.mark.asyncio
async def test_longer_cooldown_extends_an_earlier_one(clock: SimpleNamespace):
    controller = _controller()
    first = await _attempt(controller.start_execution(model="m"))
    second = await _attempt(controller.start_execution(model="m"))
    await _fail(first, _status_error(429, retry_after="120"))
    error = _status_error(429, retry_after="600")

    decision, rows = await _fail(second, error)

    assert [row["event"] for row in rows] == ["provider.recovery.cooldown"]
    episode = controller._episodes["m"]
    assert episode.terminal_until == 700.0
    assert episode.last_error is error
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.retry_not_before == 700.0


@pytest.mark.asyncio
async def test_shorter_cooldown_takes_the_covering_one(clock: SimpleNamespace):
    controller = _controller()
    first = await _attempt(controller.start_execution(model="m"))
    second = await _attempt(controller.start_execution(model="m"))
    error = _status_error(429, retry_after="600")
    await _fail(first, error)

    decision, rows = await _fail(second, _status_error(429, retry_after="120"))

    assert rows == []
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.last_error is error
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.retry_not_before == 700.0


@pytest.mark.asyncio
async def test_cooldown_inside_an_exhausted_window_reports_that_window(
    clock: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    controller = ProviderAdmissionController(
        provider_name="TEST",
        max_attempts=1,
        base_delay=60.0,
        max_delay=60.0,
        jitter=10.0,
    )

    def replace_uniform(low: object, high: float):
        return high

    monkeypatch.setattr(admission.random, "uniform", replace_uniform)
    first = await _attempt(controller.start_execution(model="m"))
    second = await _attempt(controller.start_execution(model="m"))
    exhausted_error = _status_error(503)
    await _fail(first, exhausted_error)
    assert controller._episodes["m"].terminal_until == 170.0

    decision, rows = await _fail(second, _status_error(429, retry_after="65"))

    assert [row["event"] for row in rows] == ["provider.recovery.cooldown"]
    assert rows[0]["delay_s"] == 70.0
    episode = controller._episodes["m"]
    assert episode.terminal_until == 170.0
    assert episode.reports_cooldown
    assert episode.last_error is exhausted_error
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.last_error is exhausted_error
    assert isinstance(decision.error, ProviderRecoveryExhausted)
    assert decision.error.retry_not_before == 170.0


@pytest.mark.asyncio
async def test_transitions_finish_without_yielding(clock: SimpleNamespace):
    controller = _controller()
    execution = controller.start_execution(model="m")
    attempt = await _attempt(execution)
    task = asyncio.ensure_future(attempt.fail(_status_error(503)))
    await asyncio.sleep(0)
    assert task.done()
    await attempt.aclose()
