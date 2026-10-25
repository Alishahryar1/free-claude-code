"""Recovery owns live coordination, not an unbounded history of requests."""

import asyncio
import gc
import weakref
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from free_claude_code.providers import admission
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.failure_policy import ProviderRecoveryExhausted
from free_claude_code.providers.http import ProviderAttemptScope
from tests.providers.test_provider_admission import _controller, _open, _status_error


class Payload:
    pass


async def exhaust(controller, model):
    payload = Payload()

    async def request():
        assert payload is not None
        raise TimeoutError("upstream unavailable")

    with pytest.raises(TimeoutError):
        await controller.start_execution(model=model).run_call(
            request, operation_kind=ProviderOperationKind.GENERATION
        )
    return weakref.ref(payload)


@pytest.mark.asyncio
async def test_expired_requests_are_collected_without_more_traffic():
    controller = _controller(max_attempts=1, base_delay=0.03, max_delay=0.03)
    refs = [await exhaust(controller, str(i)) for i in range(8)]
    gc.collect()
    assert any(ref() is not None for ref in refs)
    await asyncio.sleep(0.06)
    gc.collect()
    assert not controller._episodes
    assert all(ref() is None for ref in refs)


@pytest.mark.asyncio
async def test_idle_abandonment_preserves_backoff_then_expires():
    controller = _controller(base_delay=0.03, max_delay=0.03)
    execution = controller.start_execution(model="a")
    attempt = await _open(execution)
    await attempt.fail(_status_error(503))
    await attempt.aclose()
    execution.abandon()
    assert "a" in controller._episodes
    await asyncio.sleep(0.06)
    assert not controller._episodes


@pytest.mark.asyncio
async def test_cancel_between_resolved_attempts_releases_leader():
    controller = _controller()
    entered = asyncio.Event()

    class ClosingResource:
        async def aclose(self):
            entered.set()
            await asyncio.Event().wait()

    async def request():
        execution = controller.start_execution(model="a")
        scope = ProviderAttemptScope(
            await _open(execution), provider_name="TEST", request_id=None
        )
        scope.retain(ClosingResource())
        try:
            await scope.attempt.fail(_status_error(503))
        finally:
            try:
                await scope.aclose(active_error=None)
            finally:
                execution.abandon()

    task = asyncio.create_task(request())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    next_attempt = await asyncio.wait_for(
        _open(controller.start_execution(model="a")), 1
    )
    await next_attempt.accept()
    await next_attempt.aclose()
    assert controller._capacity._active_attempts == 0
    assert not controller._episodes


@pytest.mark.asyncio
async def test_terminal_waiter_detaches_before_opening_next_attempt():
    controller = _controller()
    leader = controller.start_execution(model="a")
    follower = controller.start_execution(model="a")
    first = await _open(leader)
    second = await _open(follower)
    for attempt in (first, second):
        await attempt.fail(_status_error(503))
        await attempt.aclose()
    follower.abandon()
    assert follower not in controller._episodes["a"].waiters
    leader.abandon()
    assert not controller._episodes


@pytest.mark.asyncio
async def test_timer_does_not_capture_request_context():
    controller = _controller(max_attempts=1, base_delay=60, max_delay=60)
    context = ContextVar("request_payload")
    payload = Payload()
    ref = weakref.ref(payload)
    token = context.set(payload)
    await exhaust(controller, "a")
    context.reset(token)
    del payload
    # Retain a cancelled handle: asyncio may leave it in its timer heap.
    handle = controller._expiry_timer
    controller.close()
    gc.collect()
    assert handle is not None and handle.cancelled()
    assert ref() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True])
async def test_registry_retirement_releases_scheduled_controller(custom):
    registry = ProviderAdmissionRegistry(ProviderAdmissionLimits(1000, 60, 5))
    controller = registry.get("custom_test" if custom else "groq")
    # Exhaust without waiting through retries.
    controller._max_attempts = 1
    await exhaust(controller, "a")
    handle = controller._expiry_timer
    ref = weakref.ref(controller)
    if custom:
        registry.retain_custom(())
    else:
        registry.close()
    del controller
    gc.collect()
    assert handle is not None and handle.cancelled()
    assert ref() is None


@pytest.mark.asyncio
async def test_expiry_allows_concurrent_fresh_attempts():
    controller = _controller(max_attempts=1, base_delay=0.03, max_delay=0.03)
    await exhaust(controller, "a")
    await asyncio.sleep(0.06)
    first, second = await asyncio.wait_for(
        asyncio.gather(
            _open(controller.start_execution(model="a")),
            _open(controller.start_execution(model="a")),
        ),
        1,
    )
    await first.accept()
    await second.accept()
    await first.aclose()
    await second.aclose()


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


@pytest.mark.asyncio
async def test_live_members_and_reserved_probes_survive_deadlines(clock):
    controller = _controller(base_delay=1, max_delay=1)
    leader = controller.start_execution(model="a")
    follower = controller.start_execution(model="a")
    attempts = [await _open(execution) for execution in (leader, follower)]
    for attempt in attempts:
        await attempt.fail(_status_error(503))
        await attempt.aclose()
    episode = controller._episodes["a"]
    clock[0] += 10
    controller._expire_idle_episodes()
    assert controller._episodes["a"] is episode
    assert controller._expiry_timer is None

    # Reserve the follower's probe while another model occupies capacity.
    controller.reconfigure(ProviderAdmissionLimits(1000, 60, 1))
    blocker = await _open(controller.start_execution(model="b"))
    leader.abandon()
    queued = asyncio.create_task(_open(follower))
    await asyncio.sleep(0)
    assert episode.probe_active
    clock[0] += 10
    controller._expire_idle_episodes()
    assert controller._episodes["a"] is episode
    assert not queued.done()
    await blocker.aclose()
    probe = await asyncio.wait_for(queued, 1)
    follower.abandon()
    assert controller._episodes["a"] is episode
    assert controller._capacity._active_attempts == 1
    await probe.aclose()
    await probe.aclose()
    assert not controller._episodes
    assert controller._capacity._active_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", [False, True])
async def test_terminal_execution_cannot_open_queued_attempt(probe):
    controller = _controller(max_concurrency=1)
    execution = controller.start_execution(model="a")
    if probe:
        attempt = await _open(execution)
        await attempt.fail(_status_error(503))
        await attempt.aclose()
    blocker = await _open(controller.start_execution(model="b"))
    queued = asyncio.create_task(_open(execution))
    await asyncio.sleep(0)
    execution.abandon()
    with pytest.raises(RuntimeError, match="terminal"):
        await asyncio.wait_for(queued, 1)
    await blocker.aclose()
    assert not controller._episodes
    assert execution.attempts_started == int(probe)
    assert controller._capacity._active_attempts == 0


@pytest.mark.asyncio
async def test_exhausted_waiters_keep_their_failure_after_expiry(clock):
    controller = _controller(max_attempts=2, base_delay=1, max_delay=1)
    leader = controller.start_execution(model="a")
    followers = [controller.start_execution(model="a") for _ in range(3)]
    attempts = [await _open(execution) for execution in [leader, *followers]]
    for attempt in attempts:
        await attempt.fail(_status_error(503))
        await attempt.aclose()
    waiting = [asyncio.create_task(_open(execution)) for execution in followers]
    await asyncio.sleep(0)
    clock[0] += 1
    probe = await _open(leader)
    error = _status_error(429)
    await probe.fail(error)
    await probe.aclose()
    clock[0] += 1
    controller._expire_idle_episodes()
    assert not controller._episodes
    outcomes = await asyncio.wait_for(
        asyncio.gather(*waiting, return_exceptions=True), 1
    )
    for outcome, execution in zip(outcomes, followers, strict=True):
        assert isinstance(outcome, ProviderRecoveryExhausted)
        assert outcome.last_error is error
        assert execution.last_failure is error
        assert execution.attempts_started == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_late_normal_failure_obeys_current_cooldown(clock, expired):
    controller = _controller(max_attempts=2, base_delay=1, max_delay=1)
    leader = controller.start_execution(model="a")
    late = controller.start_execution(model="a")
    first, normal = await _open(leader), await _open(late)
    await first.fail(_status_error(503))
    await first.aclose()
    clock[0] += 1
    probe = await _open(leader)
    old_error = _status_error(429)
    await probe.fail(old_error)
    await probe.aclose()
    generation = controller._episodes["a"].generation
    if expired:
        clock[0] += 1  # No timer callback: failure lookup must prune due state.
    new_error = _status_error(503)
    decision = await normal.fail(new_error)
    await normal.aclose()
    assert decision.retry_allowed is expired
    assert late.last_failure is (new_error if expired else old_error)
    episode = controller._episodes["a"]
    assert (episode.generation > generation) is expired
    late.abandon()
    controller.close()


@pytest.mark.asyncio
async def test_one_timer_tracks_multiple_deadlines_and_current_state(clock):
    controller = _controller(max_attempts=1, base_delay=1, max_delay=1)
    await exhaust(controller, "a")
    first_timer = controller._expiry_timer
    clock[0] += 0.5
    await exhaust(controller, "b")
    assert controller._expiry_timer is first_timer
    clock[0] += 0.5
    controller._expire_idle_episodes()
    assert set(controller._episodes) == {"b"}
    second_timer = controller._expiry_timer
    assert second_timer is not None and second_timer is not first_timer
    assert first_timer is not None and first_timer.cancelled()

    # A stale maintenance callback rechecks the new generation's deadline.
    await exhaust(controller, "a")
    controller._expire_idle_episodes()
    assert set(controller._episodes) == {"a", "b"}
    assert second_timer.cancelled()
    clock[0] += 0.5
    controller._expire_idle_episodes()
    assert set(controller._episodes) == {"a"}
    clock[0] += 0.5
    controller._expire_idle_episodes()
    assert not controller._episodes
    assert controller._expiry_timer is None
    controller.close()
    controller.close()
    controller._expire_idle_episodes()
    assert controller._expiry_timer is None


@pytest.mark.asyncio
async def test_zero_delay_expiry_preserves_occupied_capacity_and_rate_history():
    controller = _controller(max_attempts=1, max_concurrency=1)
    attempt = await _open(controller.start_execution(model="a"))
    history = tuple(controller._capacity._admission_times)
    await attempt.fail(_status_error(503))
    assert not controller._episodes
    assert controller._expiry_timer is None
    assert controller._capacity._active_attempts == 1
    assert tuple(controller._capacity._admission_times) == history
    queued = asyncio.create_task(_open(controller.start_execution(model="a")))
    await asyncio.sleep(0)
    assert not queued.done()
    await attempt.aclose()
    fresh = await asyncio.wait_for(queued, 1)
    await fresh.accept()
    await fresh.aclose()
    assert controller._capacity._active_attempts == 0
    assert len(controller._capacity._admission_times) == 2


@pytest.mark.asyncio
async def test_old_backoff_cancellation_cannot_detach_replacement_leader(clock):
    controller = _controller(base_delay=1, max_delay=1)
    leader = controller.start_execution(model="a")
    follower = controller.start_execution(model="a")
    attempts = [await _open(execution) for execution in (leader, follower)]
    for attempt in attempts:
        await attempt.fail(_status_error(503))
        await attempt.aclose()
    sleeping = asyncio.create_task(_open(leader))
    await asyncio.sleep(0)
    leader.abandon()
    clock[0] += 1
    replacement = await _open(follower)
    sleeping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sleeping
    assert controller._episodes["a"].leader is follower
    assert controller._episodes["a"].probe_active
    await replacement.accept()
    await replacement.aclose()
    assert not controller._episodes
