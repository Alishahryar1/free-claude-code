"""Models share provider capacity without sharing recovery failures."""

import asyncio

import pytest

from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.failure_policy import ProviderRecoveryExhausted
from tests.providers.test_provider_admission import _controller, _open, _status_error


async def admit(execution):
    return await execution.open_attempt(
        ProviderOperationKind.MODEL_DISCOVERY
        if execution.model is None
        else ProviderOperationKind.GENERATION
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.parametrize(
    "failed_model,healthy_model", [("a", "b"), ("a", None), (None, "a")]
)
async def test_recovery_is_scoped_to_model(exhausted, failed_model, healthy_model):
    controller = _controller(max_attempts=1 if exhausted else 2)
    failed = await admit(controller.start_execution(model=failed_model))
    await failed.fail(_status_error(503, retry_after="60"))
    await failed.aclose()

    healthy = await asyncio.wait_for(
        admit(controller.start_execution(model=healthy_model)), 1
    )
    await healthy.accept()
    await healthy.aclose()

    same_model = controller.start_execution(model=failed_model)
    if exhausted:
        with pytest.raises(ProviderRecoveryExhausted):
            await admit(same_model)
        assert same_model.attempts_started == 0
    else:
        pending = asyncio.create_task(admit(same_model))
        await asyncio.sleep(0)
        assert not pending.done()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "reject", "exhaust", "cancel"])
async def test_parallel_model_probes_do_not_resolve_each_others_waiters(outcome):
    controller = _controller(max_attempts=2)
    first = controller.start_execution(model="a")
    second = controller.start_execution(model="b")
    for execution in (first, second):
        attempt = await _open(execution)
        await attempt.fail(_status_error(503))
        await attempt.aclose()

    waiting = asyncio.create_task(_open(controller.start_execution(model="b")))
    await asyncio.sleep(0)
    assert not waiting.done()
    probe = await _open(first)
    try:
        if outcome == "success":
            await probe.accept()
        elif outcome == "reject":
            await probe.fail(_status_error(400))
        elif outcome == "exhaust":
            await probe.fail(_status_error(503, retry_after="60"))
        # Closing an unresolved probe releases its leader as cancellation does.
        await probe.aclose()
        await asyncio.sleep(0)
        assert not waiting.done()

        second_probe = await _open(second)
        await second_probe.accept()
        await second_probe.aclose()
        follower = await asyncio.wait_for(waiting, 1)
        await follower.accept()
        await follower.aclose()
    finally:
        await probe.aclose()
        waiting.cancel()
        results = await asyncio.gather(waiting, return_exceptions=True)
        if not isinstance(results[0], BaseException):
            await results[0].aclose()


@pytest.mark.asyncio
async def test_another_models_recovery_does_not_invalidate_queued_capacity_permit():
    controller = _controller(max_concurrency=1)
    first = await _open(controller.start_execution(model="a"))
    queued = asyncio.create_task(_open(controller.start_execution(model="b")))
    try:
        await asyncio.sleep(0)
        assert not queued.done()
        await first.fail(_status_error(503, retry_after="60"))
        await first.aclose()
        second = await asyncio.wait_for(queued, 1)
        await second.accept()
        await second.aclose()
    finally:
        await first.aclose()
        queued.cancel()
        results = await asyncio.gather(queued, return_exceptions=True)
        if not isinstance(results[0], BaseException):
            await results[0].aclose()
