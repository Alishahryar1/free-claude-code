"""Live policy changes preserve capacity and wake the correct waiting calls."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from free_claude_code.config.constants import PROVIDER_MAX_ATTEMPTS_DEFAULT
from free_claude_code.config.settings import Settings
from free_claude_code.providers import admission_capacity as capacity_module
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry


async def admit(controller: ProviderAdmissionController):
    return await controller.start_execution().open_attempt(
        ProviderOperationKind.GENERATION
    )


async def completed(controller: ProviderAdmissionController):
    attempt = await admit(controller)
    await attempt.accept()
    await attempt.aclose()


@pytest.mark.asyncio
async def test_limit_changes_preserve_occupied_slots_and_wake_waiters():
    controller = ProviderAdmissionController(
        provider_name="test", rate_limit=1000, max_concurrency=1
    )
    first = await admit(controller)
    pending = asyncio.create_task(admit(controller))
    await asyncio.sleep(0)
    assert not pending.done()
    controller.reconfigure(
        ProviderAdmissionLimits(1000, 60, 2, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    second = await asyncio.wait_for(pending, 1)
    controller.reconfigure(
        ProviderAdmissionLimits(1000, 60, 1, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    third = asyncio.create_task(admit(controller))
    await first.aclose()
    await asyncio.sleep(0)
    assert not third.done()
    await second.aclose()
    await (await asyncio.wait_for(third, 1)).aclose()


@pytest.mark.asyncio
async def test_waiting_for_concurrency_does_not_spend_rate_capacity(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        capacity_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    controller = ProviderAdmissionController(
        provider_name="test", rate_limit=1, rate_window=10, max_concurrency=1
    )
    first = await admit(controller)
    clock.now = 11
    pending = asyncio.create_task(admit(controller))
    await asyncio.sleep(0)
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    await first.aclose()
    await asyncio.wait_for(completed(controller), 1)


@pytest.mark.asyncio
async def test_rate_count_increase_wakes_waiter_without_clearing_history():
    controller = ProviderAdmissionController(
        provider_name="test", rate_limit=1, rate_window=60
    )
    await completed(controller)
    pending = asyncio.create_task(admit(controller))
    await asyncio.sleep(0)
    assert not pending.done()
    controller.reconfigure(
        ProviderAdmissionLimits(2, 60, 5, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    await (await asyncio.wait_for(pending, 1)).aclose()
    blocked = asyncio.create_task(admit(controller))
    await asyncio.sleep(0)
    assert not blocked.done()
    blocked.cancel()
    await asyncio.gather(blocked, return_exceptions=True)


@pytest.mark.asyncio
async def test_longer_window_waits_only_until_discarded_history_expires(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        capacity_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    controller = ProviderAdmissionController(
        provider_name="test", rate_limit=1, rate_window=2
    )
    await completed(controller)
    clock.now = 3
    await completed(controller)  # Drops the timestamp at zero.
    controller.reconfigure(
        ProviderAdmissionLimits(100, 10, 5, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    pending = asyncio.create_task(admit(controller))
    await asyncio.sleep(0)
    assert not pending.done()  # Capacity is ample, but older history is incomplete.
    clock.now = 9
    controller.reconfigure(
        ProviderAdmissionLimits(101, 10, 5, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    await asyncio.sleep(0)
    assert not pending.done()
    clock.now = 10
    controller.reconfigure(
        ProviderAdmissionLimits(102, 10, 5, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    await (await asyncio.wait_for(pending, 1)).aclose()


@pytest.mark.asyncio
async def test_shorter_window_wakes_waiter_and_longer_window_uses_retained_history(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        capacity_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    controller = ProviderAdmissionController(
        provider_name="test", rate_limit=1, rate_window=60
    )
    await completed(controller)
    controller.reconfigure(
        ProviderAdmissionLimits(2, 120, 5, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    await asyncio.wait_for(completed(controller), 1)  # No artificial expansion pause.
    pending = asyncio.create_task(admit(controller))
    await asyncio.sleep(0)
    assert not pending.done()
    clock.now = 2
    controller.reconfigure(
        ProviderAdmissionLimits(1, 2, 5, PROVIDER_MAX_ATTEMPTS_DEFAULT)
    )
    await (await asyncio.wait_for(pending, 1)).aclose()


@pytest.mark.parametrize(
    "field",
    [
        "provider_rate_limit",
        "provider_rate_window",
        "provider_max_concurrency",
        "provider_max_attempts",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_settings_reject_invalid_admission_limits(field: str, value: int):
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})


@pytest.mark.parametrize("window", [float("inf"), float("nan"), 0, -1])
def test_internal_policy_rejects_invalid_window(window: float | int):
    with pytest.raises(ValueError, match="rate_window"):
        ProviderAdmissionLimits(1, window, 1, PROVIDER_MAX_ATTEMPTS_DEFAULT)


@pytest.mark.asyncio
async def test_attempt_budget_follows_settings_and_applies_to_new_executions():
    registry = ProviderAdmissionRegistry(
        ProviderAdmissionLimits.from_settings(Settings(provider_max_attempts=3))
    )
    controller = registry.get("test", api_key="key")
    running = controller.start_execution()

    registry.reconfigure(
        ProviderAdmissionLimits.from_settings(Settings(provider_max_attempts=2))
    )

    assert running.max_attempts == 3
    assert controller.start_execution().max_attempts == 2
    assert registry.get("other").start_execution().max_attempts == 2
    registry.close()
