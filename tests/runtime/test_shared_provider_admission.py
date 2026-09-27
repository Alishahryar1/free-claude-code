"""Admission protections survive replacement of their provider clients."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.test_provider_manager import RuntimeFactory


@pytest.mark.asyncio
@pytest.mark.parametrize("protection", ["concurrency", "rate", "recovery"])
async def test_replacement_cannot_bypass_existing_protection(protection: str) -> None:
    settings = Settings(
        nvidia_nim_api_key="test-key",
        provider_rate_limit=1 if protection == "rate" else 1000,
        provider_rate_window=60,
        provider_max_concurrency=1,
    )
    manager = ProviderRuntimeManager(settings)
    old_lease = await manager.acquire()
    pending = None
    old_attempt = None
    try:
        old = cast(
            NvidiaNimProvider,
            await manager._current.runtime.resolve_provider("nvidia_nim"),
        )
        old_attempt = await old._admission.start_execution().open_attempt(
            ProviderOperationKind.GENERATION
        )
        if protection == "recovery":
            response = httpx.Response(
                503,
                headers={"retry-after": "60"},
                request=httpx.Request("POST", "https://provider.test"),
            )
            await old_attempt.fail(
                httpx.HTTPStatusError(
                    "unavailable", request=response.request, response=response
                )
            )
        else:
            await old_attempt.accept()
        if protection != "concurrency":
            await old_attempt.aclose()

        with patch.object(
            NvidiaNimProvider, "list_model_infos", AsyncMock(return_value=frozenset())
        ):
            await manager.replace(
                settings.model_copy(update={"log_raw_sse_events": True}),
                commit=AsyncMock(),
            )
            new = cast(
                NvidiaNimProvider,
                await manager._current.runtime.resolve_provider("nvidia_nim"),
            )
            pending = asyncio.create_task(
                new._admission.start_execution().open_attempt(
                    ProviderOperationKind.GENERATION
                )
            )
            await asyncio.sleep(0)
            assert not pending.done(), f"replacement bypassed {protection}"
    finally:
        if pending is not None:
            pending.cancel()
            outcomes = await asyncio.gather(pending, return_exceptions=True)
            if not isinstance(outcomes[0], BaseException):
                await outcomes[0].aclose()
        if old_attempt is not None:
            await old_attempt.aclose()
        await old_lease.release()
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_limits_publish_only_with_successful_settings(outcome):
    settings = Settings(provider_rate_limit=1000, provider_max_concurrency=1)
    manager = ProviderRuntimeManager(settings, runtime_factory=RuntimeFactory())
    controller = manager._admission_registry.get("nvidia_nim")
    first = await controller.start_execution().open_attempt(
        ProviderOperationKind.GENERATION
    )
    pending = asyncio.create_task(
        controller.start_execution().open_attempt(ProviderOperationKind.GENERATION)
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def commit():
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise OSError("cannot save")

    replacement = asyncio.create_task(
        manager.replace(
            settings.model_copy(update={"provider_max_concurrency": 2}), commit=commit
        )
    )
    await entered.wait()
    assert not pending.done()
    if outcome == "cancel":
        replacement.cancel()
    else:
        release.set()
    results = await asyncio.gather(replacement, return_exceptions=True)
    if outcome == "success":
        assert results == [2]
        await (await asyncio.wait_for(pending, 1)).aclose()
    else:
        assert isinstance(results[0], BaseException)
        await asyncio.sleep(0)
        assert not pending.done()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    await first.aclose()
    await manager.close()


@pytest.mark.asyncio
async def test_blocked_publication_keeps_old_limits_until_generation_switch():
    settings = Settings(provider_rate_limit=1000, provider_max_concurrency=1)
    manager = ProviderRuntimeManager(settings, runtime_factory=RuntimeFactory())
    controller = manager._admission_registry.get("nvidia_nim")
    first = await controller.start_execution().open_attempt(
        ProviderOperationKind.GENERATION
    )
    pending = asyncio.create_task(
        controller.start_execution().open_attempt(ProviderOperationKind.GENERATION)
    )
    committed = asyncio.Event()

    async def commit():
        committed.set()

    await manager._publication_lock.acquire()
    replacement = asyncio.create_task(
        manager.replace(
            settings.model_copy(update={"provider_max_concurrency": 2}), commit=commit
        )
    )
    await committed.wait()
    assert not pending.done()
    assert manager.current_generation_id == 1
    manager._publication_lock.release()
    await replacement
    await (await asyncio.wait_for(pending, 1)).aclose()
    await first.aclose()
    await manager.close()


@pytest.mark.asyncio
async def test_old_lazy_construction_uses_current_limits_and_survives_retirement():
    settings = Settings(
        nvidia_nim_api_key="test", provider_rate_limit=1000, provider_max_concurrency=1
    )
    manager = ProviderRuntimeManager(settings)
    old_lease = await manager.acquire()
    old_runtime = manager._current.runtime
    with patch.object(
        NvidiaNimProvider, "list_model_infos", AsyncMock(return_value=frozenset())
    ):
        await manager.replace(
            settings.model_copy(update={"provider_max_concurrency": 2}),
            commit=AsyncMock(),
        )
        old = cast(NvidiaNimProvider, await old_runtime.resolve_provider("nvidia_nim"))
        new = cast(
            NvidiaNimProvider,
            await manager._current.runtime.resolve_provider("nvidia_nim"),
        )
        first = await old._admission.start_execution().open_attempt(
            ProviderOperationKind.GENERATION
        )
        second = await asyncio.wait_for(
            new._admission.start_execution().open_attempt(
                ProviderOperationKind.GENERATION
            ),
            1,
        )
        blocked = asyncio.create_task(
            new._admission.start_execution().open_attempt(
                ProviderOperationKind.GENERATION
            )
        )
        await asyncio.sleep(0)
        assert not blocked.done()
        blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)
        await first.aclose()
        await second.aclose()
        await old_lease.release()
        await asyncio.wait_for(
            new._admission.start_execution().run_call(
                AsyncMock(return_value="ok"),
                operation_kind=ProviderOperationKind.GENERATION,
            ),
            1,
        )
    await manager.close()


@pytest.mark.asyncio
async def test_separate_provider_ids_and_managers_have_independent_capacity():
    limits = ProviderAdmissionLimits(1, 60, 1)
    registry = ProviderAdmissionRegistry(limits)
    other = ProviderAdmissionRegistry(limits)
    attempts = []
    for owner, provider in [(registry, "one"), (registry, "two"), (other, "one")]:
        attempts.append(
            await asyncio.wait_for(
                owner.get(provider)
                .start_execution()
                .open_attempt(ProviderOperationKind.GENERATION),
                1,
            )
        )
    for attempt in attempts:
        await attempt.aclose()
    registry.close()
    other.close()


@pytest.mark.asyncio
async def test_custom_registry_entry_lives_until_retired_generation_cleanup_succeeds():
    from free_claude_code.config.custom_providers import CustomProviderDefinition

    custom_id = "custom_12345678123412341234123456789abc"
    settings = Settings(
        custom_providers=(
            CustomProviderDefinition(
                provider_id=custom_id,
                display_name="Example",
                base_url="https://example.test/v1",
                api_format="openai_chat",
            ),
        )
    )
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(settings, runtime_factory=factory)
    lease = await manager.acquire()
    controller = manager._admission_registry.get(custom_id)
    await manager.replace(
        settings.model_copy(update={"custom_providers": ()}), commit=AsyncMock()
    )
    assert manager._admission_registry.get(custom_id) is controller
    factory.runtimes[0].cleanup_error = OSError("not closed yet")
    await lease.release()
    assert manager._admission_registry.get(custom_id) is controller
    factory.runtimes[0].cleanup_error = None
    await manager._close_generation(manager._retired[1], forced=False)
    assert custom_id not in manager._admission_registry._controllers
    await manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager._admission_registry.get("nvidia_nim")
