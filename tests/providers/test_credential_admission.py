"""Selected credentials isolate recovery while sharing provider capacity."""

import asyncio

import httpx
import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.runtime import ProviderRuntime


@pytest.mark.asyncio
@pytest.mark.parametrize("protection", ["rate", "concurrency"])
async def test_different_credentials_share_capacity_and_limit_updates(protection):
    settings = Settings(
        GROQ_API_KEY=["synthetic-first", "synthetic-second"],
        PROVIDER_RATE_LIMIT=1 if protection == "rate" else 1000,
        PROVIDER_RATE_WINDOW=60,
        PROVIDER_MAX_CONCURRENCY=1,
    )
    registry = ProviderAdmissionRegistry(
        ProviderAdmissionLimits.from_settings(settings)
    )
    runtime = ProviderRuntime(settings, registry)
    pending = None
    first_attempt = second_attempt = None
    try:
        first = await runtime.resolve_provider("groq", 1)
        second = await runtime.resolve_provider("groq", 2)
        assert isinstance(first, GroqProvider)
        assert isinstance(second, GroqProvider)
        first_attempt = await first._admission.start_execution().open_attempt(
            ProviderOperationKind.GENERATION
        )
        await first_attempt.accept()
        if protection == "rate":
            await first_attempt.aclose()
        pending = asyncio.create_task(
            second._admission.start_execution().open_attempt(
                ProviderOperationKind.GENERATION
            )
        )
        await asyncio.sleep(0)
        assert not pending.done(), f"another credential bypassed provider {protection}"
        registry.reconfigure(ProviderAdmissionLimits(1000, 60, 2))
        second_attempt = await asyncio.wait_for(pending, 1)
        await second_attempt.accept()
    finally:
        if pending is not None:
            pending.cancel()
            outcomes = await asyncio.gather(pending, return_exceptions=True)
            if not isinstance(outcomes[0], BaseException):
                await outcomes[0].aclose()
        if second_attempt is not None:
            await second_attempt.aclose()
        if first_attempt is not None:
            await first_attempt.aclose()
        await runtime.cleanup()
        registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_first", ["synthetic-second", "replacement-first"])
async def test_recovery_follows_credentials_across_key_reorder_and_replacement(
    new_first,
):
    old_settings = Settings(
        GROQ_API_KEY=["synthetic-first", "synthetic-second"],
        PROVIDER_RATE_LIMIT=1000,
    )
    new_settings = Settings(
        GROQ_API_KEY=[new_first, "synthetic-first"], PROVIDER_RATE_LIMIT=1000
    )
    registry = ProviderAdmissionRegistry(
        ProviderAdmissionLimits.from_settings(old_settings)
    )
    old_runtime = ProviderRuntime(old_settings, registry)
    new_runtime = ProviderRuntime(new_settings, registry)
    pending = None
    old_attempt = healthy_attempt = None
    try:
        old_first = await old_runtime.resolve_provider("groq", 1)
        healthy = await new_runtime.resolve_provider("groq", 1)
        same_credential = await new_runtime.resolve_provider("groq", 2)
        assert isinstance(old_first, GroqProvider)
        assert isinstance(healthy, GroqProvider)
        assert isinstance(same_credential, GroqProvider)
        old_attempt = await old_first._admission.start_execution().open_attempt(
            ProviderOperationKind.GENERATION
        )
        response = httpx.Response(
            503,
            headers={"retry-after": "60"},
            request=httpx.Request("POST", "https://provider.example"),
        )
        await old_attempt.fail(
            httpx.HTTPStatusError(
                "unavailable", request=response.request, response=response
            )
        )
        await old_attempt.aclose()
        pending = asyncio.create_task(
            same_credential._admission.start_execution().open_attempt(
                ProviderOperationKind.GENERATION
            )
        )
        await asyncio.sleep(0)
        assert not pending.done(), "moving a credential bypassed its recovery"
        healthy_attempt = await asyncio.wait_for(
            healthy._admission.start_execution().open_attempt(
                ProviderOperationKind.GENERATION
            ),
            1,
        )
        await healthy_attempt.accept()
    finally:
        if pending is not None:
            pending.cancel()
            outcomes = await asyncio.gather(pending, return_exceptions=True)
            if not isinstance(outcomes[0], BaseException):
                await outcomes[0].aclose()
        if healthy_attempt is not None:
            await healthy_attempt.aclose()
        if old_attempt is not None:
            await old_attempt.aclose()
        await old_runtime.cleanup()
        await new_runtime.cleanup()
        registry.close()
