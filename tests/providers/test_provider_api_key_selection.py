"""A generation owns distinct, coalesced clients for numbered credentials."""

import asyncio

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.providers.runtime.config import (
    build_custom_provider_config,
    build_provider_config,
)
from tests.api.support import ApiTestRuntime

KEYS = ["synthetic-first", "synthetic-second", "synthetic-third"]


@pytest.mark.parametrize("provider", ["groq", "gemini", "anthropic", "openai_api"])
def test_builtin_provider_configuration_selects_the_numbered_key(provider):
    settings = Settings(
        GROQ_API_KEY=KEYS,
        GEMINI_API_KEY=KEYS,
        ANTHROPIC_API_KEY=KEYS,
        OPENAI_API_KEY=KEYS,
    )
    config = build_provider_config(
        PROVIDER_CATALOG[provider], settings, api_key_number=3
    )
    assert config.api_key == "synthetic-third"


def test_custom_provider_configuration_selects_the_numbered_key():
    definition = CustomProviderDefinition(
        provider_id="custom_" + "b" * 32,
        display_name="Gateway",
        base_url="https://gateway.example/v1",
        api_keys=KEYS,
    )
    config = build_custom_provider_config(definition, Settings(), api_key_number=2)
    assert config.api_key == "synthetic-second"


def test_missing_selected_key_never_falls_back_to_first():
    settings = Settings(GROQ_API_KEY=KEYS[:1])
    with pytest.raises(ApplicationUnavailableError, match="Key 2"):
        build_provider_config(PROVIDER_CATALOG["groq"], settings, api_key_number=2)


@pytest.mark.asyncio
async def test_concurrent_resolutions_coalesce_by_provider_and_key():
    settings = Settings(GROQ_API_KEY=KEYS)
    runtime = ProviderRuntime(
        settings,
        ProviderAdmissionRegistry(ProviderAdmissionLimits.from_settings(settings)),
    )
    try:
        first, second, second_again, third = await asyncio.gather(
            runtime.resolve_provider("groq", 1),
            runtime.resolve_provider("groq", 2),
            runtime.resolve_provider("groq", 2),
            runtime.resolve_provider("groq", 3),
        )
        assert isinstance(first, GroqProvider)
        assert isinstance(second, GroqProvider)
        assert isinstance(third, GroqProvider)
        assert second is second_again
        assert await runtime.resolve_provider("groq") is first
        assert first is not second and second is not third
        assert [provider._config.api_key for provider in (first, second, third)] == KEYS
        assert runtime.is_cached("groq") and runtime.is_cached("groq", 2)
        assert first._admission is not second._admission
        assert second._admission is not third._admission
    finally:
        await runtime.cleanup()
    assert not runtime.is_cached("groq")
    assert not runtime.is_cached("groq", 2)
    assert not runtime.is_cached("groq", 3)
    assert all(provider._client.is_closed() for provider in (first, second, third))


@pytest.mark.asyncio
async def test_retained_generation_keeps_its_selected_key_after_a_key_edit():
    old_settings = Settings(MODEL="groq/one", GROQ_API_KEY=KEYS)
    new_settings = Settings(
        MODEL="groq/one", GROQ_API_KEY=["synthetic-first", "replacement-second"]
    )
    manager = ApiTestRuntime(old_settings)
    old_lease = await manager.acquire()
    new_lease = None
    try:
        old_client = await old_lease.resolve_provider("groq", 2)
        assert isinstance(old_client, GroqProvider)

        async def commit():
            return None

        await manager.replace(new_settings, commit=commit)
        new_lease = await manager.acquire()
        new_client = await new_lease.resolve_provider("groq", 2)
        assert isinstance(new_client, GroqProvider)
        assert old_client._config.api_key == "synthetic-second"
        assert new_client._config.api_key == "replacement-second"
        assert not old_client._client.is_closed()
        assert await old_lease.resolve_provider("groq", 2) is old_client
        await old_lease.release()
        assert old_client._client.is_closed()
        assert not new_client._client.is_closed()
    finally:
        await old_lease.release()
        if new_lease is not None:
            await new_lease.release()
        await manager.close()
