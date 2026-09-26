import asyncio

import pytest

from free_claude_code.application.model_catalog import read_model_catalog
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.settings import Settings
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.test_provider_manager import FakeRuntime, RuntimeFactory

pytestmark = pytest.mark.asyncio
ID = "custom_" + "a" * 32


def settings(**changes):
    definition = CustomProviderDefinition.model_validate(
        {
            "provider_id": ID,
            "display_name": "Gateway",
            "base_url": "https://example.org/v1",
            **changes,
        }
    )
    return Settings(custom_providers=(definition,))


async def commit():
    pass


async def test_runtime_constructs_one_custom_owner_lazily():
    runtime = ProviderRuntime(settings(model_ids=["manual"]))
    try:
        assert not runtime.is_cached(ID)
        one, two = await asyncio.gather(
            runtime.resolve_provider(ID), runtime.resolve_provider(ID)
        )
        assert one is two and isinstance(one, CustomProvider)
        assert {info.model_id for info in await one.list_model_infos()} == {"manual"}
    finally:
        await runtime.cleanup()


async def test_manual_inventory_is_immediate_and_does_not_construct_client():
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        settings(model_ids=["manual"]), runtime_factory=factory
    )
    try:
        catalog = read_model_catalog(manager)
        assert any(
            item.provider_model_ref == f"{ID}/manual"
            and item.display_name == "Gateway/manual"
            for item in catalog.models
        )
        await manager.refresh_provider(ID)
        factory.runtimes[0].provider.list_model_infos.assert_not_called()
    finally:
        await manager.close()


async def test_changed_endpoint_cannot_inherit_catalog_but_rename_can():
    factory = RuntimeFactory()
    release = asyncio.Event()

    async def wait():
        await release.wait()
        return frozenset()

    def blocked(settings):
        runtime = factory(settings)
        assert isinstance(runtime, FakeRuntime)
        runtime.provider.list_model_infos.side_effect = wait
        return runtime

    manager = ProviderRuntimeManager(settings(), runtime_factory=blocked)
    manager.cache_model_infos(ID, [ProviderModelInfo("old")])
    try:
        await manager.replace(settings(display_name="Renamed"), commit=commit)
        assert manager.cached_model_info(ID, "old") is not None
        lease = await manager.acquire()
        await manager.replace(
            settings(base_url="https://other.example/v2"), commit=commit
        )
        assert manager.cached_model_info(ID, "old") is None
        retained = lease.settings.custom_provider(ID)
        assert retained is not None and retained.base_url == "https://example.org/v1"
        await lease.release()
        assert factory.runtimes[-2].cleanup_calls == 1
    finally:
        release.set()
        await manager.close()
