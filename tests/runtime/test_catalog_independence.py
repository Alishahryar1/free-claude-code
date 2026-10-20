"""Catalog consumers depend on their own work, not the slowest provider."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.code_sessions.models import CodeValidationError
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.readiness import InitializationWait
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import no_thinking_gateway_model_id
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.runtime.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.codex_app_server import CodexHarnessFactory
from free_claude_code.runtime.codex_catalog import CodexModelCatalogPublisher
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.catalog_support import finish_discovery, finish_publication
from tests.web_tools_support import StubWebToolsClient


@pytest_asyncio.fixture
async def catalog_runtime():
    entered, release = asyncio.Event(), asyncio.Event()
    settings = Settings.model_validate(
        {
            "MODEL": "nvidia_nim/configured",
            "MODEL_SONNET": "groq/configured",
            "NVIDIA_NIM_API_KEY": "test",
            "GROQ_API_KEY": "test",
            "MESSAGING_PLATFORM": "none",
            "PROXY_AUTH_ENABLED": False,
        }
    )

    async def construct(provider_id, *_):
        provider = MagicMock(spec=BaseProvider)

        async def models():
            if provider_id == "groq":
                entered.set()
                await release.wait()
            return frozenset(
                {
                    ProviderModelInfo("discovered"),
                    ProviderModelInfo("configured", supports_thinking=False),
                }
            )

        provider.list_model_infos = AsyncMock(side_effect=models)
        return provider

    manager = ProviderRuntimeManager(
        settings,
        runtime_factory=lambda settings, registry: ProviderRuntime(
            settings, registry, provider_constructor=construct
        ),
    )
    runtime = ApplicationRuntime(
        manager,
        configuration=ConfigurationService(ManagedConfigStore()),
        transcriber=None,
    )
    manager.start_model_list_refresh()
    try:
        await asyncio.wait_for(entered.wait(), 1)
        async with await manager.acquire() as lease:
            await lease.resolve_provider("nvidia_nim")
        yield runtime, release
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/models?view=responses", "/muse-code/models"])
async def test_http_catalog_exposes_ready_models_while_another_provider_waits(
    catalog_runtime, path
):
    runtime, release = catalog_runtime
    manager = runtime.provider_manager
    app = create_app(
        ApiServices(manager, runtime, runtime, web_tools=StubWebToolsClient())
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await asyncio.wait_for(client.get(path), 1)
    assert response.status_code == 200
    ids = {entry["id"] for entry in response.json()["data"]}
    assert {
        no_thinking_gateway_model_id("nvidia_nim/configured"),
        "nvidia_nim/discovered",
    } <= ids
    assert not release.is_set()
    assert manager.catalog_status()["providers"]["groq"] == "starting"


@pytest.mark.asyncio
async def test_code_preparation_uses_only_its_selected_provider(catalog_runtime):
    runtime, release = catalog_runtime
    factory = CodexHarnessFactory(runtime.provider_manager, binary="codex")
    selection = await asyncio.wait_for(
        factory.prepare("nvidia_nim/discovered", None, "config"), 1
    )
    assert selection.model == "nvidia_nim/discovered"
    assert not release.is_set()


@pytest.mark.asyncio
async def test_code_waits_for_its_provider_and_validates_settled_metadata(
    catalog_runtime,
):
    runtime, release = catalog_runtime
    factory = CodexHarnessFactory(runtime.provider_manager, binary="codex")
    waiting = asyncio.create_task(factory.prepare("groq/configured", None, "config"))
    try:
        ready = await asyncio.wait_for(
            factory.prepare("nvidia_nim/configured", None, "config"), 1
        )
        assert ready.reasoning_effort == "off"
        assert not waiting.done()
        release.set()
        selected = await asyncio.wait_for(waiting, 1)
        assert selected.model == "groq/configured"
        assert selected.reasoning_effort == "off"
    finally:
        release.set()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_settings", [False, True])
async def test_code_waits_for_model_absent_from_partial_catalog(
    catalog_runtime, replace_settings
):
    runtime, release = catalog_runtime
    manager = runtime.provider_manager
    factory = CodexHarnessFactory(manager, binary="codex")
    model = "groq/configured" if replace_settings else "groq/discovered"
    waiting = asyncio.create_task(factory.prepare(model, None, "config"))
    try:
        await asyncio.sleep(0)
        if replace_settings:
            settings = manager.current_settings().model_copy(
                update={"model_sonnet": "groq/other"}
            )
            await manager.replace(settings, commit=AsyncMock())
            await asyncio.sleep(0)
        assert not waiting.done()
        release.set()
        selected = await asyncio.wait_for(waiting, 1)
        assert selected.model == model
    finally:
        release.set()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["unknown/model", "", "groq", "groq/"])
async def test_code_rejects_invalid_selection_without_waiting(catalog_runtime, model):
    runtime, release = catalog_runtime
    factory = CodexHarnessFactory(runtime.provider_manager, binary="codex")
    with pytest.raises(CodeValidationError, match="model is unavailable"):
        await asyncio.wait_for(factory.prepare(model, None, "config"), 1)
    assert not release.is_set()


@pytest.mark.asyncio
async def test_code_rejects_selection_when_its_provider_is_removed(catalog_runtime):
    runtime, release = catalog_runtime
    manager = runtime.provider_manager
    factory = CodexHarnessFactory(manager, binary="codex")
    waiting = asyncio.create_task(factory.prepare("groq/configured", None, "config"))
    try:
        await asyncio.sleep(0)
        settings = manager.current_settings().model_copy(
            update={"model_sonnet": None, "groq_api_keys": ()}
        )
        await manager.replace(settings, commit=AsyncMock())
        with pytest.raises(CodeValidationError, match="model is unavailable"):
            await asyncio.wait_for(waiting, 1)
        assert "groq" not in manager.catalog_status()["providers"]
        assert not release.is_set()
    finally:
        release.set()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [False, True])
async def test_model_refresh_retries_failed_file_with_unchanged_metadata(
    catalog_runtime, tmp_path, repair
):
    runtime, release = catalog_runtime
    manager = runtime.provider_manager
    release.set()
    await finish_discovery(manager)
    await finish_publication(manager)
    directory = tmp_path / "catalog"
    directory.write_text("block directory creation", encoding="utf-8")
    path = directory / "models.json"
    publisher = CodexModelCatalogPublisher(path)
    manager._model_catalog_publisher = publisher
    manager.cache_model_infos("nvidia_nim", [ProviderModelInfo("new")])
    await finish_publication(manager)
    assert manager.catalog_status()["catalog_file"] == "failed"
    # First refresh restores the fixture's normal provider metadata.
    await manager.refresh_model_list_cache()
    await finish_publication(manager)
    revision = manager.catalog_status()["catalog_revision"]
    if repair:
        directory.unlink()
    with pytest.MonkeyPatch.context() as patch:
        publish = MagicMock(wraps=publisher.publish)
        patch.setattr(publisher, "publish", publish)
        result = await manager.refresh_model_list_cache()
        await finish_publication(manager)
        publish.assert_called_once()
    assert "nvidia_nim" in result.refreshed_provider_ids
    assert manager.catalog_status()["catalog_revision"] == revision
    assert manager.catalog_status()["catalog_file"] == ("ready" if repair else "failed")
    assert path.exists() is repair


@pytest.mark.asyncio
async def test_unchanged_refresh_does_not_republish_current_file(
    catalog_runtime, tmp_path, monkeypatch
):
    runtime, release = catalog_runtime
    manager = runtime.provider_manager
    release.set()
    await finish_discovery(manager)
    await finish_publication(manager)
    publisher = CodexModelCatalogPublisher(tmp_path / "models.json")
    manager._model_catalog_publisher = publisher
    manager.cache_model_infos("nvidia_nim", [ProviderModelInfo("new")])
    await manager.refresh_model_list_cache()
    await finish_publication(manager)
    assert manager.catalog_status()["catalog_file"] == "ready"
    publish = MagicMock(wraps=publisher.publish)
    monkeypatch.setattr(publisher, "publish", publish)
    await manager.refresh_model_list_cache()
    await finish_publication(manager)
    publish.assert_not_called()


@pytest.mark.asyncio
async def test_catalog_file_is_available_before_all_discovery(catalog_runtime):
    runtime, release = catalog_runtime
    manager = runtime.provider_manager
    await asyncio.wait_for(manager.wait_for_catalog_file(InitializationWait()), 1)
    assert manager.catalog_status()["catalog_file"] == "ready"
    assert not release.is_set()


@pytest.mark.asyncio
async def test_slow_file_publication_does_not_delay_catalog_notifications(
    catalog_runtime,
):
    entered, release = threading.Event(), threading.Event()
    notified = asyncio.Event()

    class Publisher:
        def publish(self, snapshot):
            entered.set()
            assert release.wait(5)

    runtime, _ = catalog_runtime
    manager = runtime.provider_manager
    manager._model_catalog_publisher = Publisher()
    manager.set_catalog_changed_callback(notified.set)
    try:
        manager.cache_model_infos("nvidia_nim", [ProviderModelInfo("new")])
        await asyncio.wait_for(notified.wait(), 1)
        assert await asyncio.to_thread(entered.wait, 1)
        assert not release.is_set()
    finally:
        release.set()
        await manager.wait_for_catalog_file(InitializationWait())


@pytest.mark.asyncio
async def test_file_publication_coalesces_changes_and_tracks_written_version(
    catalog_runtime,
):
    runtime, _ = catalog_runtime
    manager = runtime.provider_manager
    entered, release = threading.Event(), threading.Event()
    snapshots = []

    class Publisher:
        def publish(self, snapshot):
            snapshots.append(snapshot)
            if len(snapshots) == 1:
                entered.set()
                assert release.wait(5)

    manager._model_catalog_publisher = Publisher()
    try:
        manager.cache_model_infos("nvidia_nim", [ProviderModelInfo("first")])
        assert await asyncio.to_thread(entered.wait, 1)
        for name in ("second", "third", "latest"):
            manager.cache_model_infos("nvidia_nim", [ProviderModelInfo(name)])
        assert manager.catalog_status()["catalog_file"] == "starting"
        release.set()
        version = await manager.wait_for_catalog_file(InitializationWait())
        assert version == (
            manager.current_generation_id,
            manager.catalog_status()["catalog_revision"],
        )
        assert len(snapshots) == 2
        assert snapshots[0].cached_model_info("nvidia_nim", "first") is not None
        assert snapshots[-1].cached_model_info("nvidia_nim", "latest") is not None
    finally:
        release.set()
        await manager.wait_for_catalog_file(InitializationWait())


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_next", [False, True])
async def test_file_waiter_follows_publication_started_after_previous_write(
    catalog_runtime, fail_next
):
    runtime, _ = catalog_runtime
    manager = runtime.provider_manager
    await finish_publication(manager)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    snapshots = []

    class Publisher:
        def publish(self, snapshot):
            index = len(snapshots)
            snapshots.append(snapshot)
            entered[index].set()
            assert release[index].wait(5)
            if index == 1 and fail_next:
                raise OSError("catalog write failed")

    manager._model_catalog_publisher = Publisher()
    manager.cache_model_infos("nvidia_nim", [ProviderModelInfo("first")])
    waiting = asyncio.create_task(manager.wait_for_catalog_file(InitializationWait()))
    try:
        assert await asyncio.to_thread(entered[0].wait, 1)
        assert manager._publication_task is not None
        manager._publication_task.add_done_callback(
            lambda _: manager.cache_model_infos(
                "nvidia_nim", [ProviderModelInfo("later")]
            )
        )
        release[0].set()
        assert await asyncio.to_thread(entered[1].wait, 1)
        assert not waiting.done()
        release[1].set()
        if fail_next:
            with pytest.raises(ApplicationUnavailableError, match="Could not publish"):
                await waiting
            assert manager.catalog_status()["catalog_file"] == "failed"
        else:
            version = await waiting
            assert version == (
                manager.current_generation_id,
                manager.catalog_status()["catalog_revision"],
            )
            assert snapshots[-1].cached_model_info("nvidia_nim", "later") is not None
            assert manager.catalog_status()["catalog_file"] == "ready"
        assert len(snapshots) == 2
    finally:
        for event in release:
            event.set()
        await asyncio.gather(waiting, return_exceptions=True)
        await finish_publication(manager)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_file_waiter_exit_preserves_owned_publication(catalog_runtime, cancel):
    runtime, _ = catalog_runtime
    manager = runtime.provider_manager
    entered, release = threading.Event(), threading.Event()

    class Publisher:
        def publish(self, snapshot):
            entered.set()
            assert release.wait(5)

    manager._model_catalog_publisher = Publisher()
    manager.cache_model_infos("nvidia_nim", [ProviderModelInfo("new")])
    waiting = asyncio.create_task(
        manager.wait_for_catalog_file(InitializationWait(1 if cancel else 0.01))
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        if cancel:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            from free_claude_code.application.errors import ApplicationUnavailableError

            with pytest.raises(ApplicationUnavailableError, match="still starting"):
                await waiting
        assert (
            manager._publication_task is not None
            and not manager._publication_task.done()
        )
    finally:
        release.set()
        await asyncio.gather(waiting, return_exceptions=True)
        await manager.wait_for_catalog_file(InitializationWait())
