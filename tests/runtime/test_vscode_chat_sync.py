import asyncio
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import load_json_value
from free_claude_code.harnesses import vscode_chat_integration as vscode
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.runtime.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.catalog_support import (
    finish_discovery,
    finish_publication,
    pause_next_write,
)
from tests.wire_payload import array_payload, object_array, object_payload


@pytest_asyncio.fixture
async def runtime(request: pytest.FixtureRequest):
    settings = Settings(
        model="nvidia_nim/one",
        nvidia_nim_api_keys="test",
        proxy_auth_token="managed",
        messaging_platform="none",
    )
    provider = MagicMock(spec=BaseProvider)
    provider.list_model_infos = AsyncMock(
        return_value=frozenset({ProviderModelInfo("one")})
    )

    async def construct(provider_id: str, *_):
        if provider_id in {"lmstudio", "llamacpp", "ollama", "rapid_mlx"}:
            raise ConnectionError()
        return provider

    manager = ProviderRuntimeManager(
        settings,
        runtime_factory=lambda s, admission_registry, history_cache=None: (
            ProviderRuntime(
                s,
                admission_registry,
                provider_constructor=construct,
            )
        ),
    )
    app = ApplicationRuntime(
        manager,
        configuration=ConfigurationService(ManagedConfigStore()),
        transcriber=None,
    )
    if getattr(request, "param", False):
        vscode.configure(vscode.config_path(), "http://127.0.0.1:1", "old", ())
    await app.start()
    try:
        yield app
    finally:
        await app.close()


async def settle(runtime: ApplicationRuntime):
    await finish_discovery(runtime.provider_manager)
    task = runtime.integrations._vscode_update.task
    if task:
        await asyncio.wait_for(asyncio.shield(task), 5)


@pytest.mark.asyncio
async def test_catalog_updates_connected_file_and_disconnect_stays_removed(
    runtime: ApplicationRuntime,
) -> None:
    await runtime.integrations.connect_vscode_chat()
    await settle(runtime)
    path = vscode.config_path()
    runtime.provider_manager.cache_model_infos(
        "nvidia_nim", (ProviderModelInfo("two"),)
    )
    await finish_publication(runtime.provider_manager)
    await settle(runtime)
    models = object_array(load_json_value(path.read_text()))[0]["models"]
    assert {m["id"] for m in object_array(models)} == {
        "nvidia_nim/one",
        "nvidia_nim/two",
    }
    assert (await runtime.integrations.vscode_chat_status())["connected"]
    await runtime.integrations.disconnect_vscode_chat()
    runtime.integrations.catalog_changed()
    await settle(runtime)
    assert load_json_value(path.read_text()) == []
    assert not (await runtime.integrations.vscode_chat_status())["connected"]


@pytest.mark.asyncio
async def test_update_arriving_during_write_is_not_lost(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    await runtime.integrations.connect_vscode_chat()
    await settle(runtime)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = vscode.configure
    calls = 0

    def held(
        path: Path,
        proxy_root_url: str,
        auth_token: str,
        models: Sequence[CatalogModel],
        *,
        only_existing: bool = False,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original(
            path, proxy_root_url, auth_token, models, only_existing=only_existing
        )

    monkeypatch.setattr(vscode, "configure", held)
    try:
        runtime.integrations.catalog_changed()
        await asyncio.wait_for(entered.wait(), 5)
        runtime.provider_manager.cache_model_infos(
            "nvidia_nim", (ProviderModelInfo("late"),)
        )
        await finish_publication(runtime.provider_manager)
    finally:
        release.set()
    await settle(runtime)
    assert calls >= 2
    assert "nvidia_nim/late" in [
        m["id"]
        for m in object_array(
            object_array(load_json_value(vscode.config_path().read_text()))[0]["models"]
        )
    ]


@pytest.mark.asyncio
async def test_bad_file_does_not_fail_catalog_and_disconnect_needs_no_catalog(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    await settle(runtime)
    path = vscode.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken")
    await runtime.integrations.refresh_vscode_chat()
    await settle(runtime)
    assert runtime.integrations._vscode_update.state == "failed"
    assert runtime.provider_manager.catalog_status()["catalog_file"] == "ready"
    path.unlink()
    await runtime.integrations.connect_vscode_chat()
    monkeypatch.setattr(
        runtime.provider_manager,
        "catalog_snapshot",
        AsyncMock(side_effect=AssertionError("Must not wait")),
    )
    assert (await runtime.integrations.disconnect_vscode_chat())["connected"] is False


@pytest.mark.asyncio
async def test_disconnected_startup_does_not_create_file(
    runtime: ApplicationRuntime,
) -> None:
    await settle(runtime)
    assert not vscode.config_path().exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", [True], indirect=True)
async def test_startup_refreshes_owned_group_and_only_writes_once(
    runtime: ApplicationRuntime,
) -> None:
    await settle(runtime)
    path = vscode.config_path()
    assert (
        json.loads(path.read_text())[0]["models"][0]["requestHeaders"]["x-api-key"]
        == "managed"
    )
    assert json.loads(path.read_text())[0]["models"]
    before = path.stat().st_mtime_ns
    await runtime.integrations.refresh_vscode_chat()
    await settle(runtime)
    assert path.stat().st_mtime_ns == before
    assert not runtime.integrations._vscode_update.changed


@pytest.mark.asyncio
async def test_disconnect_while_refresh_waits_for_ownership_never_reconnects(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    await runtime.integrations.connect_vscode_chat()
    await settle(runtime)
    entered, release = pause_next_write(
        monkeypatch, runtime.integrations._vscode_update.access
    )
    try:
        await runtime.integrations.refresh_vscode_chat()
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(runtime.integrations.disconnect_vscode_chat(), 5)
    finally:
        release.set()
    await settle(runtime)
    assert json.loads(vscode.config_path().read_text()) == []


@pytest.mark.asyncio
async def test_replaced_generation_cannot_publish_old_url_or_token(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    await runtime.integrations.connect_vscode_chat()
    await settle(runtime)
    entered, release = pause_next_write(
        monkeypatch, runtime.integrations._vscode_update.access
    )
    try:
        await runtime.integrations.refresh_vscode_chat()
        await asyncio.wait_for(entered.wait(), 5)
        await runtime.provider_manager.replace(
            runtime.settings.model_copy(
                update={"port": 12345, "proxy_auth_token": "rotated"}
            ),
            commit=AsyncMock(),
            reason="test",
        )
    finally:
        release.set()
    await settle(runtime)
    group = array_payload(load_json_value(vscode.config_path().read_text()))[0]
    assert all(
        m["requestHeaders"] == {"x-api-key": "rotated"}
        for m in object_array(object_payload(group)["models"])
    )
    assert all(
        m["url"] == "http://127.0.0.1:12345/v1/messages"
        for m in object_array(object_payload(group)["models"])
    )


@pytest.mark.asyncio
async def test_codex_publication_failure_does_not_block_vscode(
    runtime: ApplicationRuntime,
) -> None:
    await runtime.integrations.connect_vscode_chat()
    await settle(runtime)
    publisher = MagicMock()
    publisher.publish.side_effect = PermissionError("denied")
    runtime.provider_manager._model_catalog_publisher = publisher
    runtime.provider_manager.cache_model_infos(
        "nvidia_nim", (ProviderModelInfo("new"),)
    )
    await finish_publication(runtime.provider_manager)
    await settle(runtime)
    assert runtime.provider_manager.catalog_status()["catalog_file"] == "failed"
    assert runtime.integrations._vscode_update.state == "ready"
    assert "nvidia_nim/new" in [
        m["id"]
        for m in object_array(
            object_array(load_json_value(vscode.config_path().read_text()))[0]["models"]
        )
    ]
