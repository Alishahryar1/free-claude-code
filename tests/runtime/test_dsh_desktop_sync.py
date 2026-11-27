"""DSH models follow catalog publication without resurrecting disconnected state."""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from ruamel.yaml import YAML

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from free_claude_code.runtime.application import ApplicationRuntime
from tests.runtime.catalog_support import (
    finish_discovery,
    finish_publication,
    pause_next_write,
)
from tests.runtime.test_integration_startup import runtime as runtime
from tests.wire_payload import object_array, object_payload


def install():
    profile = desktop.config_home() / "profiles/desktop"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text(
        json.dumps(
            {
                "dsh": {
                    "profile": {
                        "bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"]
                    }
                },
            }
        )
    )


def route() -> JsonObject:
    rows: list[JsonObject] = YAML().load(
        (desktop.config_home() / "profiles/desktop/cordis.patch.yml").read_text()
    )
    row = next(row for row in rows if row["id"] == "llm-pi-ai")
    providers = object_payload(object_payload(row["config"])["providers"])
    return object_payload(providers["free-claude-code"])


async def settle(runtime: ApplicationRuntime):
    await finish_discovery(runtime.provider_manager)
    task = runtime.integrations._dsh_update.task
    if task:
        await asyncio.wait_for(asyncio.shield(task), 5)


@pytest_asyncio.fixture
async def connected(runtime: ApplicationRuntime):
    install()
    await runtime.start()
    try:
        await runtime.integrations.connect_dsh_desktop()
        await settle(runtime)
        yield runtime
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_late_catalog_updates_and_disconnect_remains_removed(
    connected: ApplicationRuntime,
) -> None:
    runtime = connected
    runtime.provider_manager.cache_model_infos(
        "nvidia_nim", (ProviderModelInfo("two"),)
    )
    await finish_publication(runtime.provider_manager)
    await settle(runtime)
    assert {model["id"] for model in object_array(route()["models"])} == {
        "nvidia_nim/one",
        "nvidia_nim/two",
    }
    await runtime.integrations.disconnect_dsh_desktop()
    runtime.integrations.catalog_changed()
    await settle(runtime)
    assert not desktop.has_provider(desktop.config_home())
    assert not (await runtime.integrations.dsh_desktop_status())["connected"]


@pytest.mark.asyncio
async def test_disconnect_during_write_wait_cannot_be_reconnected(
    connected: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = connected
    entered, release = pause_next_write(
        monkeypatch, runtime.integrations._dsh_update.access
    )
    try:
        await runtime.integrations.refresh_dsh_desktop()
        await asyncio.wait_for(entered.wait(), 5)
        assert (await asyncio.wait_for(runtime.integrations.dsh_desktop_status(), 1))[
            "connected"
        ]
        assert not (
            await asyncio.wait_for(runtime.integrations.disconnect_dsh_desktop(), 1)
        )["connected"]
    finally:
        release.set()
    await settle(runtime)
    assert not desktop.has_provider(desktop.config_home())


@pytest.mark.asyncio
async def test_status_and_disconnect_do_not_need_catalog(
    connected: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = connected
    monkeypatch.setattr(
        runtime.provider_manager,
        "catalog_snapshot",
        AsyncMock(side_effect=AssertionError("must not wait")),
    )
    assert (await runtime.integrations.dsh_desktop_status())["connected"]
    assert not (await runtime.integrations.disconnect_dsh_desktop())["connected"]


@pytest.mark.asyncio
async def test_new_generation_replaces_old_url_and_credential(
    connected: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = connected
    entered, release = pause_next_write(
        monkeypatch, runtime.integrations._dsh_update.access
    )
    try:
        await runtime.integrations.refresh_dsh_desktop()
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
    assert route()["baseURL"] == "http://127.0.0.1:12345/v1"
    credentials: JsonObject = YAML().load(
        (desktop.config_home() / ".credentials.yaml").read_text()
    )
    assert object_payload(credentials["refs"])[desktop.DSH_DESKTOP_API_KEY] == "rotated"
    assert (await runtime.integrations.dsh_desktop_status())["connected"]


@pytest.mark.asyncio
async def test_catalog_change_during_write_is_coalesced_without_loss(
    connected: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = connected
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = desktop.refresh_connected
    calls = 0

    def held(home: Path, proxy_root_url: str, auth_token: str, catalog: ModelCatalog):
        nonlocal calls
        calls += 1
        if calls == 1:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original(home, proxy_root_url, auth_token, catalog)

    monkeypatch.setattr(desktop, "refresh_connected", held)
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
    assert "nvidia_nim/late" in {
        model["id"] for model in object_array(route()["models"])
    }


@pytest.mark.asyncio
async def test_startup_without_connection_does_not_create_native_profile(
    runtime: ApplicationRuntime,
) -> None:
    try:
        await runtime.start()
        await settle(runtime)
        assert not desktop.config_home().exists()
        assert not desktop.has_provider(desktop.config_home())
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_first_configure_waiting_for_ownership_is_superseded_by_disconnect(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    install()
    await runtime.start()
    await settle(runtime)
    entered, release = pause_next_write(
        monkeypatch, runtime.integrations._dsh_update.access
    )
    task = asyncio.create_task(runtime.integrations.connect_dsh_desktop())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not (
            await asyncio.wait_for(runtime.integrations.disconnect_dsh_desktop(), 1)
        )["connected"]
        release.set()
        await asyncio.wait_for(task, 5)
        assert not desktop.has_provider(desktop.config_home())
        assert not (await runtime.integrations.dsh_desktop_status())["connected"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()
