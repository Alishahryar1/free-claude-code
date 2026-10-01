"""DSH models follow catalog publication without resurrecting disconnected state."""

import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from ruamel.yaml import YAML

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.paths import dsh_desktop_state_path
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from tests.runtime.test_integration_startup import runtime as runtime


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


def route():
    rows = YAML().load(
        (desktop.config_home() / "profiles/desktop/cordis.patch.yml").read_text()
    )
    return next(
        row["config"]["providers"]["free-claude-code"]
        for row in rows
        if row["id"] == "llm-pi-ai"
    )


async def settle(runtime):
    await runtime.provider_manager.wait_for_catalog()
    task = runtime._integrations._dsh_update.task
    if task:
        await asyncio.wait_for(asyncio.shield(task), 5)


@pytest_asyncio.fixture
async def connected(runtime):
    install()
    await runtime.start()
    try:
        await runtime.connect_dsh_desktop()
        await settle(runtime)
        yield runtime
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_late_catalog_updates_and_disconnect_remains_removed(connected):
    runtime = connected
    runtime.provider_manager.cache_model_infos(
        "nvidia_nim", (ProviderModelInfo("two"),)
    )
    await asyncio.gather(*runtime.provider_manager._publications)
    await settle(runtime)
    assert {model["id"] for model in route()["models"]} == {
        "nvidia_nim/one",
        "nvidia_nim/two",
    }
    await runtime.disconnect_dsh_desktop()
    runtime._integrations.catalog_changed()
    await settle(runtime)
    assert not dsh_desktop_state_path().exists()
    assert not (await runtime.dsh_desktop_status())["connected"]


@pytest.mark.asyncio
async def test_disconnect_during_catalog_wait_cannot_be_reconnected(
    connected, monkeypatch
):
    runtime = connected
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.provider_manager.wait_for_catalog

    async def held():
        entered.set()
        await release.wait()
        return await original()

    monkeypatch.setattr(runtime.provider_manager, "wait_for_catalog", held)
    try:
        await runtime.refresh_dsh_desktop()
        await asyncio.wait_for(entered.wait(), 5)
        assert (await asyncio.wait_for(runtime.dsh_desktop_status(), 1))["connected"]
        assert not (await asyncio.wait_for(runtime.disconnect_dsh_desktop(), 1))[
            "connected"
        ]
    finally:
        release.set()
    await settle(runtime)
    assert not dsh_desktop_state_path().exists()


@pytest.mark.asyncio
async def test_status_and_disconnect_do_not_need_catalog(connected, monkeypatch):
    runtime = connected
    monkeypatch.setattr(
        runtime.provider_manager,
        "wait_for_catalog",
        AsyncMock(side_effect=AssertionError("must not wait")),
    )
    assert (await runtime.dsh_desktop_status())["connected"]
    assert not (await runtime.disconnect_dsh_desktop())["connected"]


@pytest.mark.asyncio
async def test_new_generation_replaces_old_url_and_credential(connected, monkeypatch):
    runtime = connected
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.provider_manager.wait_for_catalog
    calls = 0

    async def held():
        nonlocal calls
        snapshot = await original()
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return snapshot

    monkeypatch.setattr(runtime.provider_manager, "wait_for_catalog", held)
    try:
        await runtime.refresh_dsh_desktop()
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
    credentials = YAML().load((desktop.config_home() / ".credentials.yaml").read_text())
    assert credentials["refs"][desktop.DSH_DESKTOP_API_KEY] == "rotated"
    assert (await runtime.dsh_desktop_status())["connected"]


@pytest.mark.asyncio
async def test_catalog_change_during_write_is_coalesced_without_loss(
    connected, monkeypatch
):
    runtime = connected
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = desktop.refresh_connected
    calls = 0

    def held(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(desktop, "refresh_connected", held)
    try:
        runtime._integrations.catalog_changed()
        await asyncio.wait_for(entered.wait(), 5)
        runtime.provider_manager.cache_model_infos(
            "nvidia_nim", (ProviderModelInfo("late"),)
        )
        await asyncio.gather(*runtime.provider_manager._publications)
    finally:
        release.set()
    await settle(runtime)
    assert calls >= 2
    assert "nvidia_nim/late" in {model["id"] for model in route()["models"]}


@pytest.mark.asyncio
async def test_startup_without_connection_does_not_create_native_profile(runtime):
    try:
        await runtime.start()
        await settle(runtime)
        assert not desktop.config_home().exists()
        assert not dsh_desktop_state_path().exists()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_first_configure_waiting_for_catalog_is_superseded_by_disconnect(
    runtime, monkeypatch
):
    install()
    await runtime.start()
    await settle(runtime)
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.provider_manager.wait_for_catalog

    async def held():
        entered.set()
        await release.wait()
        return await original()

    monkeypatch.setattr(runtime.provider_manager, "wait_for_catalog", held)
    task = asyncio.create_task(runtime.connect_dsh_desktop())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not (await asyncio.wait_for(runtime.disconnect_dsh_desktop(), 1))[
            "configured"
        ]
        release.set()
        await asyncio.wait_for(task, 5)
        assert not dsh_desktop_state_path().exists()
        assert not (await runtime.dsh_desktop_status())["configured"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()
