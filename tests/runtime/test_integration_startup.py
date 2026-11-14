import asyncio
import json
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.readiness import InitializationWait
from free_claude_code.config.api_keys import normalize_api_keys
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.paths import claude_desktop_disconnect_path
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import desktop_model_id
from free_claude_code.harnesses import (
    claude_desktop_integration as desktop,
)
from free_claude_code.harnesses import (
    claude_integration,
    claude_onboarding,
    codex_integration,
)
from free_claude_code.harnesses import (
    jetbrains_acp_integration as jb,
)
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.runtime.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.assertions import require_present
from tests.harnesses.test_integration_refresh import OLD_CLAUDE, OLD_CODEX, TOKEN
from tests.harnesses.test_jetbrains_acp_integration import install, write
from tests.web_tools_support import StubWebToolsClient
from tests.wire_payload import (
    decode_object,
    object_payload,
    response_object,
    string_payload,
)


@pytest.fixture
def runtime():
    settings = Settings().model_copy(
        update={
            "port": 8000,
            "proxy_auth_token": TOKEN,
            "messaging_platform": "none",
            "model": "nvidia_nim/one",
            "nvidia_nim_api_keys": normalize_api_keys("test"),
        }
    )
    provider = MagicMock(spec=BaseProvider)
    provider.list_model_infos = AsyncMock(
        return_value=frozenset({ProviderModelInfo("one")})
    )

    async def construct(
        _id, _settings, _admission_registry, _api_key_number, _history_cache=None
    ):
        if _id in {"lmstudio", "llamacpp", "ollama", "rapid_mlx"}:
            raise ConnectionError("Local fixture is offline")
        return provider

    manager = ProviderRuntimeManager(
        settings,
        runtime_factory=lambda snapshot, admission_registry, history_cache=None: (
            ProviderRuntime(
                snapshot, admission_registry, provider_constructor=construct
            )
        ),
    )
    return ApplicationRuntime(
        manager,
        configuration=ConfigurationService(ManagedConfigStore()),
        transcriber=None,
    )


def old_connections():
    claude, codex = claude_integration.settings_path(), codex_integration.config_path()
    claude.parent.mkdir(parents=True)
    codex.parent.mkdir(parents=True)
    claude.write_text(json.dumps(OLD_CLAUDE))
    codex.write_text(OLD_CODEX)
    return claude, codex


@pytest.mark.asyncio
async def test_jetbrains_startup_refresh_and_failed_discovery_allow_disconnect(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(jb.registry_path(), jb.system_root())

    def run_probe(
        args: Sequence[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "v24.19.0", "")

    monkeypatch.setattr(jb.subprocess, "run", run_probe)
    write(
        jb.config_path(),
        {
            "agent_servers": {
                "Claude Code (FCC)": {
                    "command": "old-node",
                    "env": {"FCC_JETBRAINS_ACP": "1", "ANTHROPIC_AUTH_TOKEN": "old"},
                }
            }
        },
    )
    try:
        await runtime.start()
        await asyncio.wait_for(
            require_present(runtime.integrations._jetbrains_update.task), 5
        )
        status = await runtime.integrations.jetbrains_acp_status()
        assert status["connected"] is True
        assert object_payload(status["update"])["changed"] is True
        entry = object_payload(
            decode_object(jb.config_path().read_text())["agent_servers"]
        )["Claude Code (FCC)"]
        assert (
            object_payload(object_payload(entry)["env"])["ANTHROPIC_AUTH_TOKEN"]
            == TOKEN
        )
        assert (
            object_payload(object_payload(entry)["env"])["ANTHROPIC_BASE_URL"]
            == "http://127.0.0.1:8000"
        )
        jb.registry_path().unlink()
        await runtime.integrations.refresh_jetbrains_acp()
        await asyncio.wait_for(
            require_present(runtime.integrations._jetbrains_update.task), 5
        )
        status = await runtime.integrations.jetbrains_acp_status()
        assert object_payload(status["update"])["state"] == "failed"
        assert status["connected"] is True
        assert (await runtime.integrations.disconnect_jetbrains_acp())[
            "connected"
        ] is False
    finally:
        await runtime.close()


async def finish(runtime: ApplicationRuntime):
    await asyncio.wait_for(
        asyncio.gather(
            require_present(runtime.integrations._claude_update.task),
            require_present(runtime.integrations._codex_update.task),
        ),
        5,
    )


@pytest.mark.asyncio
async def test_startup_refreshes_old_connections_and_status_is_read_only(
    runtime: ApplicationRuntime,
) -> None:
    claude, codex = old_connections()
    try:
        await runtime.start()
        await finish(runtime)
        for status in (
            await runtime.integrations.claude_vscode_status(),
            await runtime.integrations.codex_integration_status(),
        ):
            assert status["connected"] is True
            assert status["update"] == {
                "state": "ready",
                "changed": True,
                "message": None,
            }
        before = [p.stat().st_mtime_ns for p in (claude, codex)]
        await runtime.integrations.claude_vscode_status()
        await runtime.integrations.codex_integration_status()
        assert [p.stat().st_mtime_ns for p in (claude, codex)] == before
        assert not claude_onboarding.state_path().exists()
        await runtime.integrations.disconnect_claude_vscode()
        assert (
            object_payload(
                (await runtime.integrations.claude_vscode_status())["update"]
            )["changed"]
            is False
        )
        await runtime.integrations.refresh_claude_vscode()
        await require_present(runtime.integrations._claude_update.task)
        assert (await runtime.integrations.claude_vscode_status())["connected"] is False
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ['{"hasCompletedOnboarding":false}', "{invalid"])
async def test_startup_preserves_global_onboarding(
    runtime: ApplicationRuntime, source: str
) -> None:
    old_connections()
    state = claude_onboarding.state_path()
    state.write_text(source)
    try:
        await runtime.start()
        await finish(runtime)
        assert (await runtime.integrations.claude_vscode_status())["connected"] is True
        assert state.read_text() == source
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_codex_wait_does_not_delay_http_or_claude(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_connections()
    entered, release = asyncio.Event(), asyncio.Event()
    wait_for_catalog = runtime.provider_manager.wait_for_catalog_file

    async def held(wait: InitializationWait):
        assert wait.remaining is None
        entered.set()
        await release.wait()
        return await wait_for_catalog(wait)

    monkeypatch.setattr(runtime.provider_manager, "wait_for_catalog_file", held)
    app = create_app(
        ApiServices(
            runtime.provider_manager,
            runtime,
            runtime,
            web_tools=StubWebToolsClient(),
            integrations=runtime.integrations,
        )
    )
    try:
        await asyncio.wait_for(runtime.start(), 1)
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(
            require_present(runtime.integrations._claude_update.task), 5
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            status = response_object(await client.get("/admin/api/integrations/codex"))
            assert status["connected"] is None
            assert object_payload(status["update"])["state"] == "starting"
            assert (await runtime.integrations.claude_vscode_status())[
                "connected"
            ] is True
            assert (
                object_payload(
                    object_payload(
                        object_payload(
                            response_object(await client.get("/admin/api/status"))[
                                "startup"
                            ]
                        )["integrations"]
                    )["codex"]
                )["state"]
                == "starting"
            )
        release.set()
        await finish(runtime)
        assert (await runtime.integrations.codex_integration_status())[
            "connected"
        ] is True
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_unrecognized_codex_never_waits_for_catalog(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait = AsyncMock(side_effect=AssertionError("No connection to refresh"))
    monkeypatch.setattr(runtime.provider_manager, "wait_for_catalog_file", wait)
    try:
        await runtime.start()
        await finish(runtime)
        wait.assert_not_awaited()
        assert (await runtime.integrations.codex_integration_status())[
            "connected"
        ] is False
        assert not codex_integration.config_path().exists()
        assert not claude_onboarding.state_path().exists()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_update_failure_is_isolated_and_retry_finishes(
    runtime: ApplicationRuntime,
) -> None:
    claude, _ = old_connections()
    claude.write_text("{secret-invalid-json")
    try:
        await runtime.start()
        await finish(runtime)
        status = await runtime.integrations.claude_vscode_status()
        assert object_payload(status["update"])["state"] == "failed"
        assert "secret" not in string_payload(
            object_payload(status["update"])["message"]
        )
        assert (await runtime.integrations.codex_integration_status())[
            "connected"
        ] is True
        claude.write_text(json.dumps(OLD_CLAUDE))
        assert (
            object_payload(
                (await runtime.integrations.claude_vscode_status())["update"]
            )["state"]
            == "failed"
        )
        assert "CLAUDE_CODE_DISABLE_ADVISOR_TOOL" not in claude.read_text()
        await runtime.integrations.refresh_claude_vscode()
        task = runtime.integrations._claude_update.task
        await runtime.integrations.refresh_claude_vscode()
        assert runtime.integrations._claude_update.task is task
        await asyncio.wait_for(require_present(task), 5)
        assert (await runtime.integrations.claude_vscode_status())["connected"] is True
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_shutdown_drains_integration_writer(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    write = claude_integration.refresh_connected

    def held(path: Path, proxy_root_url: str, auth_token: str):
        entered.set()
        assert release.wait(5)
        return write(path, proxy_root_url, auth_token)

    monkeypatch.setattr(claude_integration, "refresh_connected", held)
    close = None
    try:
        await runtime.start()
        assert await asyncio.to_thread(entered.wait, 5)
        close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        assert await asyncio.wait_for(close, 5)
    finally:
        release.set()
        if close is not None:
            await close
        else:
            await runtime.close()


@pytest.mark.asyncio
async def test_unbounded_initialization_wait_remains_owned_on_cancel() -> None:
    release = asyncio.Event()
    work = asyncio.create_task(release.wait())
    wait = InitializationWait(None)
    observer = asyncio.create_task(wait.wait(work))
    try:
        await asyncio.sleep(0)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert not work.done()
        release.set()
        assert await wait.wait(work)
        assert wait.remaining is None
    finally:
        release.set()
        await work


@pytest.mark.asyncio
async def test_codex_refresh_rechecks_catalog_generation_before_writing(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_connections()
    original = runtime.provider_manager.wait_for_catalog_file
    calls = 0

    async def replace_after_wait(wait: InitializationWait):
        nonlocal calls
        generation = await original(wait)
        calls += 1
        if calls == 1:
            settings = runtime.settings.model_copy(update={"model": "nvidia_nim/new"})
            await runtime.provider_manager.replace(settings, commit=AsyncMock())
        return generation

    monkeypatch.setattr(
        runtime.provider_manager, "wait_for_catalog_file", replace_after_wait
    )
    try:
        await runtime.start()
        await finish(runtime)
        assert calls == 2
        assert (await runtime.integrations.codex_integration_status())[
            "connected"
        ] is True
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("integration", ["claude-vscode", "codex"])
async def test_refresh_endpoint_is_local_only_and_does_not_connect(
    runtime: ApplicationRuntime, integration: str
) -> None:
    app = create_app(
        ApiServices(
            runtime.provider_manager,
            runtime,
            runtime,
            web_tools=StubWebToolsClient(),
            integrations=runtime.integrations,
        )
    )
    try:
        await runtime.start()
        await finish(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            path = f"/admin/api/integrations/{integration}"
            blocked = await client.post(
                path + "/refresh", headers={"Origin": "https://other.test"}
            )
            assert blocked.status_code == 403
            response = await client.post(path + "/refresh")
            assert response.status_code == 200
            assert response.json()["update"]["state"] == "starting"
            await finish(runtime)
            assert (await client.get(path)).json()["connected"] is False
        assert not claude_integration.settings_path().exists()
        assert not codex_integration.config_path().exists()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_desktop_startup_rotates_selected_profile_and_disconnect_stays_off(
    runtime: ApplicationRuntime,
) -> None:
    root = desktop.config_root()
    desktop.configure(
        root,
        "http://localhost:9999",
        "old-token",
        True,
        disconnect_path=claude_desktop_disconnect_path(),
    )
    profile = root / "configLibrary" / f"{desktop.FCC_ID}.json"
    config = decode_object(profile.read_text())
    models = [
        {"name": desktop_model_id("nvidia_nim/one"), "labelOverride": "Pinned model"},
        {"name": desktop_model_id("nvidia_nim/two"), "maxEffort": "high"},
    ]
    config["inferenceModels"] = models
    profile.write_text(json.dumps(config))
    try:
        await runtime.start()
        await require_present(runtime.integrations._desktop_update.task)
        refreshed = decode_object(profile.read_text())
        assert refreshed["inferenceModels"] == models
        assert refreshed["inferenceGatewayApiKey"] == TOKEN
        assert refreshed["inferenceGatewayBaseUrl"] == "http://127.0.0.1:8000"
        result = await runtime.integrations.claude_desktop_status()
        assert result["connected"] is True
        assert object_payload(result["update"])["changed"] is True
        before = {p: p.stat().st_mtime_ns for p in root.rglob("*.json")}
        await runtime.integrations.claude_desktop_status()
        assert before == {p: p.stat().st_mtime_ns for p in root.rglob("*.json")}
        await runtime.integrations.disconnect_claude_desktop()
        await runtime.integrations.refresh_claude_desktop()
        await require_present(runtime.integrations._desktop_update.task)
        assert (await runtime.integrations.claude_desktop_status())[
            "connected"
        ] is False
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_desktop_failure_does_not_block_other_startup(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object):
        raise ValueError("private-invalid-file")

    monkeypatch.setattr(desktop, "refresh_connected", fail)
    try:
        await runtime.start()
        await require_present(runtime.integrations._desktop_update.task)
        await finish(runtime)
        assert runtime.integrations._desktop_update.state == "failed"
        assert "private-invalid-file" not in require_present(
            runtime.integrations._desktop_update.message
        )
        assert runtime.integrations._claude_update.state == "ready"
        assert runtime.integrations._codex_update.state == "ready"
    finally:
        await runtime.close()
