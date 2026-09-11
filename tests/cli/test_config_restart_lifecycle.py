"""Real HTTP and supervisor coverage for runtime-owned Apply restarts."""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import httpx
import pytest

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.cli import commands
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.mark.parametrize("stop_during_commit", [False, True])
def test_supervised_http_apply_finishes_and_reconnects(monkeypatch, stop_during_commit):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    store = ManagedConfigStore()
    store.initialize()
    store.commit(dict(store.read().managed) | {"PORT": str(port)})
    runtimes = []

    def build(settings, restart_callback):
        manager = ProviderRuntimeManager(
            settings, runtime_factory=lambda snapshot: ProviderRuntime(snapshot, {})
        )
        monkeypatch.setattr(manager, "warm_referenced_model_cache", AsyncMock())
        monkeypatch.setattr(manager, "start_model_list_refresh", lambda: None)
        monkeypatch.setattr(manager, "_refresh_generation_in_background", AsyncMock())
        runtime = ApplicationRuntime(
            manager,
            integrations=IntegrationService(LocalInstallations.current()),
            configuration=ConfigurationService(ManagedConfigStore()),
            transcriber=None,
            restart_callback=restart_callback,
        )
        runtimes.append(runtime)
        return RuntimeASGIApp(
            create_app(ApiServices(requests=manager, admin=runtime, tasks=runtime)),
            runtime,
        )

    monkeypatch.setattr(commands, "build_asgi_app", build)
    monkeypatch.setattr(commands, "kill_all_best_effort", lambda: None)
    supervisor = commands.ServerSupervisor(console_logging=False)
    errors = []

    def serve():
        try:
            supervisor.run(open_admin_browser=False)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    entered, release = threading.Event(), threading.Event()
    commit = ManagedConfigStore.commit

    def blocked_commit(self, values):
        entered.set()
        assert release.wait(10)
        commit(self, values)

    def wait_status(client, old_id=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert not errors
            try:
                response = client.get("/admin/api/status")
                if response.status_code == 200:
                    status = response.json()
                    if (
                        status["status"] == "running"
                        and status["instance_id"] != old_id
                    ):
                        return status
            except httpx.TransportError:
                pass
            time.sleep(0.02)
        pytest.fail("supervisor did not become ready")

    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5
        ) as client:
            before = wait_status(client)
            monkeypatch.setattr(ManagedConfigStore, "commit", blocked_commit)
            with ThreadPoolExecutor(max_workers=1) as executor:
                apply = executor.submit(
                    client.post,
                    "/admin/api/config/apply",
                    json={"values": {"LOG_LEVEL": "WARNING"}},
                )
                try:
                    assert entered.wait(5)
                    if stop_during_commit:
                        supervisor.request_stop()
                finally:
                    release.set()
                response = apply.result(timeout=10)
            assert response.status_code == 200
            result = response.json()
            assert result["applied"]
            assert result["restart"]["automatic"]
            assert result["restart"]["instance_id"] == before["instance_id"]
            assert result["restart"]["admin_url"] == f"http://127.0.0.1:{port}/admin"
            if not stop_during_commit:
                after = wait_status(client, before["instance_id"])
                assert after["pending_fields"] == []
                assert len(runtimes) == 2
                assert runtimes[0].is_closed
    finally:
        release.set()
        supervisor.request_stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors
    assert all(runtime.is_closed for runtime in runtimes)
    assert len(runtimes) == (1 if stop_during_commit else 2)
