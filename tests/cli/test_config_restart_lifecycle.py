"""Real HTTP and supervisor coverage for runtime-owned Apply restarts."""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.code_sessions import CodeService
from free_claude_code.cli import commands, desktop
from free_claude_code.cli.launchers import common
from free_claude_code.config import paths
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.code_sessions_support import FakeHarness
from tests.web_tools_support import StubWebToolsClient


@pytest.mark.parametrize("owner_kind", ["desktop", "terminal"])
@pytest.mark.parametrize("change_port", [False, True])
def test_harness_waits_through_owner_apply_restart(
    monkeypatch, change_port, owner_kind
):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("FCC_OPEN_BROWSER", "0")
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        port = first.getsockname()[1]
        next_port = second.getsockname()[1] if change_port else port
    store = ManagedConfigStore()
    store.initialize()
    store.commit(dict(store.read().managed) | {"PORT": str(port)})
    runtimes = []

    def build(settings, restart_callback):
        manager = ProviderRuntimeManager(
            settings, runtime_factory=lambda snapshot: ProviderRuntime(snapshot, {})
        )
        monkeypatch.setattr(manager, "start_model_list_refresh", lambda: None)
        monkeypatch.setattr(manager, "_start_pass", lambda *args, **kwargs: None)
        code = CodeService(
            SQLiteCodeStore(paths.code_database_path(), paths.code_lock_path()),
            FakeHarness(),
        )
        runtime = ApplicationRuntime(
            manager,
            configuration=ConfigurationService(ManagedConfigStore()),
            code_service=code,
            transcriber=None,
            restart_callback=restart_callback,
        )
        runtimes.append(runtime)
        return RuntimeASGIApp(
            create_app(
                ApiServices(
                    requests=manager,
                    admin=runtime,
                    tasks=runtime,
                    web_tools=StubWebToolsClient(),
                    code=code,
                )
            ),
            runtime,
        )

    monkeypatch.setattr("free_claude_code.runtime.bootstrap.build_asgi_app", build)
    monkeypatch.setattr(commands, "kill_all_best_effort", lambda: None)
    monkeypatch.setattr(desktop, "config_dir_path", paths.config_dir_path)
    # Exercise both installed owner paths on every host without native tray code.
    monkeypatch.setattr(
        common,
        "sys",
        SimpleNamespace(platform="win32" if owner_kind == "desktop" else "linux"),
    )
    monkeypatch.setattr(common, "PROXY_PREFLIGHT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(common, "SERVER_STARTUP_POLL_SECONDS", 0.01)

    def unexpected_spawn(_env):
        pytest.fail("The existing FCC process must own its restart")

    monkeypatch.setattr(common, "_spawn_server", unexpected_spawn)
    restart_entered, release_restart = threading.Event(), threading.Event()
    run_once = commands.ServerSupervisor._run_once

    def gated_run(self, settings, **kwargs):
        if runtimes:
            restart_entered.set()
            assert release_restart.wait(10)
        return run_once(self, settings, **kwargs)

    monkeypatch.setattr(commands.ServerSupervisor, "_run_once", gated_run)
    wait_for_server = common._wait_for_server

    def release_when_waiting(*args, **kwargs):
        release_restart.set()
        return wait_for_server(*args, **kwargs)

    monkeypatch.setattr(common, "_wait_for_server", release_when_waiting)
    trays = []

    class Tray:
        def __init__(self, controller):
            self.controller = controller
            self.stopped = threading.Event()
            trays.append(self)

        def run(self, setup):
            setup()
            assert self.stopped.wait(20)

        def stop(self):
            self.stopped.set()

    errors = []
    supervisor = commands.ServerSupervisor(console_logging=False)

    def serve_terminal():
        try:
            supervisor.run(open_admin_browser=False)
        except BaseException as exc:
            errors.append(exc)

    owner = (
        threading.Thread(target=desktop.launch_desktop, args=(Tray,))
        if owner_kind == "desktop"
        else threading.Thread(target=serve_terminal)
    )
    owner.start()
    try:
        with httpx.Client(trust_env=False, timeout=1) as client:

            def code_ready(on_port):
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    response = client.get(
                        f"http://127.0.0.1:{on_port}/admin/api/code/bootstrap"
                    )
                    if response.json()["storage"]["state"] == "ready":
                        assert response.json()["available"]
                        return
                    time.sleep(0.01)
                pytest.fail("Code storage did not become ready")

            deadline = time.monotonic() + 10
            while True:
                try:
                    before = client.get(f"http://127.0.0.1:{port}/admin/api/status")
                    if before.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline
                time.sleep(0.01)
            code_ready(port)
            response = client.post(
                f"http://127.0.0.1:{port}/admin/api/config/apply",
                json={
                    "values": {"PORT": str(next_port)}
                    if change_port
                    else {"LOG_LEVEL": "WARNING"}
                },
            )
            assert response.status_code == 200
            assert response.json()["restart"]["automatic"]
            assert restart_entered.wait(5)
            with pytest.raises(OSError, match="already owns"):
                commands.ServerSupervisor().run(open_admin_browser=False)
            assert (
                common.ensure_proxy_available(f"http://127.0.0.1:{next_port}", env={})
                is None
            )
            after = client.get(f"http://127.0.0.1:{next_port}/admin/api/status").json()
            assert after["instance_id"] != before.json()["instance_id"]
            assert after["port"] == next_port
            assert owner.is_alive()
            code_ready(next_port)
    finally:
        release_restart.set()
        if trays:
            trays[0].controller.quit()
        else:
            supervisor.request_stop()
        owner.join(10)
    assert not owner.is_alive()
    assert not errors
    released = InterprocessFileLock(paths.server_owner_lock_path())
    assert released.acquire()
    released.release()
    assert len(runtimes) == 2
    assert all(runtime.is_closed for runtime in runtimes)


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
        monkeypatch.setattr(manager, "start_model_list_refresh", lambda: None)
        monkeypatch.setattr(manager, "_start_pass", lambda *args, **kwargs: None)
        runtime = ApplicationRuntime(
            manager,
            configuration=ConfigurationService(ManagedConfigStore()),
            transcriber=None,
            restart_callback=restart_callback,
        )
        runtimes.append(runtime)
        return RuntimeASGIApp(
            create_app(
                ApiServices(
                    requests=manager,
                    admin=runtime,
                    tasks=runtime,
                    web_tools=StubWebToolsClient(),
                )
            ),
            runtime,
        )

    monkeypatch.setattr("free_claude_code.runtime.bootstrap.build_asgi_app", build)
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
