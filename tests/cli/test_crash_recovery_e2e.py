"""End-to-end proof that a crashed FCC server comes back by itself.

This is the behaviour the patch exists to provide, exercised through the real
`ServerSupervisor` against a real listening socket rather than a mocked run
loop: an unhandled failure is injected into a live server, and the test waits
for a *different* server instance to answer on the same port.
"""

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest

from free_claude_code.cli import commands
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.bootstrap import build_asgi_app


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture
def supervisor_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> Iterator[tuple[commands.ServerSupervisor, threading.Thread, int, str]]:
    """Run a supervised server whose HTTP app is real but provider-free."""

    port = _free_port()
    settings = Settings().model_copy(
        update={
            "host": "127.0.0.1",
            "port": port,
            "open_admin_browser": False,
            "messaging_platform": "none",
            "voice_note_enabled": False,
            "log_file": str(tmp_path / "server.log"),
        }
    )
    apps: list[RuntimeASGIApp] = []

    def create_app(loaded: Settings, *, restart_callback) -> RuntimeASGIApp:
        app = build_asgi_app(loaded, restart_callback=restart_callback)
        # Provider discovery is the only outbound call; keep the test hermetic.
        monkeypatch.setattr(
            app.runtime.provider_manager, "start_model_list_refresh", lambda: None
        )
        apps.append(app)
        return app

    monkeypatch.setattr(commands, "load_server_settings", lambda: settings)
    monkeypatch.setattr("free_claude_code.runtime.bootstrap.build_asgi_app", create_app)
    monkeypatch.setattr(commands, "kill_all_best_effort", lambda: None)

    supervisor = commands.ServerSupervisor(console_logging=False)
    thread = threading.Thread(target=supervisor.run, daemon=True)
    thread.start()

    def wait_for(predicate, timeout=30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            if not thread.is_alive():
                return False
            time.sleep(0.05)
        return False

    assert wait_for(lambda: supervisor.status == commands.ServerStatus.RUNNING), (
        "server never reached RUNNING"
    )

    yield supervisor, thread, port, "127.0.0.1"

    supervisor.request_stop()
    thread.join(20)
    assert not thread.is_alive(), "supervisor thread did not stop"
    assert all(app.runtime.is_closed for app in apps), "a runtime leaked"


def _instance_id(port: int) -> str:
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.get(f"http://127.0.0.1:{port}/admin/api/status")
        response.raise_for_status()
        return str(response.json()["instance_id"])


def test_server_recovers_from_uncaught_error(
    supervisor_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unhandled failure must yield a new instance on the same port."""

    supervisor, thread, port, _host = supervisor_server
    first_instance = _instance_id(port)
    assert first_instance

    # Generation 1 is already serving, so the crash has to be injected into a
    # later generation. Patching now means the first call into _run_bound is
    # generation 2, so that is the one made to fail; generation 3 must then
    # come up on the same port and answer with a new instance id.
    real_run_bound = supervisor._run_bound
    calls = {"count": 0}

    def counting_run_bound(settings, sockets, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected crash for recovery test")
        return real_run_bound(settings, sockets, **kwargs)

    monkeypatch.setattr(supervisor, "_run_bound", counting_run_bound)

    # Drive a clean generation boundary so the loop calls back into _run_bound.
    assert supervisor.request_restart()

    deadline = time.monotonic() + 60
    second_instance = None
    while time.monotonic() < deadline:
        if calls["count"] >= 2:
            try:
                candidate = _instance_id(port)
            except httpx.HTTPError:
                time.sleep(0.1)
                continue
            if candidate and candidate != first_instance:
                second_instance = candidate
                break
        time.sleep(0.1)

    assert thread.is_alive(), "supervisor died instead of recovering"
    assert calls["count"] >= 2, "server was never restarted after the crash"
    assert second_instance is not None, "replacement server never served requests"
    assert second_instance != first_instance
