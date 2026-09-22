import asyncio
import errno
import json
import socket
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.error import URLError

import pytest
import uvicorn
from fastapi import FastAPI

from free_claude_code.cli import commands
from free_claude_code.cli.launchers import common
from free_claude_code.cli.server_socket import ServerSockets
from free_claude_code.cli.uvicorn_server import RuntimeServer
from free_claude_code.config.settings import Settings


@pytest.fixture
def browser_workers(monkeypatch):
    workers = []
    start = threading.Thread.start

    def track_start(thread):
        if thread.name == "fcc-open-admin-browser":
            workers.append(thread)
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", track_start)
    yield workers
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()


def test_listeners_remain_exclusive_until_owner_closes():
    with ServerSockets.reserve("127.0.0.1", 0) as owner:
        port = owner.sockets[0].getsockname()[1]
        with pytest.raises(OSError):
            ServerSockets.reserve("127.0.0.1", port)
    with ServerSockets.reserve("127.0.0.1", port) as replacement:
        assert replacement.sockets[0].getsockname()[1] == port


def test_partial_address_failure_closes_every_reserved_socket(monkeypatch):
    from free_claude_code.cli import server_socket

    listeners = [MagicMock(), MagicMock()]
    listeners[1].bind.side_effect = OSError(errno.EADDRINUSE, "busy")
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 12345)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 12345, 0, 0)),
    ]
    monkeypatch.setattr(server_socket.socket, "getaddrinfo", lambda *args: addresses)
    monkeypatch.setattr(
        server_socket.socket, "socket", MagicMock(side_effect=listeners)
    )
    with pytest.raises(OSError):
        ServerSockets.reserve("localhost", 12345)
    for listener in listeners:
        listener.close.assert_called_once()
    listeners[0].listen.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lifespan", "quit", "normal"])
async def test_uvicorn_readiness_and_early_exit_close_runtime(failure):
    ready, close = MagicMock(), AsyncMock(return_value=True)
    began = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app):
        if failure == "lifespan":
            raise RuntimeError("controlled startup failure")
        if failure == "quit":
            server.should_exit = True
        yield

    app = FastAPI(lifespan=lifespan)
    with ServerSockets.reserve("127.0.0.1", 0) as owner:
        port = owner.sockets[0].getsockname()[1]

        def started():
            ready()
            server.should_exit = True

        server = RuntimeServer(
            uvicorn.Config(app, log_config=None, lifespan="on"),
            begin_shutdown=began.set,
            on_started=started,
            close_runtime=close,
        )
        async with asyncio.timeout(3):
            if failure == "lifespan":
                with pytest.raises(SystemExit) as exited:
                    await server.serve(owner.sockets)
                assert exited.value.code == 3
            else:
                await server.serve(owner.sockets)
        assert began.is_set()
        close.assert_awaited_once()
        assert ready.call_count == (1 if failure == "normal" else 0)
    with ServerSockets.reserve("127.0.0.1", port):
        pass


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("browser_result", [True, False, RuntimeError("opener failed")])
def test_existing_server_must_identify_itself_as_fcc(
    monkeypatch, browser_workers, valid, browser_result
):
    settings = Settings()
    payload = {"unrelated": "server"}
    if valid:
        payload = {
            "instance_id": "a" * 32,
            "status": "running",
            "host": settings.host,
            "port": settings.port,
            "provider_status": [],
            "cached_models": {},
        }
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr(
        commands, "open_local_request", MagicMock(return_value=response)
    )
    browser = MagicMock(
        side_effect=browser_result if isinstance(browser_result, Exception) else None,
        return_value=browser_result,
    )
    monkeypatch.setattr(commands.webbrowser, "open", browser)
    assert commands.open_admin_when_ready(settings) is valid
    assert browser.call_count == int(valid)


@pytest.mark.parametrize("change", ["none", "stop", "restart", "settings"])
def test_queued_browser_rechecks_owner_before_handoff(monkeypatch, change):
    supervisor = commands.ServerSupervisor()
    settings = Settings()
    supervisor.schedule_run()
    supervisor._ready_settings = settings
    queued = []
    monkeypatch.setattr(threading.Thread, "start", lambda thread: queued.append(thread))
    browser = MagicMock(return_value=True)
    monkeypatch.setattr(commands.webbrowser, "open", browser)
    supervisor.request_open_admin()
    if change == "stop":
        supervisor.request_stop()
    elif change == "restart":
        assert supervisor.request_restart()
    elif change == "settings":
        supervisor._ready_settings = settings.model_copy()
    assert len(queued) == 1
    queued[0].run()
    assert browser.call_count == int(change == "none")


@pytest.mark.parametrize("reuse", [False, True])
def test_browser_thread_start_failure_does_not_fail_fcc(monkeypatch, reuse):
    settings = Settings()
    browser = MagicMock()
    monkeypatch.setattr(commands.webbrowser, "open", browser)
    monkeypatch.setattr(
        threading.Thread, "start", MagicMock(side_effect=RuntimeError("no thread"))
    )
    if reuse:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "instance_id": "a" * 32,
                "status": "running",
                "host": settings.host,
                "port": settings.port,
                "provider_status": [],
                "cached_models": {},
            }
        ).encode()
        monkeypatch.setattr(commands, "open_local_request", lambda *a, **kw: response)
        assert commands.open_admin_when_ready(settings)
    else:
        supervisor = commands.ServerSupervisor()
        supervisor._ready_settings = settings
        supervisor.request_open_admin()
        supervisor.request_stop()
    browser.assert_not_called()


def _run_browser_shutdown_probe(mode, outcome, directory):
    """Run real FCC lifecycle owners with only OS/browser/server dependencies faked."""
    from free_claude_code.cli import desktop, uvicorn_server
    from free_claude_code.runtime import bootstrap

    entered = threading.Event()
    patcher = pytest.MonkeyPatch()
    settings = Settings.model_construct(
        host="127.0.0.1", port=0, open_admin_browser=outcome == "automatic"
    )
    patcher.setattr(commands, "load_server_settings", lambda: settings)
    patcher.setattr(desktop, "get_settings", lambda: settings)
    patcher.setattr(desktop, "config_dir_path", lambda: Path(directory))
    patcher.setattr(
        desktop,
        "desktop_port_lock_path",
        lambda port: Path(directory) / f"desktop.{port}.lock",
    )

    def browser(url):
        assert url == "http://127.0.0.1:0/admin"
        entered.set()
        if outcome in {"automatic", "tray", "quit", "timeout"}:
            threading.Event().wait()  # Intentionally never released in this process.
        if outcome == "error":
            raise RuntimeError("opener failed")
        return outcome != "false"

    patcher.setattr(commands.webbrowser, "open", browser)
    cleaned = []
    patcher.setattr(commands, "kill_all_best_effort", lambda: cleaned.append("managed"))

    if mode in {"server", "desktop"}:
        supervisor = commands.ServerSupervisor(console_logging=False)
        runtime = SimpleNamespace(
            is_closed=True,
            begin_shutdown=lambda: None,
            http_started=lambda: None,
            close=AsyncMock(return_value=True),
        )
        patcher.setattr(
            bootstrap,
            "build_asgi_app",
            lambda *a, **kw: SimpleNamespace(runtime=runtime),
        )

        class Server:
            def __init__(self, _config, *, on_started, **kw):
                self.on_started = on_started
                self.started = True
                self.should_exit = False

            def run(self, **kw):
                self.on_started()
                if mode == "server":
                    if outcome == "tray":
                        supervisor.request_open_admin()
                    assert entered.wait(2)
                    supervisor.request_stop()
                else:
                    assert supervisor.stop_event.wait(3)

        patcher.setattr(uvicorn_server, "RuntimeServer", Server)

        class Tray:
            def __init__(self, controller):
                self.controller = controller

            def run(self, setup):
                setup()
                if outcome == "tray":
                    self.controller.open_admin()
                assert entered.wait(2)
                self.controller.quit()

            def stop(self):
                pass

        if mode == "server":
            supervisor.run()
        else:
            desktop.DesktopController(
                supervisor, Tray, supervisor.request_open_admin
            ).run()
        assert cleaned == ["managed"]
        assert supervisor.status is commands.ServerStatus.STOPPED
    else:
        # Keep a real existing listener and (for desktop reuse) its singleton lock.
        from free_claude_code.core.interprocess_lock import InterprocessFileLock

        with ServerSockets.reserve("127.0.0.1", 0) as owner:
            settings.port = owner.sockets[0].getsockname()[1]

            def reuse_browser(url):
                assert url == f"http://127.0.0.1:{settings.port}/admin"
                return browser("http://127.0.0.1:0/admin")

            patcher.setattr(commands.webbrowser, "open", reuse_browser)
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {
                    "instance_id": "a" * 32,
                    "status": "running",
                    "host": settings.host,
                    "port": settings.port,
                    "provider_status": [],
                    "cached_models": {},
                }
            ).encode()
            patcher.setattr(commands, "open_local_request", lambda *a, **kw: response)

            class Tray:
                def __init__(self, controller):
                    self.controller = controller
                    self.stopped = threading.Event()

                def run(self, setup):
                    setup()
                    assert entered.wait(2)
                    if outcome == "quit":
                        self.controller.quit()
                    assert self.stopped.wait(6 if outcome == "timeout" else 2)

                def stop(self):
                    self.stopped.set()

            lock = InterprocessFileLock(Path(directory) / "desktop.lock")
            port_lock = InterprocessFileLock(
                Path(directory) / f"desktop.{settings.port}.lock"
            )
            try:
                if mode == "reuse-desktop":
                    assert port_lock.acquire()
                    assert lock.acquire()
                desktop.launch_desktop(Tray)
                assert cleaned == []
                with pytest.raises(OSError):
                    ServerSockets.reserve(settings.host, settings.port)
            finally:
                lock.release()
                port_lock.release()
    patcher.undo()
    print("FCC exited while preserving its resource ownership", flush=True)


@pytest.mark.parametrize("mode", ["server", "desktop"])
@pytest.mark.parametrize("outcome", ["automatic", "tray"])
def test_owned_fcc_process_exits_with_stalled_browser(tmp_path, mode, outcome):
    _assert_browser_probe_exits(tmp_path, mode, outcome)


@pytest.mark.parametrize("mode", ["reuse-desktop", "reuse-terminal"])
@pytest.mark.parametrize("outcome", ["success", "false", "error", "quit", "timeout"])
def test_reusing_desktop_process_exits_without_owning_browser(tmp_path, mode, outcome):
    _assert_browser_probe_exits(tmp_path, mode, outcome)


def _assert_browser_probe_exits(tmp_path, mode, outcome):
    script = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1])['_run_browser_shutdown_probe'](*sys.argv[2:])"
    )
    # subprocess.run kills and reaps only this disposable child on timeout.
    completed = subprocess.run(
        [sys.executable, "-c", script, __file__, mode, outcome, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "FCC exited while preserving its resource ownership" in completed.stdout


def test_external_probe_is_cancelled_before_any_request(monkeypatch):
    stop = threading.Event()
    stop.set()
    request = MagicMock()
    monkeypatch.setattr(commands, "open_local_request", request)
    assert not commands.open_admin_when_ready(Settings(), stop_event=stop)
    request.assert_not_called()


def test_launcher_retries_starting_http_but_not_refused_socket(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.status = 200
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        request = MagicMock(side_effect=[TimeoutError("starting"), response])
        monkeypatch.setattr(common, "open_local_request", request)
        assert common.preflight_proxy(url) is None
        assert request.call_count == 2
    request.reset_mock(side_effect=True)
    request.side_effect = URLError(ConnectionRefusedError("refused"))
    assert common.preflight_proxy(url) == "refused"
    assert request.call_count == 1


def test_launcher_http_wait_has_a_finite_budget(monkeypatch):
    monkeypatch.setattr(common.time, "monotonic", MagicMock(side_effect=[0, 0, 31]))
    request = MagicMock(side_effect=TimeoutError("still starting"))
    monkeypatch.setattr(common, "open_local_request", request)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert (
            common.preflight_proxy(f"http://127.0.0.1:{listener.getsockname()[1]}")
            == "still starting"
        )
    assert request.call_count == 1
