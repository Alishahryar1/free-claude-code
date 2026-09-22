"""Launchers start one shared platform owner when the local proxy is down."""

import socket
import subprocess
import sys
import threading
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from free_claude_code.cli.launchers import common
from free_claude_code.config.paths import server_startup_lock_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch

URL = "http://127.0.0.1:8182"


def _status(code: int) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.status = code
    return response


def _healthy() -> MagicMock:
    return _status(200)


def _refused() -> URLError:
    return URLError(ConnectionRefusedError("refused"))


class FakeServer:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = 0

    def wait(self, timeout: float | None = None) -> int:
        assert self.exit_code is not None
        return self.exit_code


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_POLL_SECONDS", 0)
    monkeypatch.setattr(common, "_desktop_owner_running", lambda: False)
    monkeypatch.setattr(common, "_desktop_port_owner_running", lambda _url: False)


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
    calls: list[tuple[list[str], dict]] = []

    def popen(command: list[str], **kwargs: object) -> FakeServer:
        calls.append((command, kwargs))
        return FakeServer()

    monkeypatch.setattr(common.subprocess, "Popen", popen)
    return calls


def _responses(monkeypatch: pytest.MonkeyPatch, *responses: object) -> MagicMock:
    request = MagicMock(side_effect=list(responses))
    monkeypatch.setattr(common, "open_local_request", request)
    return request


def test_healthy_proxy_needs_no_server(monkeypatch, spawned) -> None:
    request = _responses(monkeypatch, _healthy())
    assert common.ensure_proxy_available(URL, env={}) is None
    assert request.call_count == 1
    assert spawned == []


def test_answering_but_unhealthy_proxy_is_not_restarted(monkeypatch, spawned) -> None:
    _responses(monkeypatch, _status(503))
    message = common.ensure_proxy_available(URL, env={})
    assert message is not None
    assert "returned HTTP 503" in message
    assert "A server is answering" in message
    assert "Start it manually with:" in message
    assert spawned == []


def test_missing_server_is_started_once_and_awaited(
    monkeypatch, spawned, capsys: pytest.CaptureFixture[str]
) -> None:
    request = _responses(monkeypatch, _refused(), _refused(), _refused(), _healthy())
    env = {"PATH": "/usr/bin", "MODEL": "nvidia_nim/test"}

    assert common.ensure_proxy_available(URL, env=env) is None

    assert request.call_count == 4
    assert len(spawned) == 1
    command, kwargs = spawned[0]
    assert command == common.server_command()
    expected_env = dict(env)
    if sys.platform in {"win32", "darwin"}:
        expected_env["FCC_DESKTOP_STARTED_BY_LAUNCHER"] = "1"
    assert kwargs["env"] == expected_env
    assert kwargs["env"] is not env
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is not subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    if sys.platform == "win32":
        assert kwargs["creationflags"]
    else:
        assert kwargs["start_new_session"] is True
    assert "Starting Free Claude Code" in capsys.readouterr().err
    # The startup lock is released for the next launcher.
    lock = InterprocessFileLock(server_startup_lock_path())
    assert lock.acquire()
    lock.release()


def test_server_exit_before_ready_is_reported(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused(), _refused())
    monkeypatch.setattr(
        common.subprocess, "Popen", lambda *_a, **_k: FakeServer(exit_code=3)
    )
    message = common.ensure_proxy_available(URL, env={})
    assert message is not None
    assert "exited with code 3" in message
    assert "Startup log:" in message


def test_startup_wait_is_bounded(monkeypatch, spawned) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 0.0)
    _responses(monkeypatch, *([_refused()] * 10))
    message = common.ensure_proxy_available(URL, env={})
    assert message is not None
    assert "did not become ready" in message
    assert len(spawned) == 1


def test_timed_out_start_terminates_only_its_child(monkeypatch) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 0.0)
    _responses(monkeypatch, _refused(), _refused(), _refused())
    child = FakeServer()
    monkeypatch.setattr(common, "server_command", lambda: ["fcc-test"])
    monkeypatch.setattr(common.subprocess, "Popen", lambda *_a, **_k: child)

    message = common.ensure_proxy_available(URL, env={})

    assert message is not None and "did not become ready" in message
    assert child.terminated


def test_unhealthy_started_server_is_stopped(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused(), _status(503))
    child = FakeServer()
    monkeypatch.setattr(common, "server_command", lambda: ["fcc-test"])
    monkeypatch.setattr(common.subprocess, "Popen", lambda *_a, **_k: child)

    message = common.ensure_proxy_available(URL, env={})

    assert message is not None and "returned HTTP 503" in message
    assert child.terminated


def test_ready_server_outlives_harness(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused(), _healthy())
    child = FakeServer()
    monkeypatch.setattr(common, "server_command", lambda: ["fcc-test"])
    monkeypatch.setattr(common.subprocess, "Popen", lambda *_a, **_k: child)

    assert common.ensure_proxy_available(URL, env={}) is None
    assert not child.terminated


def test_manual_desktop_startup_is_awaited_without_another_spawn(
    monkeypatch, spawned
) -> None:
    monkeypatch.setattr(common, "_desktop_port_owner_running", lambda _url: True)
    _responses(monkeypatch, _refused(), _refused(), _healthy())

    assert common.ensure_proxy_available(URL, env={}) is None
    assert spawned == []


def test_short_lived_duplicate_desktop_yields_to_manual_owner(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused(), _refused(), _healthy())
    ownership = iter((False, True))
    monkeypatch.setattr(
        common, "_desktop_port_owner_running", lambda _url: next(ownership)
    )
    child = FakeServer(exit_code=0)
    monkeypatch.setattr(common, "server_command", lambda: ["fcc-test"])
    monkeypatch.setattr(common.subprocess, "Popen", lambda *_a, **_k: child)

    assert common.ensure_proxy_available(URL, env={}) is None
    assert not child.terminated


def test_other_port_desktop_owner_fails_without_waiting_or_spawning(
    monkeypatch, spawned
) -> None:
    monkeypatch.setattr(common, "_desktop_owner_running", lambda: True)
    _responses(monkeypatch, _refused(), _refused())

    message = common.ensure_proxy_available(URL, env={})

    assert message is not None and "different port" in message
    assert spawned == []


def test_cancelled_start_stops_only_new_child(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused())
    child = FakeServer()
    monkeypatch.setattr(common, "_spawn_server", lambda _env: child)
    monkeypatch.setattr(
        common,
        "_wait_for_server",
        lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        common.ensure_proxy_available(URL, env={})

    assert child.terminated
    lock = InterprocessFileLock(server_startup_lock_path())
    assert lock.acquire()
    lock.release()


def test_early_output_is_kept_and_browser_preference_is_inherited(monkeypatch) -> None:
    monkeypatch.setattr(common, "server_command", lambda: ["fcc-test"])
    observed = {}

    def popen(_command, **kwargs):
        kwargs["stdout"].write(b"startup failed before app logging\n")
        observed.update(kwargs)
        return FakeServer()

    monkeypatch.setattr(common.subprocess, "Popen", popen)
    common._spawn_server({"FCC_OPEN_BROWSER": "0"})

    assert observed["env"]["FCC_OPEN_BROWSER"] == "0"
    assert b"startup failed before app logging" in (
        common.server_startup_log_path().read_bytes()
    )


def test_existing_unresponsive_listener_is_not_replaced(monkeypatch, spawned) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        monkeypatch.setattr(common, "PROXY_PREFLIGHT_BUDGET_SECONDS", 0)
        _responses(monkeypatch, URLError(TimeoutError("starting")))
        assert common.ensure_proxy_available(url, env={}) is not None
        assert spawned == []


def test_timed_out_tcp_connect_is_not_mistaken_for_starting_server(
    monkeypatch, spawned
) -> None:
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{unused.getsockname()[1]}"
        _responses(
            monkeypatch,
            URLError(TimeoutError("timed out")),
            URLError(TimeoutError("timed out")),
            _healthy(),
        )
        assert common.ensure_proxy_available(url, env={}) is None
        assert len(spawned) == 1


def test_other_network_error_is_not_treated_as_absent(monkeypatch, spawned) -> None:
    _responses(monkeypatch, URLError(PermissionError("blocked")))
    assert common.ensure_proxy_available(URL, env={}) is not None
    assert spawned == []


def test_spawn_failure_is_reported(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused())

    def fail(*_args: object, **_kwargs: object) -> FakeServer:
        raise OSError("no such executable")

    monkeypatch.setattr(common.subprocess, "Popen", fail)
    message = common.ensure_proxy_available(URL, env={})
    assert message is not None
    assert "Could not start" in message
    assert "no such executable" in message


@pytest.mark.parametrize("winner_succeeds", [True, False])
def test_concurrent_launchers_wait_for_the_lock_holder(
    monkeypatch, spawned, winner_succeeds: bool
) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(common, "SERVER_STARTUP_LOCK_TIMEOUT_SECONDS", 0.05)
    holder = InterprocessFileLock(server_startup_lock_path())
    assert holder.acquire()
    try:
        _responses(
            monkeypatch, _refused(), _healthy() if winner_succeeds else _refused()
        )
        message = common.ensure_proxy_available(URL, env={})
    finally:
        holder.release()
    if winner_succeeds:
        assert message is None
    else:
        assert message is not None
        assert "Another launcher is still starting" in message
    assert spawned == []


def test_startup_budget_starts_after_the_lock_is_acquired(monkeypatch, spawned) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(common, "SERVER_STARTUP_LOCK_TIMEOUT_SECONDS", 5.0)
    holder = InterprocessFileLock(server_startup_lock_path())
    assert holder.acquire()
    # The previous launcher gave up without a server, well after this
    # launcher's own startup budget would have expired.
    releaser = threading.Timer(0.2, holder.release)
    releaser.start()
    try:
        _responses(monkeypatch, _refused(), _refused(), _refused(), _healthy())
        assert common.ensure_proxy_available(URL, env={}) is None
    finally:
        releaser.cancel()
        holder.release()
    assert len(spawned) == 1


@pytest.mark.parametrize(
    "platform,command_name",
    [("win32", "fcc-desktop.exe"), ("darwin", "fcc-desktop"), ("linux", "fcc-server")],
)
def test_server_command_selects_platform_owner(
    monkeypatch, tmp_path, platform, command_name
) -> None:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    monkeypatch.setattr(
        common, "sys", SimpleNamespace(executable=str(executable), platform=platform)
    )
    sibling = executable.parent / command_name
    sibling.write_text("")
    if sys.platform != "win32":
        sibling.chmod(0o755)
    assert common.server_command() == [str(sibling)]


def test_missing_installed_owner_does_not_fallback_to_python(
    monkeypatch, tmp_path
) -> None:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    monkeypatch.setattr(
        common,
        "sys",
        SimpleNamespace(executable=str(executable), platform=sys.platform),
    )
    with pytest.raises(OSError, match="fcc-"):
        common.server_command()


@pytest.mark.parametrize("name", ["claude", "aider"])
def test_launcher_starts_the_server_then_runs_the_client(
    name: str,
    launch_capture: LaunchCapture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launch_capture.health_error = _refused()
    client_start = subprocess.Popen
    server_command = common.server_command()

    def popen(command: list[str], **kwargs: object) -> object:
        if command == server_command:
            assert kwargs["stdout"] is not subprocess.DEVNULL
            launch_capture.health_error = None
            return FakeServer()
        env = kwargs["env"]
        assert isinstance(env, Mapping)
        return client_start(command, env=env)

    monkeypatch.setattr(subprocess, "Popen", popen)
    launch(name, ["--help"])

    assert [command[-1] for command in launch_capture.commands] == ["--help"]
    health = [r for r in launch_capture.requests if r.full_url.endswith("/health")]
    assert len(health) >= 3
    assert "Starting Free Claude Code" in capsys.readouterr().err
    assert server_startup_lock_path().exists()


def test_launcher_keeps_the_manual_hint_when_the_server_cannot_start(
    launch_capture: LaunchCapture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launch_capture.health_error = _refused()
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: FakeServer(exit_code=1))
    launch("claude", [], exit_code=1)
    assert not launch_capture.commands
    err = capsys.readouterr().err
    assert "exited with code 1" in err
    assert "Startup log:" in err
