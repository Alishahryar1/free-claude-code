"""Launchers start one shared fcc-server when the local proxy is down."""

import subprocess
import sys
import threading
from collections.abc import Mapping
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

    def poll(self) -> int | None:
        return self.exit_code


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_POLL_SECONDS", 0)


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
    assert message.endswith(common.MANUAL_START_HINT)
    assert spawned == []


@pytest.mark.parametrize("value", ["0", "false", "No", " OFF "])
def test_opt_out_keeps_the_manual_hint(monkeypatch, spawned, value: str) -> None:
    _responses(monkeypatch, _refused())
    message = common.ensure_proxy_available(URL, env={common.AUTO_START_ENV: value})
    assert message is not None
    assert common.AUTO_START_ENV in message
    assert message.endswith(common.MANUAL_START_HINT)
    assert spawned == []


@pytest.mark.parametrize("value", ["", "1", "true", "anything"])
def test_auto_start_is_on_unless_disabled(value: str) -> None:
    assert common.auto_start_enabled({common.AUTO_START_ENV: value})


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
    assert kwargs["env"] == env
    assert kwargs["env"] is not env
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    if sys.platform == "win32":
        assert kwargs["creationflags"]
    else:
        assert kwargs["start_new_session"] is True
    assert "Starting Free Claude Code server" in capsys.readouterr().err
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
    assert message.endswith(common.MANUAL_START_HINT)


def test_startup_wait_is_bounded(monkeypatch, spawned) -> None:
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 0.0)
    _responses(monkeypatch, *([_refused()] * 10))
    message = common.ensure_proxy_available(URL, env={})
    assert message is not None
    assert "did not become ready" in message
    assert len(spawned) == 1


def test_spawn_failure_is_reported(monkeypatch) -> None:
    _responses(monkeypatch, _refused(), _refused())

    def fail(*_args: object, **_kwargs: object) -> FakeServer:
        raise OSError("no such executable")

    monkeypatch.setattr(common.subprocess, "Popen", fail)
    message = common.ensure_proxy_available(URL, env={})
    assert message is not None
    assert "Could not start fcc-server: no such executable" in message


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


def test_server_command_prefers_the_installed_sibling(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    monkeypatch.setattr(common.sys, "executable", str(executable))
    fallback = common.server_command()
    assert fallback[:2] == [str(executable), "-c"]
    assert "free_claude_code.cli.entrypoints" in fallback[2]

    suffix = ".exe" if sys.platform == "win32" else ""
    sibling = executable.parent / f"fcc-server{suffix}"
    sibling.write_text("")
    if sys.platform != "win32":
        # A stale, non-executable script must not shadow the package fallback.
        sibling.chmod(0o644)
        assert common.server_command() == fallback
        sibling.chmod(0o755)
    assert common.server_command() == [str(sibling)]


@pytest.mark.parametrize("name", ["claude", "aider"])
def test_launcher_starts_the_server_then_runs_the_client(
    name: str,
    launch_capture: LaunchCapture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(common.AUTO_START_ENV)
    launch_capture.health_error = _refused()
    client_start = subprocess.Popen
    server_command = common.server_command()

    def popen(command: list[str], **kwargs: object) -> object:
        if command == server_command:
            assert kwargs["stdout"] is subprocess.DEVNULL
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
    assert "Starting Free Claude Code server" in capsys.readouterr().err
    assert server_startup_lock_path().exists()


def test_launcher_keeps_the_manual_hint_when_the_server_cannot_start(
    launch_capture: LaunchCapture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(common.AUTO_START_ENV)
    launch_capture.health_error = _refused()
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: FakeServer(exit_code=1))
    launch("claude", [], exit_code=1)
    assert not launch_capture.commands
    err = capsys.readouterr().err
    assert "exited with code 1" in err
    assert err.rstrip().endswith(common.MANUAL_START_HINT)
