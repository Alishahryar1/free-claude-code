"""Prove whether the supervisor survives an unexpected server crash.

These tests describe the reliability contract: a server that dies without an
explicit stop request must come back, because a silently dead proxy breaks every
connected coding agent with no way to recover except a manual relaunch.
"""

import threading
from unittest.mock import patch

import pytest

from free_claude_code.cli import commands
from free_claude_code.config.settings import Settings


def _settings() -> Settings:
    return Settings().model_copy(
        update={
            "host": "127.0.0.1",
            "port": 0,
            "open_admin_browser": False,
            "messaging_platform": "none",
            "voice_note_enabled": False,
        }
    )


def test_supervisor_restarts_after_unexpected_crash(monkeypatch) -> None:
    """A crashed server must be relaunched instead of ending the process."""

    settings = _settings()
    outcomes = iter([RuntimeError("boom"), False])
    calls: list[int] = []

    def run_once(
        self: commands.ServerSupervisor,
        loaded: Settings,
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        calls.append(1)
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(commands, "load_server_settings", lambda: settings)
    monkeypatch.setattr(commands, "kill_all_best_effort", lambda: None)
    with patch.object(commands.ServerSupervisor, "_run_once", run_once):
        commands.ServerSupervisor().run(open_admin_browser=False)

    assert len(calls) == 2, "supervisor exited after the crash instead of restarting"


def test_supervisor_does_not_restart_after_explicit_stop(monkeypatch) -> None:
    """An intentional stop must not be undone by the crash-recovery loop."""

    settings = _settings()
    calls: list[int] = []

    def run_once(
        self: commands.ServerSupervisor,
        loaded: Settings,
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        calls.append(1)
        self.request_stop()
        raise RuntimeError("crash during shutdown")

    monkeypatch.setattr(commands, "load_server_settings", lambda: settings)
    monkeypatch.setattr(commands, "kill_all_best_effort", lambda: None)
    with patch.object(commands.ServerSupervisor, "_run_once", run_once):
        commands.ServerSupervisor().run(open_admin_browser=False)

    assert len(calls) == 1
