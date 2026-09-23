import os
import subprocess
from unittest.mock import patch


def test_process_registry_register_pid_zero_noop():
    """register_pid(0) is a no-op (early return)."""
    from free_claude_code.cli import process_registry as pr

    before = len(pr._pids)
    pr.register_pid(0)
    assert len(pr._pids) == before


def test_process_registry_unregister_pid_zero_noop():
    """unregister_pid(0) is a no-op."""
    from free_claude_code.cli import process_registry as pr

    pr.register_pid(99999)
    pr.unregister_pid(0)
    assert 99999 in pr._pids
    pr.unregister_pid(99999)


def test_process_registry_ensure_atexit_idempotent():
    """Second call to ensure_atexit_registered is idempotent."""
    from free_claude_code.cli import process_registry as pr

    pr.ensure_atexit_registered()
    pr.ensure_atexit_registered()
    # Should not raise; atexit handler registered once


def test_process_registry_kill_all_exception_logged_no_raise(monkeypatch):
    """Exception in os.kill/taskkill is logged but does not raise."""
    from free_claude_code.cli import process_registry as pr

    monkeypatch.setattr(pr, "_pids", {99999})
    monkeypatch.setattr(os, "name", "posix", raising=False)

    def _kill_raises(pid, sig):
        raise ProcessLookupError("no such process")

    with patch("os.kill", _kill_raises):
        pr.kill_all_best_effort()
    # Should not raise


def test_process_registry_register_unregister_does_not_crash():
    from free_claude_code.cli import process_registry as pr

    pr.register_pid(12345)
    pr.unregister_pid(12345)


def test_process_registry_kill_all_best_effort_empty_is_noop():
    from free_claude_code.cli import process_registry as pr

    # Ensure no exception on empty set
    pr.kill_all_best_effort()


def test_process_registry_kill_all_best_effort_windows_noop_when_taskkill_missing(
    monkeypatch,
):
    from free_claude_code.cli import process_registry as pr

    # Simulate windows path in a stable way.
    monkeypatch.setattr(pr, "_pids", {12345})
    monkeypatch.setattr(os, "name", "nt", raising=False)

    # If taskkill isn't callable, we still should not crash.
    import subprocess

    def _boom(*args, **kwargs):
        raise FileNotFoundError("taskkill missing")

    monkeypatch.setattr(subprocess, "run", _boom)
    pr.kill_all_best_effort()


def test_process_registry_kill_pid_tree_windows_uses_taskkill(monkeypatch):
    from free_claude_code.cli import process_registry as pr

    calls = []
    monkeypatch.setattr(os, "name", "nt", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append((a, kw)))

    pr.kill_pid_tree_best_effort(12345)

    assert calls
    assert calls[0][0][0] == ["taskkill", "/PID", "12345", "/T", "/F"]
