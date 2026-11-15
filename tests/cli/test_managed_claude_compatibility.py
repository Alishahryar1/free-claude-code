"""Managed messaging checks and owns the exact Claude binary before inference."""

import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from free_claude_code.cli.managed import session as sessions

pytestmark = pytest.mark.asyncio


@pytest.fixture
def probe(monkeypatch, tmp_path):
    session = sessions.ManagedClaudeSession(
        str(tmp_path),
        "http://localhost:8082",
        auth_token="test",
        claude_bin="./custom claude",
    )
    process = Mock(pid=123, returncode=0)
    process.communicate = AsyncMock(return_value=(b"2.1.310 (Claude Code)", None))
    process.wait = AsyncMock(return_value=0)
    create = AsyncMock(return_value=process)
    resolve = Mock(return_value=str(tmp_path / "custom claude"))
    monkeypatch.setattr(sessions.shutil, "which", resolve)
    monkeypatch.setattr(sessions.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(sessions, "register_pid", Mock())
    monkeypatch.setattr(sessions, "unregister_pid", Mock())
    monkeypatch.setattr(sessions, "kill_pid_tree_best_effort", Mock())
    return session, process, create, resolve


@pytest.mark.parametrize(
    "output,code",
    [(b"2.1.288 (Claude Code)", 0), (b"invalid", 0), (b"2.1.310 (Claude Code)", 1)],
)
async def test_rejected_probe_does_not_launch_inference(probe, output, code):
    session, process, create, _ = probe
    process.communicate.return_value = (output, None)
    process.returncode = code
    with pytest.raises(RuntimeError, match=r"2\.1\.310"):
        _ = [event async for event in session.start_task("hello")]
    assert create.await_count == 1
    assert create.call_args.args[1:] == ("--version",)
    assert not session.is_busy


async def test_supported_probe_launches_the_same_resolved_binary(probe):
    session, process, create, resolve = probe
    inference = Mock(pid=124, returncode=0, stderr=None)
    inference.stdout.read = AsyncMock(return_value=b"")
    inference.wait = AsyncMock(return_value=0)
    create.side_effect = [process, inference]
    events = [event async for event in session.start_task("hello")]
    assert events[-1]["code"] == 0
    assert (
        create.call_args_list[0].args[0]
        == create.call_args_list[1].args[0]
        == resolve.return_value
    )
    assert resolve.call_args.args[0] == str(Path(session.workspace) / "custom claude")


@pytest.mark.parametrize("cancel", [False, True])
async def test_probe_timeout_or_cancellation_reaps_child(probe, monkeypatch, cancel):
    session, process, create, _ = probe
    entered = asyncio.Event()
    process.returncode = None

    async def communicate():
        entered.set()
        await asyncio.Event().wait()

    async def wait():
        process.returncode = -9
        return -9

    process.communicate.side_effect = communicate
    process.wait.side_effect = wait
    monkeypatch.setattr(
        sessions,
        "CLAUDE_COMPATIBILITY_CHECK",
        replace(sessions.CLAUDE_COMPATIBILITY_CHECK, timeout_seconds=0.02),
    )

    async def run():
        return [event async for event in session.start_task("hello")]

    task = asyncio.create_task(run())
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    process.wait.assert_awaited()
    assert process.returncode == -9
    assert create.await_count == 1
    assert not session.is_busy


async def test_missing_configured_claude_never_spawns(probe):
    session, _, create, resolve = probe
    resolve.return_value = None
    with pytest.raises(RuntimeError, match="Claude Code"):
        _ = [event async for event in session.start_task("hello")]
    create.assert_not_awaited()
