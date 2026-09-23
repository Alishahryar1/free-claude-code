import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.cli.managed.manager import ManagedClaudeSessionManager
from free_claude_code.cli.managed.session import ManagedClaudeSession


def _manager() -> ManagedClaudeSessionManager:
    return ManagedClaudeSessionManager(
        workspace_path="/tmp",
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="freecc",
    )


def _session() -> ManagedClaudeSession:
    return ManagedClaudeSession(
        "/tmp",
        "http://127.0.0.1:8082",
        auth_token="freecc",
    )


def _mock_session(*stop_results: object) -> MagicMock:
    session = MagicMock(spec=ManagedClaudeSession)
    session.is_busy = False
    session.stop = AsyncMock(side_effect=list(stop_results))
    return session


def _completed_process(pid: int) -> MagicMock:
    process = MagicMock()
    process.pid = pid
    process.returncode = 0
    process.stdout.read = AsyncMock(return_value=b"")
    process.stderr = None
    process.wait = AsyncMock(return_value=0)
    return process


@pytest.mark.asyncio
async def test_stop_is_idempotent_without_a_live_process() -> None:
    session = _session()

    assert await session.stop() is True

    process = MagicMock()
    process.pid = 101
    process.returncode = 0
    process.wait = AsyncMock()
    session.process = process

    with (
        patch(
            "free_claude_code.cli.managed.session.kill_pid_tree_best_effort"
        ) as kill_tree,
        patch("free_claude_code.cli.managed.session.unregister_pid") as unregister,
    ):
        assert await session.stop() is True

    kill_tree.assert_not_called()
    process.wait.assert_not_awaited()
    unregister.assert_called_once_with(101)


@pytest.mark.asyncio
async def test_stopped_session_reference_cannot_launch_a_new_process() -> None:
    session = _session()

    assert await session.stop() is True

    with (
        patch(
            "free_claude_code.cli.managed.session.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_completed_process(100)),
        ) as create_process,
        patch("free_claude_code.cli.managed.session.trace_event") as trace,
        pytest.raises(RuntimeError, match="closed"),
    ):
        async for _ in session.start_task("must not launch"):
            pass

    create_process.assert_not_awaited()
    trace.assert_not_called()
    assert session.is_busy is False


@pytest.mark.asyncio
async def test_launch_publication_wins_before_concurrent_stop() -> None:
    session = _session()
    launch_entered = asyncio.Event()
    release_launch = asyncio.Event()
    release_stream = asyncio.Event()
    process = MagicMock()
    process.pid = 102
    process.returncode = None
    process.stderr = None

    async def create_process(*_args: object, **_kwargs: object) -> MagicMock:
        launch_entered.set()
        await release_launch.wait()
        return process

    async def read_stdout(*_args: object, **_kwargs: object) -> bytes:
        await release_stream.wait()
        return b""

    async def wait_process() -> int:
        process.returncode = 0
        return 0

    process.stdout.read = AsyncMock(side_effect=read_stdout)
    process.wait = AsyncMock(side_effect=wait_process)

    with (
        patch(
            "free_claude_code.cli.managed.session.asyncio.create_subprocess_exec",
            side_effect=create_process,
        ),
        patch("free_claude_code.cli.managed.session.kill_pid_tree_best_effort"),
        patch("free_claude_code.cli.managed.session.register_pid"),
        patch("free_claude_code.cli.managed.session.unregister_pid"),
    ):
        stream_task = asyncio.create_task(
            _collect_session_events(session, "launch wins")
        )
        await launch_entered.wait()
        stop_task = asyncio.create_task(session.stop())
        await asyncio.sleep(0)
        stopped_before_publication = stop_task.done()

        release_launch.set()
        assert await asyncio.wait_for(stop_task, timeout=1) is True
        release_stream.set()
        events = await asyncio.wait_for(stream_task, timeout=1)

        with pytest.raises(RuntimeError, match="closed"):
            async for _ in session.start_task("cannot relaunch"):
                pass

    assert events == [{"type": "exit", "code": 0, "stderr": None}]
    assert stopped_before_publication is False


@pytest.mark.asyncio
async def test_concurrent_stop_wins_before_launch_publication() -> None:
    session = _session()
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()
    process = MagicMock()
    process.pid = 103
    process.returncode = None

    async def wait_process() -> int:
        stop_entered.set()
        await release_stop.wait()
        process.returncode = 0
        return 0

    process.wait = AsyncMock(side_effect=wait_process)
    session.process = process
    unexpected_process = _completed_process(106)

    with (
        patch(
            "free_claude_code.cli.managed.session.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=unexpected_process),
        ) as create_process,
        patch("free_claude_code.cli.managed.session.kill_pid_tree_best_effort"),
        patch("free_claude_code.cli.managed.session.unregister_pid"),
        patch("free_claude_code.cli.managed.session.trace_event") as trace,
    ):
        stop_task = asyncio.create_task(session.stop())
        await stop_entered.wait()
        stream_task = asyncio.create_task(_collect_session_events(session, "stop wins"))
        await asyncio.sleep(0)

        release_stop.set()
        assert await asyncio.wait_for(stop_task, timeout=1) is True
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wait_for(stream_task, timeout=1)

    create_process.assert_not_awaited()
    trace.assert_not_called()


@pytest.mark.asyncio
async def test_normal_sequential_starts_remain_allowed_before_stop() -> None:
    session = _session()
    processes = [_completed_process(104), _completed_process(105)]

    with (
        patch(
            "free_claude_code.cli.managed.session.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=processes),
        ) as create_process,
        patch("free_claude_code.cli.managed.session.register_pid"),
        patch("free_claude_code.cli.managed.session.unregister_pid"),
    ):
        first = await _collect_session_events(session, "first")
        second = await _collect_session_events(session, "second")

    assert first == [{"type": "exit", "code": 0, "stderr": None}]
    assert second == [{"type": "exit", "code": 0, "stderr": None}]
    assert create_process.await_count == 2


async def _collect_session_events(
    session: ManagedClaudeSession,
    prompt: str,
) -> list[dict]:
    return [event async for event in session.start_task(prompt)]


@pytest.mark.asyncio
async def test_failed_stop_keeps_pid_registered_until_retry_confirms_exit() -> None:
    session = _session()
    process = MagicMock()
    process.pid = 202
    process.returncode = None
    process.wait = AsyncMock(side_effect=[RuntimeError("wait failed"), 0])
    session.process = process

    with (
        patch("free_claude_code.cli.managed.session.kill_pid_tree_best_effort"),
        patch("free_claude_code.cli.managed.session.unregister_pid") as unregister,
    ):
        assert await session.stop() is False
        unregister.assert_not_called()

        assert await session.stop() is True

    unregister.assert_called_once_with(202)


@pytest.mark.asyncio
async def test_cancelled_stop_keeps_pid_registered_and_can_be_retried() -> None:
    session = _session()
    process = MagicMock()
    process.pid = 303
    process.returncode = None
    wait_entered = asyncio.Event()

    async def wait_until_cancelled() -> int:
        wait_entered.set()
        await asyncio.Event().wait()
        return 0

    process.wait = AsyncMock(side_effect=wait_until_cancelled)
    session.process = process

    with (
        patch("free_claude_code.cli.managed.session.kill_pid_tree_best_effort"),
        patch("free_claude_code.cli.managed.session.unregister_pid") as unregister,
    ):
        stopping = asyncio.create_task(session.stop())
        await wait_entered.wait()
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        unregister.assert_not_called()

        process.wait = AsyncMock(return_value=0)
        assert await session.stop() is True

    unregister.assert_called_once_with(303)


@pytest.mark.asyncio
async def test_task_failure_keeps_unconfirmed_process_pid_registered() -> None:
    session = _session()
    process = MagicMock()
    process.pid = 404
    process.returncode = None
    process.stdout.read = AsyncMock(side_effect=RuntimeError("read failed"))
    process.stderr = None

    with (
        patch(
            "free_claude_code.cli.managed.session.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        patch("free_claude_code.cli.managed.session.register_pid") as register,
        patch("free_claude_code.cli.managed.session.unregister_pid") as unregister,
        pytest.raises(RuntimeError, match="read failed"),
    ):
        async for _ in session.start_task("hello"):
            pass

    register.assert_called_once_with(404)
    unregister.assert_not_called()


@pytest.mark.asyncio
async def test_completed_task_wait_unregisters_confirmed_process_pid() -> None:
    session = _session()
    process = MagicMock()
    process.pid = 405
    process.returncode = None
    process.stdout.read = AsyncMock(return_value=b"")
    process.stderr = None
    process.wait = AsyncMock(return_value=0)

    with (
        patch(
            "free_claude_code.cli.managed.session.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        patch("free_claude_code.cli.managed.session.register_pid") as register,
        patch("free_claude_code.cli.managed.session.unregister_pid") as unregister,
    ):
        events = [event async for event in session.start_task("hello")]

    assert events == [{"type": "exit", "code": 0, "stderr": None}]
    register.assert_called_once_with(405)
    unregister.assert_called_once_with(405)


@pytest.mark.parametrize(
    ("first_stop", "raised"),
    [
        (False, None),
        (RuntimeError("stop failed"), RuntimeError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
@pytest.mark.asyncio
async def test_remove_failure_retains_exact_aliases_and_closes_session_for_reuse(
    first_stop: object,
    raised: type[BaseException] | None,
) -> None:
    manager = _manager()
    session = _mock_session(first_stop, True)
    manager._sessions["real_1"] = session
    manager._temp_to_real["temp_1"] = "real_1"
    manager._real_to_temp["real_1"] = "temp_1"
    expected_sessions = dict(manager._sessions)
    expected_temp_to_real = dict(manager._temp_to_real)
    expected_real_to_temp = dict(manager._real_to_temp)

    if raised is None:
        assert await manager.remove_session("real_1") is False
    else:
        with pytest.raises(raised):
            await manager.remove_session("real_1")

    assert manager._sessions == expected_sessions
    assert manager._temp_to_real == expected_temp_to_real
    assert manager._real_to_temp == expected_real_to_temp
    with pytest.raises(RuntimeError, match="closing"):
        await manager.get_or_create_session("real_1")
    with pytest.raises(RuntimeError, match="closing"):
        await manager.get_or_create_session("temp_1")

    assert await manager.remove_session("temp_1") is True
    assert manager._sessions == {}
    assert manager._temp_to_real == {}
    assert manager._real_to_temp == {}
    assert session.stop.await_count == 2


@pytest.mark.asyncio
async def test_register_rejects_a_pending_session_after_stop_failure() -> None:
    manager = _manager()
    session = _mock_session(False)
    manager._pending_sessions["temp_1"] = session

    assert await manager.remove_session("temp_1") is False
    assert await manager.register_real_session_id("temp_1", "real_1") is False
    assert manager._pending_sessions == {"temp_1": session}
    assert manager._sessions == {}
    assert manager._temp_to_real == {}
    assert manager._real_to_temp == {}


@pytest.mark.asyncio
async def test_register_rejects_real_id_owned_by_a_different_session() -> None:
    manager = _manager()
    existing = _mock_session(False)
    pending = _mock_session(True)
    manager._sessions["real_1"] = existing
    manager._pending_sessions["temp_2"] = pending

    assert await manager.register_real_session_id("temp_2", "real_1") is False

    assert manager._sessions == {"real_1": existing}
    assert manager._pending_sessions == {"temp_2": pending}
    assert manager._temp_to_real == {}
    assert manager._real_to_temp == {}


@pytest.mark.asyncio
async def test_stop_all_never_loses_a_closing_owner_outside_identity_maps() -> None:
    manager = _manager()
    orphaned_owner = _mock_session(False)
    mapped_owner = _mock_session(True)
    manager._closing_sessions.add(orphaned_owner)
    manager._sessions["real_1"] = mapped_owner

    with pytest.raises(
        RuntimeError,
        match=r"^Managed Claude session shutdown failures: 1\.$",
    ):
        await manager.stop_all()

    orphaned_owner.stop.assert_awaited_once()
    mapped_owner.stop.assert_awaited_once()
    assert manager._sessions == {}
    assert manager._closing_sessions == {orphaned_owner}

    orphaned_owner.stop = AsyncMock(return_value=True)
    await manager.stop_all()

    orphaned_owner.stop.assert_awaited_once()
    assert manager._closing_sessions == set()


@pytest.mark.asyncio
async def test_stop_all_attempts_every_session_and_retains_only_failures() -> None:
    manager = _manager()
    returned_false = _mock_session(False)
    raised_error = _mock_session(RuntimeError("secret failure detail"))
    stopped = _mock_session(True)
    manager._sessions.update(
        {
            "real_error": raised_error,
            "real_false": returned_false,
            "real_stopped": stopped,
        }
    )
    manager._temp_to_real.update(
        {
            "temp_error": "real_error",
            "temp_false": "real_false",
            "temp_stopped": "real_stopped",
        }
    )
    manager._real_to_temp.update(
        {
            "real_error": "temp_error",
            "real_false": "temp_false",
            "real_stopped": "temp_stopped",
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        await manager.stop_all()

    assert str(exc_info.value) == "Managed Claude session shutdown failures: 2."
    returned_false.stop.assert_awaited_once()
    raised_error.stop.assert_awaited_once()
    stopped.stop.assert_awaited_once()
    assert manager._sessions == {
        "real_error": raised_error,
        "real_false": returned_false,
    }
    assert manager._pending_sessions == {}
    assert manager._temp_to_real == {
        "temp_error": "real_error",
        "temp_false": "real_false",
    }
    assert manager._real_to_temp == {
        "real_error": "temp_error",
        "real_false": "temp_false",
    }
    with pytest.raises(RuntimeError, match="closing"):
        await manager.get_or_create_session("temp_false")
    with pytest.raises(RuntimeError, match="closing"):
        await manager.get_or_create_session("temp_error")

    returned_false.stop = AsyncMock(return_value=True)
    raised_error.stop = AsyncMock(return_value=True)
    await manager.stop_all()

    assert manager._sessions == {}
    assert manager._pending_sessions == {}
    assert manager._temp_to_real == {}
    assert manager._real_to_temp == {}
