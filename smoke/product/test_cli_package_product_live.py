import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.cli.managed.manager import ManagedClaudeSessionManager
from free_claude_code.cli.managed.session import ManagedClaudeSession
from free_claude_code.core.version import package_version
from smoke.lib.child_process import cmd_fcc_version, run_captured_text
from smoke.lib.config import SmokeConfig

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("cli")]


def test_entrypoint_version_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["USERPROFILE"] = str(tmp_path)

    result = run_captured_text(
        cmd_fcc_version(),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == f"free-claude-code {package_version()}\n"
    assert result.stderr == ""
    assert not (tmp_path / ".fcc" / ".env").exists()


@pytest.mark.asyncio
async def test_cli_session_resume_fork_e2e(tmp_path: Path) -> None:
    session = ManagedClaudeSession(
        str(tmp_path), "http://127.0.0.1:8082", auth_token="freecc"
    )
    process = AsyncMock()
    process.stdout.read.side_effect = [b""]
    process.stderr.read.return_value = b""
    process.wait.return_value = 0
    process.returncode = 0

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
        spawn.return_value = process
        async for _event in session.start_task(
            "resume this",
            session_id="sess_product",
            fork_session=True,
        ):
            pass

    args = spawn.call_args[0]
    assert args[:3] == ("claude", "--resume", "sess_product")
    assert "--fork-session" in args
    assert "-p" in args
    assert "resume this" in args


@pytest.mark.asyncio
async def test_cli_process_cleanup_e2e(tmp_path: Path) -> None:
    manager = ManagedClaudeSessionManager(
        workspace_path=str(tmp_path),
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="freecc",
    )
    session, pending_id, is_new = await manager.get_or_create_session()
    assert is_new is True
    assert pending_id.startswith("pending_")

    mocked_stop = AsyncMock(return_value=True)
    with patch.object(session, "stop", mocked_stop):
        await manager.stop_all()

    mocked_stop.assert_awaited_once()
    assert manager.get_stats() == {
        "active_sessions": 0,
        "pending_sessions": 0,
        "busy_count": 0,
    }


@pytest.mark.asyncio
async def test_cli_session_stop_kills_child_e2e(tmp_path: Path) -> None:
    session = ManagedClaudeSession(
        str(tmp_path), "http://127.0.0.1:8082", auth_token="freecc"
    )
    process = MagicMock()
    process.pid = 123456
    process.returncode = None
    process.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])
    session.process = process

    with patch(
        "free_claude_code.cli.managed.session.kill_pid_tree_best_effort"
    ) as kill_tree:
        stopped = await session.stop()

    assert stopped is True
    kill_tree.assert_called_once_with(process.pid)
    process.kill.assert_called_once()
