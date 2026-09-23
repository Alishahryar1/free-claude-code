"""Tests for cli/ module."""

import asyncio
import os
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.messaging.event_parser import parse_cli_event

# --- Existing Parser Tests ---


class TestCLIParser:
    """Test CLI event parsing."""

    def test_parse_text_content(self):
        """Test parsing text content from assistant message."""
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello, world!"}]},
        }
        result = parse_cli_event(event)
        assert len(result) == 1
        assert result[0]["type"] == "text_chunk"
        assert result[0]["text"] == "Hello, world!"

    def test_parse_thinking_content(self):
        """Test parsing thinking content."""
        event = {
            "type": "assistant",
            "message": {
                "content": [{"type": "thinking", "thinking": "Let me think..."}]
            },
        }
        result = parse_cli_event(event)
        assert len(result) == 1
        assert result[0]["type"] == "thinking_chunk"
        assert (
            result[0]["text"] == "Let me think...\n"
            or result[0]["text"] == "Let me think..."
        )

    def test_parse_multiple_content(self):
        """Test parsing mixed content (thinking + tools)."""
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Thinking..."},
                    {"type": "tool_use", "name": "ls", "input": {}},
                ]
            },
        }
        result = parse_cli_event(event)
        assert len(result) == 2
        assert result[0]["type"] == "thinking_chunk"
        assert result[0]["text"] == "Thinking..."
        assert result[1]["type"] == "tool_use"

    def test_parse_tool_use(self):
        """Test parsing tool use content."""
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "read_file",
                        "input": {"path": "/test"},
                    }
                ]
            },
        }
        result = parse_cli_event(event)
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "read_file"
        assert result[0]["input"] == {"path": "/test"}

    def test_parse_text_delta(self):
        """Test parsing streaming text delta."""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "streaming text"},
        }
        result = parse_cli_event(event)
        assert len(result) == 1
        assert result[0]["type"] == "text_delta"
        assert result[0]["text"] == "streaming text"

    def test_parse_thinking_delta(self):
        """Test parsing streaming thinking delta."""
        event = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "thinking..."},
        }
        result = parse_cli_event(event)
        assert len(result) == 1
        assert result[0]["type"] == "thinking_delta"
        assert result[0]["text"] == "thinking..."

    def test_parse_error(self):
        """Test parsing error event."""
        event = {"type": "error", "error": {"message": "Something went wrong"}}
        result = parse_cli_event(event)
        assert result[0]["type"] == "error"
        assert result[0]["message"] == "Something went wrong"

    def test_parse_exit_success(self):
        """Test parsing exit event with success."""
        event = {"type": "exit", "code": 0}
        result = parse_cli_event(event)
        assert result[0]["type"] == "complete"
        assert result[0]["status"] == "success"

    def test_parse_exit_failure(self):
        """Test parsing exit event with failure returns an error only."""
        event = {"type": "exit", "code": 1}
        result = parse_cli_event(event)
        assert len(result) == 1
        assert result[0] == {
            "type": "error",
            "message": "Process exited with code 1",
            "source": "exit",
            "exit_code": 1,
        }
        assert (
            "exit" in result[0]["message"].lower()
            or "code" in result[0]["message"].lower()
        )

    def test_parse_invalid_event(self):
        """Test parsing returns empty list for unrecognized event."""
        result = parse_cli_event({"type": "unknown"})
        assert result == []

    def test_parse_non_dict(self):
        """Test parsing returns empty list for non-dict input."""
        result = parse_cli_event("not a dict")
        assert result == []


# --- CLI Session Tests ---


class TestManagedClaudeSession:
    """Test ManagedClaudeSession."""

    def test_session_init(self):
        """Test ManagedClaudeSession initialization."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            workspace_path="/tmp/test",
            proxy_root_url="http://localhost:8082",
            allowed_dirs=["/home/user/projects"],
            auth_token="freecc",
        )
        assert session.workspace == os.path.normpath(os.path.abspath("/tmp/test"))
        assert session.proxy_root_url == "http://localhost:8082"
        assert not session.is_busy

    def test_session_extract_session_id(self):
        """Test session ID extraction from various event formats."""
        from free_claude_code.cli.managed.claude import (
            extract_managed_claude_session_id,
        )

        # Direct session_id field
        assert extract_managed_claude_session_id({"session_id": "abc123"}) == "abc123"
        assert extract_managed_claude_session_id({"sessionId": "abc123"}) == "abc123"

        # Nested in init
        assert (
            extract_managed_claude_session_id({"init": {"session_id": "nested123"}})
            == "nested123"
        )

        # Nested in result
        assert (
            extract_managed_claude_session_id({"result": {"session_id": "res123"}})
            == "res123"
        )

        # Conversation id
        assert (
            extract_managed_claude_session_id({"conversation": {"id": "conv123"}})
            == "conv123"
        )

        # No session ID
        assert extract_managed_claude_session_id({"type": "message"}) is None
        assert extract_managed_claude_session_id("not a dict") is None

    @pytest.mark.asyncio
    async def test_start_task_basic_flow(self):
        """Test start_task running a basic command flow."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [
            b'{"type": "message", "content": "Hello"}\n',
            b'{"session_id": "sess_1"}\n',
            b"",  # EOF
        ]
        mock_process.stderr.read.return_value = b""  # No error
        mock_process.wait.return_value = 0
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("Hello")]

            # Verify command construction
            # Arg 1 is subprocess command
            args = mock_exec.call_args[0]
            assert args[0] == "claude"
            assert "-p" in args
            assert "Hello" in args

            # Verify events
            assert (
                len(events) == 4
            )  # message, session_id, session_info (synthesized), exit
            assert events[0] == {"type": "message", "content": "Hello"}
            assert events[1] == {"type": "session_info", "session_id": "sess_1"}
            # The session_info event is yielded by _handle_line_gen right after extracting ID
            assert events[2] == {"session_id": "sess_1"}  # The original event
            assert events[3] == {"type": "exit", "code": 0, "stderr": None}

            assert session.current_session_id == "sess_1"

    @pytest.mark.asyncio
    async def test_start_task_with_session_resume(self):
        """Test resuming an existing session."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [
            b"",
        ]  # Immediate EOF
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            async for _ in session.start_task("Hello", session_id="sess_abc"):
                pass

            args = mock_exec.call_args[0]
            assert "--resume" in args
            assert "sess_abc" in args
            assert "--fork-session" not in args

    @pytest.mark.asyncio
    async def test_start_task_with_session_resume_and_fork(self):
        """Test resuming an existing session and forking."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]  # Immediate EOF
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            async for _ in session.start_task(
                "Hello", session_id="sess_abc", fork_session=True
            ):
                pass

            args = mock_exec.call_args[0]
            assert "--resume" in args
            assert "sess_abc" in args
            assert "--fork-session" in args

    @pytest.mark.asyncio
    async def test_start_task_process_failure_with_stderr(self):
        """Test process exit with error code and stderr output."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]  # No stdout
        mock_process.stderr.read.side_effect = [b"Fatal error", b""]
        mock_process.wait.return_value = 1

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("Hello")]

            # Should have error event from stderr, then exit event
            assert len(events) == 2
            assert events[0]["type"] == "error"
            assert events[0]["error"]["message"] == "Fatal error"

            assert events[1]["type"] == "exit"
            assert events[1]["code"] == 1
            assert events[1]["stderr"] == "Fatal error"

    @pytest.mark.asyncio
    async def test_start_task_stderr_while_stdout_streams(self):
        """Stderr is drained concurrently so stdout streaming is not blocked."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [
            b'{"type": "message", "content": "Hi"}\n',
            b"",
        ]
        mock_process.stderr.read.side_effect = [b"warning on stderr\n", b""]
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("Hello")]

        assert mock_process.stderr.read.await_count >= 2
        err_events = [e for e in events if e.get("type") == "error"]
        assert len(err_events) == 1
        assert "warning on stderr" in err_events[0]["error"]["message"]
        assert events[-1]["type"] == "exit"
        assert events[-1]["code"] == 0

    @pytest.mark.asyncio
    async def test_start_task_ignores_benign_claude_connectors_stderr(self):
        """Known Claude diagnostics on stderr are not surfaced as task failures."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [
            b'{"type": "message", "content": "Hi"}\n',
            b"",
        ]
        mock_process.stderr.read.side_effect = [
            b"claude.ai connectors are disabled in this environment\n",
            b"",
        ]
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("Hello")]

        assert [e for e in events if e.get("type") == "error"] == []
        assert events[-1] == {"type": "exit", "code": 0, "stderr": None}

    @pytest.mark.asyncio
    async def test_start_task_mixed_stderr_reports_only_fatal_lines(self):
        """Benign stderr diagnostics are filtered without hiding real failures."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]
        mock_process.stderr.read.side_effect = [
            (b"claude.ai connectors are disabled in this environment\nFatal error\n"),
            b"",
        ]
        mock_process.wait.return_value = 1

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("Hello")]

        assert len(events) == 2
        assert events[0] == {"type": "error", "error": {"message": "Fatal error"}}
        assert events[1] == {"type": "exit", "code": 1, "stderr": "Fatal error"}

    @pytest.mark.asyncio
    async def test_start_task_nonzero_with_only_benign_stderr_has_no_stderr_error(
        self,
    ):
        """A benign stderr line is not duplicated as the process failure reason."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]
        mock_process.stderr.read.side_effect = [
            b"claude.ai connectors are disabled in this environment\n",
            b"",
        ]
        mock_process.wait.return_value = 1

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("Hello")]

        assert events == [{"type": "exit", "code": 1, "stderr": None}]

    @pytest.mark.asyncio
    async def test_drain_stderr_bounded_retains_cap_but_drains_to_eof(self):
        """Oversized stderr is fully drained so the pipe cannot deadlock; capture is bounded."""
        from free_claude_code.cli.managed.session import (
            _MAX_STDERR_CAPTURE_BYTES,
            ManagedClaudeSession,
        )

        total_len = _MAX_STDERR_CAPTURE_BYTES + 100_000
        remaining: dict[str, int] = {"n": total_len}

        class _FakeStderr:
            async def read(self, n: int = 65536) -> bytes:
                left = remaining["n"]
                if left <= 0:
                    return b""
                take = min(n, left)
                remaining["n"] = left - take
                return b"y" * take

        class _FakeProcess:
            stderr = _FakeStderr()

        out = await ManagedClaudeSession._drain_stderr_bounded(
            cast(asyncio.subprocess.Process, _FakeProcess())
        )
        assert len(out) == _MAX_STDERR_CAPTURE_BYTES
        assert out == b"y" * _MAX_STDERR_CAPTURE_BYTES
        assert remaining["n"] == 0

    @pytest.mark.asyncio
    async def test_stop_session(self):
        """Test stopping the session process."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = MagicMock()
        mock_process.returncode = None  # Running
        # Mock wait to simulate async finish
        mock_process.wait = AsyncMock(return_value=0)

        session.process = mock_process

        with patch(
            "free_claude_code.cli.managed.session.kill_pid_tree_best_effort"
        ) as kill_tree:
            stopped = await session.stop()

        assert stopped is True
        kill_tree.assert_called_once_with(mock_process.pid)
        mock_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_stop_session_timeout_force_kill(self):
        """Test force kill if terminate times out."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = MagicMock()
        mock_process.returncode = None

        # First wait times out
        async def wait_side_effect():
            if not mock_process.kill.called:
                await asyncio.sleep(6)  # Should be > 5.0 timeout
            return 0

        # We can simulate timeout by raising TimeoutError directly on first call
        mock_process.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])

        session.process = mock_process

        with patch(
            "free_claude_code.cli.managed.session.kill_pid_tree_best_effort"
        ) as kill_tree:
            stopped = await session.stop()

        assert stopped is True
        kill_tree.assert_called_once_with(mock_process.pid)
        mock_process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_start_task_split_buffer(self):
        """Test handling of JSON split across chunks."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        # Split json: {"type": "mess... age"}
        mock_process.stdout.read.side_effect = [
            b'{"type": "mess',
            b'age", "content": "Split"}\n',
            b"",
        ]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [
                e async for e in session.start_task("test") if e["type"] == "message"
            ]

            assert len(events) == 1
            assert events[0]["content"] == "Split"

    @pytest.mark.asyncio
    async def test_start_task_remnant_buffer(self):
        """Test handling of buffer remnant at EOF (no newline at end)."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [
            b'{"type": "message", "content": "Remnant"}',  # No newline
            b"",
        ]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [
                e async for e in session.start_task("test") if e["type"] == "message"
            ]

            assert len(events) == 1
            assert events[0]["content"] == "Remnant"

    @pytest.mark.asyncio
    async def test_start_task_targets_proxy_root(self):
        """Test start_task passes the configured proxy root to Claude Code."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process
            async for _ in session.start_task("test"):
                pass

            # Check env var
            kwargs = mock_exec.call_args[1]
            env = kwargs["env"]
            assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8082"

    @pytest.mark.asyncio
    async def test_start_task_sets_proxy_auth_token(self):
        """Test start_task forwards configured proxy auth to Claude Code."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="proxy-token"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "official-key"}, clear=False),
            patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec,
        ):
            mock_exec.return_value = mock_process
            async for _ in session.start_task("test"):
                pass

            env = mock_exec.call_args.kwargs["env"]
            assert env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
            assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
            assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
            assert env["DISABLE_TELEMETRY"] == "1"
            assert "ANTHROPIC_API_KEY" not in env
            assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in env

    @pytest.mark.asyncio
    async def test_start_task_uses_sentinel_when_proxy_auth_blank(self):
        """Test start_task does not leak inherited Claude auth into proxy calls."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp",
            "http://localhost:8082",
            auth_token="freecc",
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with (
            patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "stale"}, clear=False),
            patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec,
        ):
            mock_exec.return_value = mock_process
            async for _ in session.start_task("test"):
                pass

            env = mock_exec.call_args.kwargs["env"]
            assert env["ANTHROPIC_AUTH_TOKEN"] == "freecc"

    @pytest.mark.asyncio
    async def test_start_task_allowed_dirs(self):
        """Test start_task includes allowed dirs in command."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp",
            "http://localhost:8082",
            allowed_dirs=["/dir1", "/dir2"],
            auth_token="freecc",
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b""]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process
            async for _ in session.start_task("test"):
                pass

            cmd = mock_exec.call_args[0]
            assert "--add-dir" in cmd
            assert os.path.normpath("/dir1") in cmd
            assert os.path.normpath("/dir2") in cmd

    @pytest.mark.asyncio
    async def test_start_task_json_error(self):
        """Test handling of non-JSON output from free_claude_code.cli."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b"Not valid json\n", b""]
        mock_process.stderr.read.return_value = b""
        mock_process.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_process

            events = [e async for e in session.start_task("test") if e["type"] == "raw"]

            assert len(events) == 1
            assert events[0]["content"] == "Not valid json"

    @pytest.mark.asyncio
    async def test_stop_exception(self):
        """Test exception handling during stop."""
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        session = ManagedClaudeSession(
            "/tmp", "http://localhost:8082", auth_token="freecc"
        )

        mock_process = MagicMock()
        mock_process.returncode = None

        session.process = mock_process

        with patch(
            "free_claude_code.cli.managed.session.kill_pid_tree_best_effort",
            side_effect=RuntimeError("Permission denied"),
        ):
            stopped = await session.stop()
        assert stopped is False


class TestManagedClaudeSessionManager:
    """Test ManagedClaudeSessionManager."""

    @pytest.mark.asyncio
    async def test_manager_create_session(self):
        """Test creating a new session."""
        from free_claude_code.cli.managed.manager import ManagedClaudeSessionManager

        manager = ManagedClaudeSessionManager(
            workspace_path="/tmp/test",
            proxy_root_url="http://localhost:8082",
            auth_token="freecc",
        )

        session, sid, is_new = await manager.get_or_create_session()
        assert session is not None
        assert sid.startswith("pending_")
        assert is_new is True

    @pytest.mark.asyncio
    async def test_manager_reuse_session(self):
        """Test reusing an existing session."""
        from free_claude_code.cli.managed.manager import ManagedClaudeSessionManager

        manager = ManagedClaudeSessionManager(
            workspace_path="/tmp/test",
            proxy_root_url="http://localhost:8082",
            auth_token="freecc",
        )

        # Create first session
        s1, sid1, _is_new1 = await manager.get_or_create_session()

        # Request same session
        s2, _sid2, is_new2 = await manager.get_or_create_session(session_id=sid1)

        assert s1 is s2
        assert is_new2 is False

    @pytest.mark.asyncio
    async def test_manager_stats(self):
        """Test manager stats."""
        from free_claude_code.cli.managed.manager import ManagedClaudeSessionManager

        manager = ManagedClaudeSessionManager(
            workspace_path="/tmp/test",
            proxy_root_url="http://localhost:8082",
            auth_token="freecc",
        )

        stats = manager.get_stats()
        assert stats["active_sessions"] == 0
        assert stats["pending_sessions"] == 0
