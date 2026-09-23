"""Managed Claude Code subprocess session."""

import asyncio
import os
from collections.abc import AsyncGenerator

from loguru import logger

from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.core.trace import trace_event

from .claude import (
    ManagedClaudeConfig,
    ManagedClaudeParseState,
    ManagedClaudeTaskRequest,
    build_managed_claude_invocation,
    parse_managed_claude_stdout_line,
)
from .diagnostics import classify_managed_claude_stderr

# Cap stderr capture so a runaway child cannot exhaust memory; pipe is still drained.
_MAX_STDERR_CAPTURE_BYTES = 256 * 1024


class ManagedClaudeSession:
    """Manages a single persistent Claude Code subprocess."""

    def __init__(
        self,
        workspace_path: str,
        proxy_root_url: str,
        allowed_dirs: list[str] | None = None,
        claude_bin: str = "claude",
        *,
        auth_token: str,
        log_raw_cli_diagnostics: bool = False,
    ):
        self.config = ManagedClaudeConfig(
            workspace_path=os.path.normpath(os.path.abspath(workspace_path)),
            proxy_root_url=proxy_root_url,
            allowed_dirs=[os.path.normpath(d) for d in (allowed_dirs or [])],
            claude_bin=claude_bin,
            auth_token=auth_token,
        )
        self.workspace = self.config.workspace_path
        self.proxy_root_url = self.config.proxy_root_url
        self.allowed_dirs = self.config.allowed_dirs
        self.claude_bin = self.config.claude_bin
        self.auth_token = self.config.auth_token
        self._log_raw_cli_diagnostics = log_raw_cli_diagnostics
        self.process: asyncio.subprocess.Process | None = None
        self.current_session_id: str | None = None
        self._is_busy = False
        self._cli_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    async def _drain_stderr_bounded(
        process: asyncio.subprocess.Process,
        *,
        max_bytes: int = _MAX_STDERR_CAPTURE_BYTES,
    ) -> bytes:
        """Read stderr concurrently with stdout to avoid subprocess pipe deadlocks.

        Retains at most ``max_bytes`` for logging; any excess is discarded, but
        the pipe is read until EOF so a noisy child cannot fill the buffer and
        block forever.
        """
        if not process.stderr:
            return b""
        parts: list[bytes] = []
        received = 0
        while True:
            chunk = await process.stderr.read(65_536)
            if not chunk:
                break
            if received < max_bytes:
                take = min(len(chunk), max_bytes - received)
                if take:
                    parts.append(chunk[:take])
                    received += take
            # If already at cap, keep reading and discarding until EOF.
        return b"".join(parts)

    @property
    def is_busy(self) -> bool:
        """Check if a task is currently running."""
        return self._is_busy

    async def start_task(
        self, prompt: str, session_id: str | None = None, fork_session: bool = False
    ) -> AsyncGenerator[dict]:
        """
        Start a new task or continue an existing session.

        Args:
            prompt: The user's message/prompt
            session_id: Optional session ID to resume

        Yields:
            Event dictionaries from the CLI
        """
        async with self._cli_lock:
            process: asyncio.subprocess.Process | None = None
            termination_confirmed = False
            try:
                async with self._lifecycle_lock:
                    if self._closed:
                        raise RuntimeError("Managed Claude session is closed.")
                    self._is_busy = True
                    invocation = build_managed_claude_invocation(
                        config=self.config,
                        request=ManagedClaudeTaskRequest(
                            prompt=prompt,
                            session_id=session_id,
                            fork_session=fork_session,
                        ),
                        base_env=os.environ,
                    )

                    trace_event(
                        stage="claude_cli",
                        event="claude_cli.process.launch",
                        source="claude_cli",
                        **invocation.trace_metadata,
                    )

                    process = await asyncio.create_subprocess_exec(
                        *invocation.argv,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=invocation.cwd,
                        env=invocation.env,
                    )
                    self.process = process
                    if process.pid:
                        register_pid(process.pid)

                if not process.stdout:
                    yield {"type": "exit", "code": 1}
                    return

                parse_state = ManagedClaudeParseState(
                    log_raw_cli_diagnostics=self._log_raw_cli_diagnostics
                )
                buffer = bytearray()
                stderr_task: asyncio.Task[bytes] | None = None
                if process.stderr:
                    stderr_task = asyncio.create_task(
                        self._drain_stderr_bounded(process)
                    )

                try:
                    while True:
                        chunk = await process.stdout.read(65536)
                        if not chunk:
                            if buffer:
                                line_str = buffer.decode(
                                    "utf-8", errors="replace"
                                ).strip()
                                if line_str:
                                    async for event in self._handle_line_gen(
                                        line_str, parse_state
                                    ):
                                        yield event
                            break

                        buffer.extend(chunk)

                        while True:
                            newline_pos = buffer.find(b"\n")
                            if newline_pos == -1:
                                break

                            line = buffer[:newline_pos]
                            buffer = buffer[newline_pos + 1 :]

                            line_str = line.decode("utf-8", errors="replace").strip()
                            if line_str:
                                async for event in self._handle_line_gen(
                                    line_str, parse_state
                                ):
                                    yield event
                except asyncio.CancelledError:
                    # Cancelling the handler task should not leave a Claude CLI
                    # subprocess running in the background.
                    await asyncio.shield(self.stop())
                    raise
                finally:
                    stderr_bytes = b""
                    if stderr_task is not None:
                        stderr_bytes = await stderr_task

                stderr_text = None
                if stderr_bytes:
                    raw_stderr_text = stderr_bytes.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    if raw_stderr_text:
                        diagnostics = classify_managed_claude_stderr(raw_stderr_text)
                        if diagnostics.has_benign:
                            logger.info(
                                "Claude CLI benign stderr diagnostics: lines={}",
                                len(diagnostics.benign_lines),
                            )
                        stderr_text = diagnostics.fatal_text
                    if stderr_text:
                        if self._log_raw_cli_diagnostics:
                            logger.error("Claude CLI stderr: {}", stderr_text)
                        else:
                            logger.error(
                                "Claude CLI stderr: bytes={} text_chars={}",
                                len(stderr_bytes),
                                len(stderr_text),
                            )
                        logger.info("CLI_SESSION: Yielding error event from stderr")
                        yield {"type": "error", "error": {"message": stderr_text}}

                return_code = await process.wait()
                termination_confirmed = True
                logger.info(
                    f"Claude CLI exited with code {return_code}, stderr_present={bool(stderr_text)}"
                )
                if return_code != 0 and not stderr_text:
                    logger.warning(
                        f"CLI_SESSION: Process exited with code {return_code} but no stderr captured"
                    )
                yield {
                    "type": "exit",
                    "code": return_code,
                    "stderr": stderr_text,
                }
            finally:
                self._is_busy = False
                if (
                    process
                    and process.pid
                    and (termination_confirmed or process.returncode is not None)
                ):
                    unregister_pid(process.pid)

    async def _handle_line_gen(
        self, line_str: str, parse_state: ManagedClaudeParseState
    ) -> AsyncGenerator[dict]:
        """Process a single line and yield events."""
        for event in parse_managed_claude_stdout_line(line_str, parse_state):
            if isinstance(event, dict) and event.get("type") == "session_info":
                session_id = event.get("session_id")
                if isinstance(session_id, str):
                    self.current_session_id = session_id
            yield event

    async def stop(self) -> bool:
        """Stop the CLI process, retaining PID ownership until exit is confirmed."""
        async with self._lifecycle_lock:
            self._closed = True
            process = self.process
            if process is None:
                return True
            if process.returncode is not None:
                if process.pid:
                    unregister_pid(process.pid)
                return True

            try:
                logger.info(f"Stopping Claude CLI process {process.pid}")
                kill_pid_tree_best_effort(process.pid)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                if process.pid:
                    unregister_pid(process.pid)
                return True
            except Exception as e:
                if self._log_raw_cli_diagnostics:
                    logger.error(
                        "Error stopping process: {}: {}",
                        type(e).__name__,
                        e,
                    )
                else:
                    logger.error(
                        "Error stopping process: exc_type={}",
                        type(e).__name__,
                    )
                return False
