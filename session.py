"""Claude Code CLI session management."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .process_registry import register_pid, unregister_pid

# Cap stderr capture so a runaway child cannot exhaust memory; pipe is still drained.
_MAX_STDERR_CAPTURE_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class ClaudeCliConfig:
    """Configuration for a managed Claude CLI subprocess."""

    workspace_path: str
    api_url: str
    allowed_dirs: list[str] = field(default_factory=list)
    plans_directory: str | None = None
    claude_bin: str = "claude"


class CLISession:
    """Manages a single persistent Claude Code CLI subprocess."""

    def __init__(
        self,
        workspace_path: str,
        api_url: str,
        allowed_dirs: list[str] | None = None,
        plans_directory: str | None = None,
        claude_bin: str = "claude",
        *,
        log_raw_cli_diagnostics: bool = False,
    ):
        self.config = ClaudeCliConfig(
            workspace_path=os.path.normpath(os.path.abspath(workspace_path)),
            api_url=api_url,
            allowed_dirs=[os.path.normpath(d) for d in (allowed_dirs or [])],
            plans_directory=plans_directory,
            claude_bin=claude_bin,
        )
        self.workspace = self.config.workspace_path
        self.api_url = self.config.api_url
        self.allowed_dirs = self.config.allowed_dirs
        self.plans_directory = self.config.plans_directory
        self.claude_bin = self.config.claude_bin
        self._log_raw_cli_diagnostics = log_raw_cli_diagnostics
        self.process: asyncio.subprocess.Process | None = None
        self.current_session_id: str | None = None
        self._is_busy = False
        self._cli_lock = asyncio.Lock()

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
            self._is_busy = True
            env = os.environ.copy()

            if "ANTHROPIC_API_KEY" not in env:
                env["ANTHROPIC_API_KEY"] = "sk-placeholder-key-for-proxy"

            env["ANTHROPIC_API_URL"] = self.api_url
            if self.api_url.endswith("/v1"):
                env["ANTHROPIC_BASE_URL"] = self.api_url[:-3]
            else:
                env["ANTHROPIC_BASE_URL"] = self.api_url

            env["TERM"] = "dumb"
            env["PYTHONIOENCODING"] = "utf-8"

            # Build command
            if session_id and not session_id.startswith("pending_"):
                cmd = [
                    self.claude_bin,
                    "--resume",
                    session_id,
                ]
                if fork_session:
                    cmd.append("--fork-session")
                cmd += [
                    "-p",
                    prompt,
                    "--output-format",
                    "stream-json",
                    "--dangerously-skip-permissions",
                    "--verbose",
                ]
                logger.info(f"Resuming Claude session {session_id}")
            else:
                cmd = [
                    self.claude_bin,
                    "-p",
                    prompt,
                    "--output-format",
                    "stream-json",
                    "--dangerously-skip-permissions",
                    "--verbose",
                ]
                logger.info("Starting new Claude session")

            if self.allowed_dirs:
                for d in self.allowed_dirs:
                    cmd.extend(["--add-dir", d])

            if self.plans_directory is not None:
                settings_json = json.dumps({"plansDirectory": self.plans_directory})
                cmd.extend(["--settings", settings_json])

            try:
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace,
                    env=env,
                )
                if self.process and self.process.pid:
                    register_pid(self.process.pid)

                if not self.process or not self.process.stdout:
                    yield {"type": "exit", "code": 1}
                    return

                session_id_extracted = False
                buffer = bytearray()
                stderr_task: asyncio.Task[bytes] | None = None
                if self.process.stderr:
                    stderr_task = asyncio.create_task(
                        self._drain_stderr_bounded(self.process)
                    )

                try:
                    while True:
                        chunk = await self.process.stdout.read(65536)
                        if not chunk:
                            if buffer:
                                line_str = buffer.decode(
                                    "utf-8", errors="replace"
                                ).strip()
                                if line_str:
                                    async for event in self._handle_line_gen(
                                        line_str, session_id_extracted
                                    ):
                                        if event.get("type") == "session_info":
                                            session_id_extracted = True
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
                                    line_str, session_id_extracted
                                ):
                                    if event.get("type") == "session_info":
                                        session_id_extracted = True
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
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
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

                return_code = await self.process.wait()
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
                if self.process and self.process.pid:
                    unregister_pid(self.process.pid)

    async def _handle_line_gen(
        self, line_str: str, session_id_extracted: bool
    ) -> AsyncGenerator[dict]:
        """Process a single line and yield events."""
        try:
            event = json.loads(line_str)
            if not session_id_extracted:
                extracted_id = self._extract_session_id(event)
                if extracted_id:
                    self.current_session_id = extracted_id
                    logger.info(f"Extracted session ID: {extracted_id}")
                    yield {"type": "session_info", "session_id": extracted_id}

            yield event
        except json.JSONDecodeError:
            if self._log_raw_cli_diagnostics:
                logger.debug("Non-JSON output: {}", line_str)
            else:
                logger.debug("Non-JSON CLI line: char_len={}", len(line_str))
            yield {"type": "raw", "content": line_str}

    def _extract_session_id(self, event: Any) -> str | None:
        """Extract session ID from CLI event."""
        if not isinstance(event, dict):
            return None

        if "session_id" in event:
            return event["session_id"]
        if "sessionId" in event:
            return event["sessionId"]

        for key in ["init", "system", "result", "metadata"]:
            if key in event and isinstance(event[key], dict):
                nested = event[key]
                if "session_id" in nested:
                    return nested["session_id"]
                if "sessionId" in nested:
                    return nested["sessionId"]

        if "conversation" in event and isinstance(event["conversation"], dict):
            conv = event["conversation"]
            if "id" in conv:
                return conv["id"]

        return None

    async def stop(self):
        """Stop the CLI process."""
        if self.process and self.process.returncode is None:
            try:
                logger.info(f"Stopping Claude CLI process {self.process.pid}")
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                if self.process and self.process.pid:
                    unregister_pid(self.process.pid)
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
        return False
