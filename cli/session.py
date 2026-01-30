"""Claude Code CLI session management."""

import asyncio
import os
import json
import logging
from typing import AsyncGenerator, Optional, Dict, List

logger = logging.getLogger(__name__)


class CLISession:
    """Manages a single persistent Claude Code CLI subprocess."""

    def __init__(
        self,
        workspace_path: str,
        api_url: str,
        allowed_dirs: Optional[List[str]] = None,
    ):
        self.workspace = os.path.normpath(os.path.abspath(workspace_path))
        self.api_url = api_url
        self.allowed_dirs = [os.path.normpath(d) for d in (allowed_dirs or [])]
        self.process: Optional[asyncio.subprocess.Process] = None
        self.current_session_id: Optional[str] = None
        self._is_busy = False
        self._cli_lock = asyncio.Lock()

    @property
    def is_busy(self) -> bool:
        """Check if a task is currently running."""
        return self._is_busy

    async def start_task(
        self, prompt: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
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
                    "claude",
                    "--resume",
                    session_id,
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
                    "claude",
                    "-p",
                    prompt,
                    "--output-format",
                    "stream-json",
                    "--dangerously-skip-permissions",
                    "--verbose",
                ]
                logger.info(f"Starting new Claude session")

            if self.allowed_dirs:
                for d in self.allowed_dirs:
                    cmd.extend(["--add-dir", d])

            try:
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace,
                    env=env,
                )

                if not self.process or not self.process.stdout:
                    yield {"type": "exit", "code": 1}
                    return

                session_id_extracted = False
                buffer = bytearray()

                while True:
                    chunk = await self.process.stdout.read(65536)
                    if not chunk:
                        if buffer:
                            line_str = buffer.decode("utf-8", errors="replace").strip()
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

                stderr_text = None
                if self.process.stderr:
                    stderr_output = await self.process.stderr.read()
                    if stderr_output:
                        stderr_text = stderr_output.decode(
                            "utf-8", errors="replace"
                        ).strip()
                        logger.error(f"Claude CLI Stderr: {stderr_text}")
                        # Yield stderr as error event so it shows in UI
                        if stderr_text:
                            logger.info(f"CLI_SESSION: Yielding error event from stderr")
                            yield {"type": "error", "error": {"message": stderr_text}}

                return_code = await self.process.wait()
                logger.info(f"Claude CLI exited with code {return_code}, stderr_present={bool(stderr_text)}")
                if return_code != 0 and not stderr_text:
                    logger.warning(f"CLI_SESSION: Process exited with code {return_code} but no stderr captured")
                yield {
                    "type": "exit",
                    "code": return_code,
                    "stderr": stderr_text,
                }
            finally:
                self._is_busy = False

    async def _handle_line_gen(
        self, line_str: str, session_id_extracted: bool
    ) -> AsyncGenerator[dict, None]:
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
            logger.debug(f"Non-JSON output: {line_str[:100]}")
            yield {"type": "raw", "content": line_str}

    def _extract_session_id(self, event: Dict) -> Optional[str]:
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
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                return True
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
                return False
        return False
