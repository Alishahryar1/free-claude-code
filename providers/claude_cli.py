import asyncio
import os
import json
import logging
from typing import AsyncGenerator, Optional, Dict, List

logger = logging.getLogger(__name__)


class CLIParser:
    """Helper to structure raw CLI events."""

    @staticmethod
    def parse_event(event: Dict) -> Optional[Dict]:
        if not isinstance(event, dict):
            return None

        etype = event.get("type")

        # 1. Handle full messages (assistant or result)
        msg_obj = None
        if etype == "assistant":
            msg_obj = event.get("message")
        elif etype == "result":
            # Safely get message from result which might be a list or dict
            res = event.get("result")
            logger.debug(f"Parsing result event: res type={type(res)}, keys={res.keys() if isinstance(res, dict) else 'N/A'}")
            if isinstance(res, dict):
                msg_obj = res.get("message")
            if not msg_obj:
                msg_obj = event.get("message")
            logger.debug(f"Result msg_obj: {msg_obj is not None}, type={type(msg_obj) if msg_obj else 'None'}")

        if msg_obj and isinstance(msg_obj, dict):
            content = msg_obj.get("content", [])
            if isinstance(content, list):
                parts = []
                thinking_parts = []
                tool_calls = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    ctype = c.get("type")
                    if ctype == "text":
                        parts.append(c.get("text", ""))
                    elif ctype == "thinking":
                        thinking_parts.append(c.get("thinking", ""))
                    elif ctype == "tool_use":
                        tool_calls.append(c)

                if tool_calls:
                    # Check for subagents (Task tool)
                    subagents = [
                        t.get("input", {}).get("description", "Subagent")
                        for t in tool_calls
                        if t.get("name") == "Task"
                    ]
                    if subagents:
                        return {"type": "subagent_start", "tasks": subagents}
                    return {"type": "tool_start", "tools": tool_calls}

                # Return combined result if we have content
                result = {}
                if thinking_parts:
                    result["thinking"] = "\n".join(thinking_parts)
                if parts:
                    result["text"] = "".join(parts)
                if result:
                    result["type"] = "content"
                    return result

        # 2. Handle streaming deltas
        if etype == "content_block_delta":
            delta = event.get("delta", {})
            if not isinstance(delta, dict):
                return None
            if delta.get("type") == "text_delta":
                return {"type": "content", "text": delta.get("text", "")}
            if delta.get("type") == "thinking_delta":
                return {"type": "thinking", "text": delta.get("thinking", "")}

        # 3. Handle tool usage start
        if etype == "content_block_start":
            block = event.get("content_block", {})
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name") == "Task":
                    desc = block.get("input", {}).get("description", "Subagent")
                    return {"type": "subagent_start", "tasks": [desc]}
                return {"type": "tool_start", "tools": [block]}

        # 4. Handle errors and exit
        if etype == "error":
            err = event.get("error")
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return {"type": "error", "message": msg}
        elif etype == "exit":
            return {
                "type": "complete",
                "status": "success" if event.get("code") == 0 else "failed",
            }
        return None


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
            session_id: Optional session ID to resume. If provided, uses --resume flag.
        
        Yields:
            Event dictionaries from the CLI, including a special 'session_info' event
            with the session ID when available.
        """
        self._is_busy = True
        env = os.environ.copy()

        # Ensure we have a dummy key if none exists, as CLI often requires it
        if "ANTHROPIC_API_KEY" not in env:
            env["ANTHROPIC_API_KEY"] = "sk-placeholder-key-for-proxy"

        env["ANTHROPIC_API_URL"] = self.api_url
        if self.api_url.endswith("/v1"):
            env["ANTHROPIC_BASE_URL"] = self.api_url[:-3]
        else:
            env["ANTHROPIC_BASE_URL"] = self.api_url

        # Force non-interactive non-TTY
        env["TERM"] = "dumb"
        env["PYTHONIOENCODING"] = "utf-8"

        # Build command based on whether resuming or starting new
        if session_id:
            # Resume existing session
            cmd = [
                "claude",
                "--resume", session_id,
                "-p", prompt,
                "--output-format", "stream-json",
                "--dangerously-skip-permissions",
                "--verbose",
            ]
            logger.info(f"Resuming Claude session {session_id} with prompt: {prompt[:50]}...")
        else:
            # Start new session
            cmd = [
                "claude",
                "-p", prompt,
                "--output-format", "stream-json",
                "--dangerously-skip-permissions",
                "--verbose",
            ]
            logger.info(f"Starting new Claude session with prompt: {prompt[:50]}...")

        if self.allowed_dirs:
            for d in self.allowed_dirs:
                cmd.extend(["--add-dir", d])

        logger.info(f"Launching Claude CLI in {self.workspace}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
                env=env,
            )

            if not self.process or not self.process.stdout:
                logger.error("Claude CLI process failed to start or stdout not captured.")
                yield {"type": "exit", "code": 1}
                return

            session_id_extracted = False

            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                    
                    # Extract session ID from various event types
                    if not session_id_extracted:
                        extracted_id = self._extract_session_id(event)
                        if extracted_id:
                            self.current_session_id = extracted_id
                            session_id_extracted = True
                            logger.info(f"Extracted session ID: {extracted_id}")
                            # Emit a special event with the session ID
                            yield {"type": "session_info", "session_id": extracted_id}
                    
                    yield event
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON output: {line_str}")
                    yield {"type": "raw", "content": line_str}

            if self.process.stderr:
                stderr_output = await self.process.stderr.read()
                if stderr_output:
                    logger.error(
                        f"Claude CLI Stderr: {stderr_output.decode('utf-8', errors='replace')}"
                    )

            return_code = await self.process.wait()
            yield {"type": "exit", "code": return_code}
        finally:
            self._is_busy = False

    def _extract_session_id(self, event: Dict) -> Optional[str]:
        """
        Extract session ID from CLI event.
        
        The session ID appears in different places depending on event type:
        - In 'init' or 'system' events as 'session_id' or 'sessionId'
        - In 'result' events under result.session_id
        - Sometimes in message metadata
        """
        if not isinstance(event, dict):
            return None
        
        # Direct session_id field
        if "session_id" in event:
            return event["session_id"]
        if "sessionId" in event:
            return event["sessionId"]
        
        # Check in nested structures
        for key in ["init", "system", "result", "metadata"]:
            if key in event and isinstance(event[key], dict):
                nested = event[key]
                if "session_id" in nested:
                    return nested["session_id"]
                if "sessionId" in nested:
                    return nested["sessionId"]
        
        # Check in conversation info
        if "conversation" in event and isinstance(event["conversation"], dict):
            conv = event["conversation"]
            if "id" in conv:
                return conv["id"]
        
        return None

    async def stop(self):
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await self.process.wait()
                return True
            except:
                return False
        return False
