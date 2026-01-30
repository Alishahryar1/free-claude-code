"""CLI event parser for Claude Code CLI output."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CLIParser:
    """Helper to structure raw CLI events."""

    @staticmethod
    def parse_event(event: Dict) -> List[Dict]:
        """
        Parse a CLI event and return a structured result.

        Args:
            event: Raw event dictionary from CLI

        Returns:
            List of parsed event dicts. Empty list if not recognized.
        """
        if not isinstance(event, dict):
            return []

        etype = event.get("type")
        results = []

        # 1. Handle full messages (assistant or result)
        msg_obj = None
        if etype == "assistant":
            msg_obj = event.get("message")
        elif etype == "result":
            res = event.get("result")
            if isinstance(res, dict):
                msg_obj = res.get("message")
            if not msg_obj:
                msg_obj = event.get("message")

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

                # Prioritize thinking first
                if thinking_parts:
                    results.append(
                        {"type": "thinking", "text": "\n".join(thinking_parts)}
                    )

                # Then tools or subagents
                if tool_calls:
                    # Check for subagents (Task tool)
                    subagents = [
                        t.get("input", {}).get("description", "Subagent")
                        for t in tool_calls
                        if t.get("name") == "Task"
                    ]
                    if subagents:
                        results.append({"type": "subagent_start", "tasks": subagents})
                    else:
                        results.append({"type": "tool_start", "tools": tool_calls})

                # Then text content if any
                if parts:
                    results.append({"type": "content", "text": "".join(parts)})

                if results:
                    return results

        # 2. Handle streaming deltas
        if etype == "content_block_delta":
            delta = event.get("delta", {})
            if isinstance(delta, dict):
                if delta.get("type") == "text_delta":
                    return [{"type": "content", "text": delta.get("text", "")}]
                if delta.get("type") == "thinking_delta":
                    return [{"type": "thinking", "text": delta.get("thinking", "")}]

        # 3. Handle tool usage start
        if etype == "content_block_start":
            block = event.get("content_block", {})
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name") == "Task":
                    desc = block.get("input", {}).get("description", "Subagent")
                    return [{"type": "subagent_start", "tasks": [desc]}]
                return [{"type": "tool_start", "tools": [block]}]

        # 4. Handle errors and exit
        if etype == "error":
            err = event.get("error")
            msg = err.get("message") if isinstance(err, dict) else str(err)
            logger.info(f"CLI_PARSER: Parsed error event: {msg[:100]}")
            return [{"type": "error", "message": msg}]
        elif etype == "exit":
            code = event.get("code", 0)
            stderr = event.get("stderr")
            if code == 0:
                logger.debug(f"CLI_PARSER: Successful exit (code={code})")
                return [{"type": "complete", "status": "success"}]
            else:
                # Non-zero exit is an error
                error_msg = stderr if stderr else f"Process exited with code {code}"
                logger.warning(f"CLI_PARSER: Error exit (code={code}): {error_msg[:100]}")
                return [
                    {"type": "error", "message": error_msg},
                    {"type": "complete", "status": "failed"},
                ]

        # Log unrecognized events for debugging
        if etype:
            logger.debug(f"CLI_PARSER: Unrecognized event type: {etype}")
        return []
