"""CLI event parser for Claude Code CLI output.

This parser emits an ordered stream of low-level events suitable for building a
Claude Code-like transcript in messaging UIs.
"""

from typing import Dict, List, Any
from loguru import logger


def parse_cli_event(event: Any) -> List[Dict]:
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
    results: List[Dict[str, Any]] = []

    # Some CLI/proxy layers emit "system" events that are not user-visible and
    # carry no transcript content. Ignore them explicitly to avoid noisy logs.
    if etype == "system":
        return []

    # 1. Handle full messages (assistant/user or result)
    msg_obj = None
    if etype == "assistant":
        msg_obj = event.get("message")
    elif etype == "user":
        msg_obj = event.get("message")
    elif etype == "result":
        res = event.get("result")
        if isinstance(res, dict):
            msg_obj = res.get("message")
            # Some variants put content directly on the result.
            if not msg_obj and isinstance(res.get("content"), list):
                msg_obj = {"content": res.get("content")}
        if not msg_obj:
            msg_obj = event.get("message")
        # Some variants put content directly on the event.
        if not msg_obj and isinstance(event.get("content"), list):
            msg_obj = {"content": event.get("content")}

    if msg_obj and isinstance(msg_obj, dict):
        content = msg_obj.get("content", [])
        if isinstance(content, list):
            # Preserve order exactly as content blocks appear.
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "text":
                    results.append({"type": "text_chunk", "text": c.get("text", "")})
                elif ctype == "thinking":
                    results.append(
                        {"type": "thinking_chunk", "text": c.get("thinking", "")}
                    )
                elif ctype == "tool_use":
                    results.append(
                        {
                            "type": "tool_use",
                            "id": str(c.get("id", "") or "").strip(),
                            "name": c.get("name", ""),
                            "input": c.get("input"),
                        }
                    )
                elif ctype == "tool_result":
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": str(c.get("tool_use_id", "") or "").strip(),
                            "content": c.get("content"),
                            "is_error": bool(c.get("is_error", False)),
                        }
                    )

        if results:
            return results

    # 2. Handle streaming deltas
    if etype == "content_block_delta":
        delta = event.get("delta", {})
        if isinstance(delta, dict):
            if delta.get("type") == "text_delta":
                return [
                    {
                        "type": "text_delta",
                        "index": event.get("index", -1),
                        "text": delta.get("text", ""),
                    }
                ]
            if delta.get("type") == "thinking_delta":
                return [
                    {
                        "type": "thinking_delta",
                        "index": event.get("index", -1),
                        "text": delta.get("thinking", ""),
                    }
                ]
            if delta.get("type") == "input_json_delta":
                return [
                    {
                        "type": "tool_use_delta",
                        "index": event.get("index", -1),
                        "partial_json": delta.get("partial_json", ""),
                    }
                ]

    # 3. Handle tool usage start
    if etype == "content_block_start":
        block = event.get("content_block", {})
        if isinstance(block, dict):
            btype = block.get("type")
            if btype == "thinking":
                return [{"type": "thinking_start", "index": event.get("index", -1)}]
            if btype == "text":
                return [{"type": "text_start", "index": event.get("index", -1)}]
            if btype == "tool_use":
                return [
                    {
                        "type": "tool_use_start",
                        "index": event.get("index", -1),
                        "id": str(block.get("id", "") or "").strip(),
                        "name": block.get("name", ""),
                        "input": block.get("input"),
                    }
                ]

    # 3.5 Handle block stop (to close open streaming segments)
    if etype == "content_block_stop":
        return [{"type": "block_stop", "index": event.get("index", -1)}]

    # 4. Handle errors and exit
    if etype == "error":
        err = event.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        logger.info(f"CLI_PARSER: Parsed error event: {msg}")
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
            logger.warning(f"CLI_PARSER: Error exit (code={code}): {error_msg}")
            return [
                {"type": "error", "message": error_msg},
                {"type": "complete", "status": "failed"},
            ]

    # Log unrecognized events for debugging
    if etype:
        logger.debug(f"CLI_PARSER: Unrecognized event type: {etype}")
    return []
