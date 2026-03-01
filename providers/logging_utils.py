"""Logging utilities for compact, traceable request logging.

Provides fingerprinting and summary functions to reduce log file sizes
while maintaining full traceability through request IDs and content hashes.
"""

import hashlib
import json
from typing import Any

from loguru import logger

from providers.common.text import extract_text_from_content


def generate_request_fingerprint(messages: list[Any]) -> str:
    """Generate unique short hash for message content.

    Uses incremental SHA256 hashing to avoid building a large intermediate
    string. Returns an 8-char hex prefix sufficient for correlation.
    """
    h = hashlib.sha256()
    sep = b"|"
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                h.update(content.encode("utf-8"))
                h.update(sep)
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        h.update(block.text.encode("utf-8"))
                        h.update(sep)
                    elif hasattr(block, "type"):
                        h.update(f"<{block.type}>".encode())
                        h.update(sep)
        elif hasattr(msg, "role"):
            h.update(msg.role.encode("utf-8"))
            h.update(sep)
    return f"fp_{h.hexdigest()[:8]}"


def get_last_user_message_preview(messages: list[Any], max_len: int = 100) -> str:
    """Extract a preview of the last user message."""
    for msg in reversed(messages):
        if hasattr(msg, "role") and msg.role == "user":
            text = extract_text_from_content(getattr(msg, "content", ""))
            if text:
                preview = text.replace("\n", " ").replace("\r", "")
                return preview[:max_len] + "..." if len(preview) > max_len else preview
    return "(no user message)"


def get_tool_names(tools: list[Any] | None, max_count: int = 5) -> list[str]:
    """Extract tool names from tool list, limiting to max_count."""
    if not tools:
        return []
    names = []
    for tool in tools[:max_count]:
        if hasattr(tool, "name"):
            names.append(tool.name)
        elif isinstance(tool, dict) and "name" in tool:
            names.append(tool["name"])
    if len(tools) > max_count:
        names.append(f"+{len(tools) - max_count} more")
    return names


def build_request_summary(request_data: Any) -> dict[str, Any]:
    """Build compact metadata dict for logging.

    Returns a dictionary with key metrics about the request without
    including the full content.
    """
    messages = getattr(request_data, "messages", [])
    tools = getattr(request_data, "tools", None)
    system = getattr(request_data, "system", None)
    thinking = getattr(request_data, "thinking", None)

    # Count message types
    user_count = sum(1 for m in messages if getattr(m, "role", None) == "user")
    assistant_count = sum(
        1 for m in messages if getattr(m, "role", None) == "assistant"
    )

    return {
        "fingerprint": generate_request_fingerprint(messages),
        "model": getattr(request_data, "model", "unknown"),
        "message_count": len(messages),
        "user_msgs": user_count,
        "assistant_msgs": assistant_count,
        "user_preview": get_last_user_message_preview(messages),
        "tool_count": len(tools) if tools else 0,
        "tool_names": get_tool_names(tools),
        "has_thinking": bool(thinking and getattr(thinking, "enabled", False)),
        "has_system": bool(system),
        "max_tokens": getattr(request_data, "max_tokens", 0),
    }


def log_full_payload(
    logger_instance: Any, request_id: str, payload: dict[str, Any]
) -> None:
    """Log full payload to the standard logger."""
    logger_instance.debug(
        f"FULL_PAYLOAD [{request_id}]: {json.dumps(payload, default=str)}"
    )


def log_request_compact(
    logger_instance: Any,
    request_id: str,
    request_data: Any,
    prefix: str = "API_REQUEST",
) -> None:
    """Log a compact request summary with fingerprint for correlation.

    This is the main entry point for logging requests. It logs a single-line
    JSON summary to the main log and always writes full payload.
    """
    summary = build_request_summary(request_data)
    summary["request_id"] = request_id

    logger_instance.info(f"{prefix}: {json.dumps(summary)}")

    # Always log full payload
    try:
        payload = (
            request_data.model_dump() if hasattr(request_data, "model_dump") else {}
        )
        log_full_payload(logger_instance, request_id, payload)
    except Exception as e:
        logger.debug(f"Could not dump request data: {e}")
