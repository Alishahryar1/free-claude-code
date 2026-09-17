"""Neutral SSE parsing and Anthropic stream shape assertions.

Used by default CI contract tests and by opt-in live smoke scenarios.
"""

from free_claude_code.core.sse import SSEEvent

from .server_tool_types import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
)

# Content blocks that only use content_block_start/stop (no deltas), including
# Anthropic server tools and eager text emitted in a single start event.
_NO_DELTA_BLOCK_KINDS = frozenset(
    {
        SERVER_TOOL_USE,
        WEB_SEARCH_TOOL_RESULT,
        WEB_FETCH_TOOL_RESULT,
        "text_eager",
        "redacted_thinking",
    }
)

_ALLOWED_BLOCK_START_TYPES = frozenset(
    {
        "text",
        "thinking",
        "tool_use",
        "redacted_thinking",
        SERVER_TOOL_USE,
        WEB_SEARCH_TOOL_RESULT,
        WEB_FETCH_TOOL_RESULT,
    }
)


def assert_anthropic_stream_contract(
    events: list[SSEEvent], *, allow_error: bool = False
) -> None:
    """Check minimal Anthropic-style SSE invariants and block nesting.

    Does *not* assert strict event ordering (e.g. :class:`message_delta` vs
    content blocks). Successful streams end in ``message_stop``; when explicitly
    allowed, failed streams may instead end in a protocol-native ``error``.
    """
    assert events, "stream produced no SSE events"
    event_names = [event.event for event in events]
    assert "message_start" in event_names, event_names
    allowed_terminal_events = (
        {"message_stop", "error"} if allow_error else {"message_stop"}
    )
    assert event_names[-1] in allowed_terminal_events, event_names

    open_blocks: dict[int, str] = {}
    seen_blocks: set[int] = set()
    for event in events:
        if event.event == "error" and not allow_error:
            raise AssertionError(f"unexpected SSE error event: {event.data}")

        if event.event == "content_block_start":
            index = event_index(event)
            block = event.data.get("content_block", {})
            assert isinstance(block, dict), event.data
            block_type = str(block.get("type", ""))
            assert block_type in _ALLOWED_BLOCK_START_TYPES, event.data
            assert index not in open_blocks, f"block {index} started twice"
            assert index not in seen_blocks, f"block {index} reused after stop"
            if block_type == "text" and str(block.get("text", "")).strip():
                storage = "text_eager"
            else:
                storage = block_type
            open_blocks[index] = storage
            seen_blocks.add(index)
            continue

        if event.event == "content_block_delta":
            index = event_index(event)
            assert index in open_blocks, f"delta for unopened block {index}"
            kind = open_blocks[index]
            assert kind not in _NO_DELTA_BLOCK_KINDS, (
                f"unexpected delta for start/stop-only block {kind} at index {index}"
            )
            delta = event.data.get("delta", {})
            assert isinstance(delta, dict), event.data
            delta_type = str(delta.get("type", ""))
            if kind == "thinking":
                assert delta_type in (
                    "thinking_delta",
                    "signature_delta",
                ), f"block {index} is {kind}, got {delta_type}"
                continue
            expected = {
                "text": "text_delta",
                "tool_use": "input_json_delta",
            }[kind]
            assert delta_type == expected, f"block {index} is {kind}, got {delta_type}"
            continue

        if event.event == "content_block_stop":
            index = event_index(event)
            assert index in open_blocks, f"stop for unopened block {index}"
            open_blocks.pop(index)

    assert not open_blocks, f"unclosed blocks: {open_blocks}"
    assert seen_blocks, "stream did not emit any content blocks"


def event_names(events: list[SSEEvent]) -> list[str]:
    return [event.event for event in events]


def text_content(events: list[SSEEvent]) -> str:
    parts: list[str] = []
    for event in events:
        if event.event == "content_block_start":
            block = event.data.get("content_block", {})
            if isinstance(block, dict) and block.get("type") == "text":
                eager = str(block.get("text", ""))
                if eager:
                    parts.append(eager)
        delta = event.data.get("delta", {})
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            parts.append(str(delta.get("text", "")))
    return "".join(parts)


def thinking_content(events: list[SSEEvent]) -> str:
    parts: list[str] = []
    for event in events:
        delta = event.data.get("delta", {})
        if isinstance(delta, dict) and delta.get("type") == "thinking_delta":
            parts.append(str(delta.get("thinking", "")))
    return "".join(parts)


def has_tool_use(events: list[SSEEvent]) -> bool:
    for event in events:
        block = event.data.get("content_block", {})
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return True
    return False


def event_index(event: SSEEvent) -> int:
    """Return the content block ``index`` field from an SSE payload (strict)."""
    value = event.data.get("index")
    assert isinstance(value, int), event.data
    return value
