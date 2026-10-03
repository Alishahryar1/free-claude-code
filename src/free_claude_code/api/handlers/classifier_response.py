"""Hide structured provider reasoning from Claude's classifier response."""

import sys
from collections.abc import AsyncIterator

from free_claude_code.core.anthropic.stream_contracts import SSEEvent
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.core.trace import close_stream_input


async def classifier_response(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Project block lifecycles without interpreting any classifier text."""
    decoder = AnthropicSSEDecoder()
    indices: dict[int, int | None] = {}
    next_index = 0
    delivery = current_stream_delivery()
    revision = delivery.attempt_revision if delivery else 0

    def synchronize() -> None:
        nonlocal decoder, next_index, revision
        if delivery is None or revision == delivery.attempt_revision:
            return
        revision = delivery.attempt_revision
        decoder = AnthropicSSEDecoder()
        indices.clear()
        next_index = 0

    def project(event: SSEEvent) -> str | None:
        nonlocal next_index
        payload = event.data
        kind = payload.get("type", event.event)
        if kind not in {
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        }:
            return event.raw + "\n\n"
        index = payload.get("index")
        if not isinstance(index, int):
            return event.raw + "\n\n"
        if kind == "content_block_start":
            block = payload.get("content_block")
            if isinstance(block, dict) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                indices[index] = None
            else:
                indices[index] = next_index
                next_index += 1
        if index not in indices:
            return event.raw + "\n\n"
        visible_index = indices[index]
        if kind == "content_block_stop":
            del indices[index]
        if visible_index is None:
            return None
        return format_sse_event(event.event, {**payload, "index": visible_index})

    try:
        async for chunk in source:
            synchronize()
            for event in decoder.feed(chunk):
                projected = project(event)
                if projected is not None:
                    yield projected
        synchronize()
        for event in decoder.finish():
            projected = project(event)
            if projected is not None:
                yield projected
    finally:
        synchronize()
        await close_stream_input(
            source,
            owner="classifier_response",
            source="api",
            preserved_error=sys.exception(),
        )
