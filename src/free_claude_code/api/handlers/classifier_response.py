"""Hide structured provider reasoning from Claude's classifier response."""

import sys
from collections.abc import AsyncIterator

from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.trace import close_stream_input

from ..classifier_projection import ClassifierProjection


async def classifier_response(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Apply the same projection to private, nonstreaming aggregation."""
    decoder = AnthropicSSEDecoder()
    projection = ClassifierProjection()
    try:
        async for chunk in source:
            for frame in decoder.feed_frames(chunk):
                projected = projection.feed(frame)
                if projected is not None:
                    yield projected
        for frame in decoder.finish_frames():
            projected = projection.feed(frame)
            if projected is not None:
                yield projected
    finally:
        await close_stream_input(
            source,
            owner="classifier_response",
            source="api",
            preserved_error=sys.exception(),
        )
