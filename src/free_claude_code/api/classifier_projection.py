"""Project classifier content within one public attempt."""

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event


class ClassifierProjection:
    """Project one attempt's blocks without interpreting classifier text."""

    def __init__(self) -> None:
        self._indices: dict[int, int | None] = {}
        self._next_index = 0

    def feed(self, frame: str) -> str | None:
        events = parse_sse_text(frame)
        if not events:
            return frame
        event = events[0]
        payload = event.data
        kind = payload.get("type", event.event)
        if kind not in {
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        }:
            return frame
        index = payload.get("index")
        if not isinstance(index, int):
            return frame
        if kind == "content_block_start":
            block = payload.get("content_block")
            if isinstance(block, dict) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                self._indices[index] = None
            else:
                self._indices[index] = self._next_index
                self._next_index += 1
        if index not in self._indices:
            return frame
        visible_index = self._indices[index]
        if kind == "content_block_stop":
            del self._indices[index]
        if visible_index is None:
            return None
        return format_sse_event(event.event, {**payload, "index": visible_index})
