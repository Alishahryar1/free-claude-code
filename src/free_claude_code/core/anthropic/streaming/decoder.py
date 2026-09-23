"""Incremental framing for Anthropic-compatible SSE streams."""

import re

from ..stream_contracts import SSEEvent, parse_sse_text

_EVENT_BOUNDARY = re.compile(r"(?>\r\n|\r|\n){2}")


class AnthropicSSEDecoder:
    """Decode arbitrarily split SSE text without losing frame order."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._boundary_tail = ""

    def feed(self, chunk: str) -> tuple[SSEEvent, ...]:
        """Consume one wire chunk and return every complete event."""

        events: list[SSEEvent] = []
        probe = self._boundary_tail + chunk
        prefix_length = len(self._boundary_tail)
        chunk_start = 0
        for match in _EVENT_BOUNDARY.finditer(probe):
            chunk_end = match.end() - prefix_length
            self._parts.append(chunk[chunk_start:chunk_end])
            raw = "".join(self._parts)
            self._parts.clear()
            events.extend(parse_sse_text(raw))
            chunk_start = chunk_end

        remainder = chunk[chunk_start:]
        if remainder:
            self._parts.append(remainder)
        if chunk_start:
            self._boundary_tail = remainder[-3:]
        else:
            self._boundary_tail = (self._boundary_tail + chunk)[-3:]
        return tuple(events)

    def finish(self) -> tuple[SSEEvent, ...]:
        """Return a final unterminated event, if one is present."""

        remainder = "".join(self._parts)
        self._parts.clear()
        self._boundary_tail = ""
        if not remainder.strip():
            return ()
        return tuple(parse_sse_text(remainder))
