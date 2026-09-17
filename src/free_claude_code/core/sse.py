"""Incremental, protocol-neutral JSON SSE framing."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str
    data: dict[str, Any]
    raw: str


def parse_sse_lines(lines: Iterable[str]) -> list[SSEEvent]:
    events: list[SSEEvent] = []
    current_event = ""
    data_parts: list[str] = []
    raw_parts: list[str] = []

    for line in lines:
        stripped = line.rstrip("\r\n")
        if stripped == "":
            _append_event(events, current_event, data_parts, raw_parts)
            current_event = ""
            data_parts = []
            raw_parts = []
            continue
        raw_parts.append(stripped)
        if stripped.startswith("event:"):
            current_event = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("data:"):
            data_parts.append(stripped.split(":", 1)[1].strip())

    _append_event(events, current_event, data_parts, raw_parts)
    return events


def parse_sse_text(text: str) -> list[SSEEvent]:
    # SSE uses CR/LF framing; Unicode line separators can occur inside JSON text.
    return parse_sse_lines(re.split(r"\r\n|\r|\n", text))


def _append_event(
    events: list[SSEEvent],
    current_event: str,
    data_parts: list[str],
    raw_parts: list[str],
) -> None:
    if not current_event and not data_parts:
        return
    data_text = "\n".join(data_parts)
    data: dict[str, Any]
    try:
        parsed = json.loads(data_text) if data_text else {}
        data = parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        data = {"raw": data_text}
    events.append(SSEEvent(current_event, data, "\n".join(raw_parts)))


_EVENT_BOUNDARY = re.compile(r"(?>\r\n|\r|\n){2}")


class SSEDecoder:
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
