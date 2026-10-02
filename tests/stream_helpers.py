"""Serialize decoded projector events for existing wire-contract assertions."""

from collections.abc import Iterable

from free_claude_code.core.stream_events import StreamEvent


def serialize_events(events: Iterable[StreamEvent | str]) -> str:
    return "".join(
        event.serialize() if isinstance(event, StreamEvent) else event
        for event in events
    )
