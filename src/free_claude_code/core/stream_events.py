"""Decoded provider events and public events, before wire serialization."""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import simplejson

from .history_replay import ReplayOrigin


class ItemCompletion(Enum):
    """Source evidence at an item boundary, independent of public event names."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID_INPUT = "invalid_input"


class RequestOutcome(Enum):
    """Authoritative source termination, independent of projected events."""

    SUCCESS = "success"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: str
    payload: dict[str, Any]
    item_completion: ItemCompletion | None = None

    def serialize(self) -> str:
        return (
            f"event: {self.kind}\n"
            f"data: {simplejson.dumps(self.payload, use_decimal=True, ensure_ascii=False)}\n\n"
        )


@dataclass(frozen=True, slots=True)
class NativeMessage:
    """A complete opaque Messages JSON response, without SSE framing."""

    payload: dict[str, Any]

    def serialize(self) -> str:
        return simplejson.dumps(self.payload, use_decimal=True, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class DecodedStreamEvent:
    """Retain the source payload alongside its lossless public projection."""

    origin: ReplayOrigin
    source: StreamEvent
    projected: tuple[StreamEvent, ...]
    progress: bool = False
    outcome: RequestOutcome | None = None
    stop_reason: str | None = None
    replay_safe: bool = True
    required_origins: tuple[ReplayOrigin, ...] = ()
    native_reasoning_pending: bool = False
    allow_empty_completion: bool = False

    @property
    def completed(self) -> bool:
        return self.outcome is not None


@dataclass(slots=True)
class BufferedStreamItem:
    atomic: bool
    events: deque[StreamEvent] = field(default_factory=deque)
    complete: bool = False


class OrderedStreamBuffer:
    """Keep later items behind a pending call and release each complete call."""

    def __init__(self) -> None:
        self._order: deque[int] = deque()
        self._items: dict[int, BufferedStreamItem] = {}

    def start(
        self, index: int, event: StreamEvent, *, atomic: bool, complete: bool = False
    ) -> None:
        if index in self._items:
            raise ValueError("Duplicate stream item index.")
        self._items[index] = BufferedStreamItem(atomic, deque((event,)), complete)
        self._order.append(index)

    @property
    def pending(self) -> bool:
        return bool(self._order)

    def contains(self, index: int) -> bool:
        return index in self._items

    def complete(self, index: int) -> bool:
        return self._items[index].complete

    def atomic(self, index: int) -> bool:
        return self._items[index].atomic

    def append(self, index: int, event: StreamEvent, *, complete: bool = False) -> None:
        item = self._items[index]
        item.events.append(event)
        item.complete = complete

    def drain(self) -> Iterator[StreamEvent]:
        while self._order:
            item = self._items[self._order[0]]
            if item.atomic and not item.complete:
                return
            while item.events:
                yield item.events.popleft()
            if not item.complete:
                return
            self._order.popleft()


class ExactPrefixFilter:
    """Remove a full replay of a committed prefix, retaining every other byte."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._pending: list[str] = []
        self._matched = 0
        self._decided = not prefix

    def feed(self, text: str) -> str:
        if self._decided:
            return text
        length = min(len(text), len(self._prefix) - self._matched)
        expected = self._prefix[self._matched : self._matched + length]
        if text[:length] != expected:
            self._decided = True
            result = "".join(self._pending) + text
            self._pending.clear()
            return result
        self._pending.append(text[:length])
        self._matched += length
        if self._matched < len(self._prefix):
            return ""
        self._decided = True
        self._pending.clear()
        return text[length:]

    def finish(self) -> str:
        # A proper prefix is not proof of replay. Only a complete exact match
        # permits dropping bytes, even when the upstream terminates early.
        result = "".join(self._pending)
        self._pending.clear()
        self._decided = True
        return result
