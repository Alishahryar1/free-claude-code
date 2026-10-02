"""Client-protocol presenters for the shared Responses transport."""

from collections.abc import Callable, Iterable
from typing import Protocol

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    NativeResponsesRelay,
    ResponsesProviderStream,
    ResponsesToolEventAdapter,
)
from free_claude_code.core.stream_events import StreamEvent


class ResponsesStreamPresenter(Protocol):
    """One client-protocol view over an upstream Responses attempt."""

    @property
    def completed(self) -> bool: ...

    def start(self) -> Iterable[StreamEvent]: ...

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[StreamEvent]: ...


class MessagesResponsesPresenter:
    """Translate one Responses attempt into Anthropic Messages SSE."""

    def __init__(self, stream: ResponsesProviderStream) -> None:
        self._stream = stream

    @property
    def completed(self) -> bool:
        return self._stream.completed

    def start(self) -> Iterable[StreamEvent]:
        return self._stream.start()

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[StreamEvent]:
        return self._stream.feed(event_type, payload)


class NativeResponsesPresenter:
    """Relay one Responses attempt as native Responses SSE."""

    def __init__(
        self, *, public_model: str, tool_events: ResponsesToolEventAdapter | None = None
    ) -> None:
        self._relay = NativeResponsesRelay(public_model=public_model)
        self._tool_events = tool_events

    @property
    def completed(self) -> bool:
        return self._relay.completed

    def start(self) -> Iterable[StreamEvent]:
        return ()

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[StreamEvent]:
        if self._tool_events is not None:
            return tuple(
                self._relay.feed(kind, value)
                for kind, value in self._tool_events.feed(event_type, payload)
            )
        return (self._relay.feed(event_type, payload),)


type ResponsesPresenterFactory = Callable[[], ResponsesStreamPresenter]
