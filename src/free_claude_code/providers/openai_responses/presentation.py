"""Client-protocol presenters for the shared Responses transport."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    NativeResponsesRelay,
    ResponsesProviderStream,
    ResponsesStreamFailure,
    ResponsesToolEventAdapter,
)


class ResponsesStreamPresenter(Protocol):
    """One client-protocol view over an upstream Responses attempt."""

    @property
    def completed(self) -> bool: ...

    @property
    def terminal_failure_completes_wire(self) -> bool: ...

    def start(self) -> Iterable[str]: ...

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[str]: ...

    def terminal_failure(
        self,
        raw_error: Exception,
        failure: ExecutionFailure,
    ) -> Iterable[str]: ...


class MessagesResponsesPresenter:
    """Translate one Responses attempt into Anthropic Messages SSE."""

    def __init__(self, stream: ResponsesProviderStream) -> None:
        self._stream = stream

    @property
    def completed(self) -> bool:
        return self._stream.completed

    @property
    def terminal_failure_completes_wire(self) -> bool:
        return False

    def start(self) -> Iterable[str]:
        return self._stream.start()

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[str]:
        return self._stream.feed(event_type, payload)

    def terminal_failure(
        self,
        raw_error: Exception,
        failure: ExecutionFailure,
    ) -> Iterable[str]:
        del raw_error, failure
        return self._stream.ledger.close_unclosed_blocks()


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

    @property
    def terminal_failure_completes_wire(self) -> bool:
        return True

    def start(self) -> Iterable[str]:
        return ()

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[str]:
        if self._tool_events is not None:
            return tuple(
                self._relay.feed(kind, value)
                for kind, value in self._tool_events.feed(event_type, payload)
            )
        return (self._relay.feed(event_type, payload),)

    def terminal_failure(
        self,
        raw_error: Exception,
        failure: ExecutionFailure,
    ) -> Iterable[str]:
        if (
            isinstance(raw_error, ResponsesStreamFailure)
            and raw_error.event_type == "response.failed"
            and raw_error.payload is not None
        ):
            return self.feed(raw_error.event_type, raw_error.payload)
        return (self._relay.synthesize_failure(failure),)


@dataclass(slots=True)
class ResponsesExecutionOutcome:
    """Provider outcome retained when terminal failure is consumed on-wire."""

    failure: Exception | None = None


type ResponsesPresenterFactory = Callable[[], ResponsesStreamPresenter]
