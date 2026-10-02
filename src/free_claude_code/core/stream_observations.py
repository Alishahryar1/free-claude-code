"""Source-protocol observations passed to the request's public response owner."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .chat_observations import ChatChange
from .history_replay import ReplayOrigin
from .stream_events import RequestOutcome, StreamEvent
from .tool_adaptation import ResponsesToolIdentity, ResponsesToolMapping

if TYPE_CHECKING:
    from .anthropic.native_stream import CompletedMessagesBlock
    from .openai_tool_names import OpenAIToolNameCodec


@dataclass(frozen=True, slots=True)
class ResponsesSnapshot:
    """Authoritative metadata for an item whose completion was already observed."""

    index: int
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResponsesObservation:
    """Reconciled source events and metadata updates, without a public lifecycle."""

    events: tuple[StreamEvent, ...] = ()
    snapshots: tuple[ResponsesSnapshot, ...] = ()
    tools: ResponsesToolMapping | None = None
    tool_names: OpenAIToolNameCodec | None = None


@dataclass(frozen=True, slots=True)
class MessagesObservation:
    """A Messages source boundary and its request-specific tool identities."""

    completed: CompletedMessagesBlock | None = None
    tool_identities: Mapping[str, ResponsesToolIdentity] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChatObservation:
    changes: tuple[ChatChange, ...]
    tools: ResponsesToolMapping | None = None


@dataclass(frozen=True, slots=True)
class DecodedStreamEvent:
    """Retain source payload, completion facts and protocol-specific observations."""

    origin: ReplayOrigin
    source: StreamEvent
    progress: bool = False
    outcome: RequestOutcome | None = None
    stop_reason: str | None = None
    replay_safe: bool = True
    required_origins: tuple[ReplayOrigin, ...] = ()
    native_reasoning_pending: bool = False
    allow_empty_completion: bool = False
    observation: ChatObservation | MessagesObservation | ResponsesObservation | None = (
        None
    )

    @property
    def completed(self) -> bool:
        return self.outcome is not None
