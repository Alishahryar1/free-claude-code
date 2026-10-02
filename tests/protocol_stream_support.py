"""Exercise real source decoders and the sole public writer without HTTP fixtures."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from free_claude_code.core.anthropic.native_stream import NativeMessagesStreamState
from free_claude_code.core.anthropic.recovery_stream import MessagesRecoveryWriter
from free_claude_code.core.chat_observations import ChatChange
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.provider_stream import (
    responses_stream_failure_from_event,
)
from free_claude_code.core.openai_responses.recovery_stream import (
    ResponsesRecoveryWriter,
)
from free_claude_code.core.openai_responses.source_state import ResponsesSourceState
from free_claude_code.core.openai_responses.tool_adaptation import (
    ResponsesToolAdapter,
    ResponsesToolEventAdapter,
)
from free_claude_code.core.openai_responses.tools import ResponsesToolIdentity
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.stream_events import RequestOutcome, StreamEvent
from free_claude_code.core.stream_observations import (
    ChatObservation,
    DecodedStreamEvent,
    MessagesObservation,
)
from free_claude_code.providers.openai_chat.source_state import ChatSourceState


class MessagesSourceHarness:
    def __init__(
        self,
        request: OpenAIResponsesRequest | None = None,
        *,
        public_model: str,
        replay_origin: ReplayOrigin | None = None,
        tool_identities: Mapping[str, ResponsesToolIdentity] | None = None,
        permissive: bool = False,
    ) -> None:
        self.source = NativeMessagesStreamState(permissive=permissive)
        self.origin = replay_origin or ReplayOrigin(
            "test", "messages", "", "", "upstream"
        )
        self.tools = tool_identities or {}
        self.writer = (
            ResponsesRecoveryWriter(model=public_model, input_tokens=0, request=request)
            if request is not None
            else MessagesRecoveryWriter(
                model=public_model, input_tokens=0, native=replay_origin is None
            )
        )
        self.writer.begin_attempt()

    @property
    def completed(self) -> bool:
        return self.source.completed

    @property
    def started(self) -> bool:
        return self.source.started

    def feed(self, kind: str, payload: Mapping[str, JsonValue]) -> list[StreamEvent]:
        completed = self.source.accept(kind, payload)
        if (
            kind == "message_start"
            and isinstance(message := payload.get("message"), dict)
            and isinstance(model := message.get("model"), str)
        ):
            self.origin = replace(self.origin, model=model)
        result = list(
            self.writer.feed(
                DecodedStreamEvent(
                    self.origin,
                    StreamEvent(kind, dict(payload)),
                    observation=MessagesObservation(completed, self.tools),
                    stop_reason=self.source.stop_reason,
                    native_reasoning_pending=self.source.native_reasoning_pending,
                )
            )
        )
        if self.source.completed:
            result.extend(self.writer.finish())
        return result


class ResponsesSourceHarness:
    def __init__(
        self,
        *,
        public_model: str,
        messages: bool = False,
        input_tokens: int = 0,
        tool_names: OpenAIToolNameCodec | None = None,
    ) -> None:
        self.source = ResponsesSourceState()
        self.origin = ReplayOrigin("test", "responses", "", "", "upstream")
        self.writer = (
            MessagesRecoveryWriter(model=public_model, input_tokens=input_tokens)
            if messages
            else ResponsesRecoveryWriter(model=public_model, input_tokens=input_tokens)
        )
        self.tools = tool_names
        self.writer.begin_attempt()
        self.completed = False

    def start(self) -> list[StreamEvent]:
        return (
            self.writer.start_message()
            if isinstance(self.writer, MessagesRecoveryWriter)
            else []
        )

    def feed(self, kind: str, payload: dict[str, Any]) -> list[StreamEvent]:
        if self.completed:
            raise ValueError("Event arrived after terminal response.")
        if kind in {"error", "response.error", "response.failed"}:
            raise responses_stream_failure_from_event(kind, payload)
        event = StreamEvent(kind, {"type": kind, **payload})
        observation = replace(self.source.observe(event), tool_names=self.tools)
        self.completed = kind in {"response.completed", "response.incomplete"}
        result = list(
            self.writer.feed(
                DecodedStreamEvent(
                    self.origin,
                    event,
                    observation=observation,
                    outcome=RequestOutcome.SUCCESS if self.completed else None,
                )
            )
        )
        if self.completed:
            result.extend(self.writer.finish())
        return result


class ChatSourceHarness(ChatSourceState):
    def __init__(
        self,
        tool_adapter: ResponsesToolAdapter | None = None,
        *,
        input_tokens: int,
        model: str = "public",
        response_model: str | None = None,
    ) -> None:
        super().__init__(input_tokens=input_tokens)
        self.tools = tool_adapter
        self.writer = (
            ResponsesRecoveryWriter(
                model=response_model or tool_adapter.original.model,
                input_tokens=input_tokens,
                request=tool_adapter.original,
            )
            if tool_adapter is not None
            else MessagesRecoveryWriter(model=model, input_tokens=input_tokens)
        )
        self.writer.begin_attempt()

    def project(self, changes: Iterable[ChatChange]) -> list[StreamEvent]:
        result: list[StreamEvent] = []
        for change in changes:
            result.extend(
                list(
                    self.writer.feed(
                        DecodedStreamEvent(
                            self.replay_origin
                            or ReplayOrigin("test", "chat", "", "", "upstream"),
                            StreamEvent("chat.completion.chunk", {}),
                            observation=ChatObservation((change,), self.tools),
                        )
                    )
                )
            )
            if change.kind == "complete":
                result.extend(self.writer.finish())
        return result


class ToolEventHarness:
    """Expose formatter events for tests of tool restoration alone."""

    def __init__(self, *, tool_events: ResponsesToolEventAdapter | None) -> None:
        self.tools = tool_events

    def feed(self, kind: str, body: dict[str, Any]) -> list[StreamEvent]:
        if self.tools is None:
            return [StreamEvent(kind, body)]
        return [
            StreamEvent(event, payload)
            for event, payload in self.tools.feed(kind, body)
        ]
