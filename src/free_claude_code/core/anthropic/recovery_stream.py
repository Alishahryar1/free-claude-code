"""One public Messages lifecycle spanning physical provider attempts."""

import json
import uuid
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.recovery import AttemptDispatch, RecoveryCheckpoint
from free_claude_code.core.stream_events import (
    ExactPrefixFilter,
    ItemCompletion,
    NativeMessage,
    OrderedStreamBuffer,
    StreamEvent,
)
from free_claude_code.core.stream_observations import (
    ChatObservation,
    DecodedStreamEvent,
    MessagesObservation,
    ResponsesObservation,
)
from free_claude_code.core.token_estimation import estimate_text_tokens

from .chat_projection import ChatMessagesProjection
from .errors import anthropic_failure_payload
from .native_projection import native_messages_events
from .responses_projection import ResponsesMessagesProjection
from .usage import anthropic_input_usage_fields


@dataclass(slots=True)
class _PublishedBlock:
    body: dict[str, Any]
    origin: ReplayOrigin
    parts: dict[str, list[str]] = field(default_factory=dict)
    closed: bool = False

    def content(self) -> JsonObject:
        body = deepcopy(self.body)
        for key, parts in self.parts.items():
            if key != "partial_json":
                body[key] = str(body.get(key, "")) + "".join(parts)
        return cast(JsonObject, body)


class MessagesRecoveryWriter:
    """Publish committed blocks and keep unfinished client calls private."""

    def __init__(self, *, model: str, input_tokens: int, native: bool = False) -> None:
        self._model = model
        self._input_tokens = input_tokens
        self._native = native
        self._started = False
        self._completed = False
        self._blocks: dict[int, _PublishedBlock] = {}
        self._next_index = 0
        self._revision = 0
        self._recovered = False
        self._blocked_reason: str | None = None
        self._uncertain_operation = False
        self._native_reasoning_pending = False
        self._explicit_stop = False
        self._terminal: list[StreamEvent] = []
        self._buffer = OrderedStreamBuffer()
        self._indexes: dict[int, int] = {}
        self._prefix = ExactPrefixFilter("")
        self._last_text_index: int | None = None
        self._origin: ReplayOrigin | None = None
        self._required_origins: set[ReplayOrigin] = set()
        self._attempt_origins: set[ReplayOrigin] = set()
        self._responses_projection: ResponsesMessagesProjection | None = None
        self._chat_projection: ChatMessagesProjection | None = None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def started(self) -> bool:
        return self._started

    @property
    def checkpoint(self) -> RecoveryCheckpoint:
        content: list[JsonObject] = []
        text: list[str] = []
        required_origins: list[ReplayOrigin] = list(self._required_origins)
        blocked = (
            "A provider-owned operation may already have run."
            if self._uncertain_operation
            else self._blocked_reason
        )
        tools = False
        if self._native_reasoning_pending:
            blocked = "Native reasoning is still awaiting its final replay metadata."
        for block in self._blocks.values():
            body = block.content()
            kind = body.get("type")
            if kind == "text":
                if not body.get("text") and not body.get("citations"):
                    continue
                text.append(str(body.get("text", "")))
            elif kind == "tool_use":
                tools = True
            elif kind in {"thinking", "redacted_thinking"}:
                key = "signature" if kind == "thinking" else "data"
                value = body.get(key)
                if value:
                    required_origins.append(block.origin)
                elif block.origin.protocol != "chat":
                    blocked = (
                        "Interrupted native reasoning has no complete replay state."
                    )
                if not block.closed and block.origin.protocol != "chat":
                    blocked = "Interrupted native reasoning cannot be closed by another attempt."
            else:
                blocked = (
                    "Committed native content has no supported continuation contract."
                )
            content.append(body)
        return RecoveryCheckpoint(
            "messages",
            tuple(content),
            "".join(text),
            self._revision,
            tools,
            tuple(dict.fromkeys(required_origins)),
            blocked,
        )

    def begin_attempt(self) -> None:
        self._attempt_origins = self._required_origins.copy()
        self._explicit_stop = False
        self._buffer = OrderedStreamBuffer()
        self._indexes = {}
        self._terminal = []
        self._last_text_index = None
        self._origin = None
        self._prefix = ExactPrefixFilter(self.checkpoint.text)
        self._recovered = self._recovered or self._started
        self._responses_projection = None
        self._chat_projection = None

    @property
    def allow_empty_completion(self) -> bool:
        return self._explicit_stop or any(
            event.kind == "message_delta"
            and event.payload.get("delta", {}).get("stop_reason")
            in {"max_tokens", "pause_turn", "refusal", "stop_sequence"}
            for event in self._terminal
        )

    def reject_attempt(self) -> None:
        self._required_origins = self._attempt_origins.copy()
        self._uncertain_operation = False

    def dispatch(self, evidence: AttemptDispatch) -> None:
        self._required_origins.update(evidence.required_origins)
        if not evidence.replay_safe:
            self._uncertain_operation = True

    def feed(self, event: DecodedStreamEvent) -> Iterator[StreamEvent]:
        self._origin = event.origin
        self._native_reasoning_pending = event.native_reasoning_pending
        self._explicit_stop = self._explicit_stop or event.allow_empty_completion
        self._required_origins.update(event.required_origins)
        if not event.replay_safe:
            self._uncertain_operation = True
        if isinstance(event.observation, ChatObservation):
            if self._chat_projection is None:
                self._chat_projection = ChatMessagesProjection(self)
            yield from self._chat_projection.feed(event.observation)
            return
        if isinstance(event.observation, ResponsesObservation):
            if self._responses_projection is None:
                self._responses_projection = ResponsesMessagesProjection(
                    self, event.observation.tool_names
                )
            yield from self._responses_projection.feed(event.observation, event.origin)
            return
        if isinstance(event.observation, MessagesObservation):
            for observed in native_messages_events(
                event, event.observation, opaque=self._native
            ):
                yield from self._accept(observed)

    def allocate_block_index(self) -> int:
        index = self._next_index
        self._next_index += 1
        self._indexes[index] = index
        return index

    def accept_event(self, event: StreamEvent) -> list[StreamEvent]:
        return self._accept(event)

    def start_message(self) -> list[StreamEvent]:
        if self._started:
            return []
        return self._accept(
            StreamEvent(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid.uuid4()}",
                        "type": "message",
                        "role": "assistant",
                        "model": self._model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": self._input_tokens,
                            "output_tokens": 1,
                        },
                    },
                },
            )
        )

    def responses_terminal(self, response: dict[str, Any], *, incomplete: bool) -> None:
        usage = response.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        input_tokens = usage.get("input_tokens", self._input_tokens)
        output_tokens = usage.get(
            "output_tokens", self._logical_usage()["output_tokens"]
        )
        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            input_tokens = self._input_tokens
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            output_tokens = self._logical_usage()["output_tokens"]
        reason = (
            "max_tokens"
            if incomplete
            else "tool_use"
            if self.checkpoint.published_tools
            else "end_turn"
        )
        self._terminal = [
            StreamEvent(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": reason, "stop_sequence": None},
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        **anthropic_input_usage_fields(
                            usage.get("input_tokens"),
                            cache_read_tokens=details.get("cached_tokens"),
                            cache_creation_tokens=details.get("cache_write_tokens"),
                        ),
                    },
                },
            ),
            StreamEvent("message_stop", {"type": "message_stop"}),
        ]

    def _accept(self, event: StreamEvent) -> list[StreamEvent]:
        kind, payload = event.kind, event.payload
        if set(payload) - {
            "type",
            "message",
            "index",
            "content_block",
            "delta",
            "usage",
        }:
            self._blocked_reason = (
                "An unknown native event field may contain source-owned references."
            )
        if self._completed:
            raise ValueError("Messages event arrived after logical completion.")
        if kind == "message_start":
            if set(payload["message"]) - {
                "id",
                "type",
                "role",
                "content",
                "model",
                "stop_reason",
                "stop_sequence",
                "usage",
                "container",
                "context_management",
            }:
                self._blocked_reason = (
                    "Unknown native message state has no continuation contract."
                )
            if self._started:
                return []
            self._started = True
            message = {**payload["message"], "model": self._model}
            return [StreamEvent(kind, {**payload, "message": message})]
        if kind in {"message_delta", "message_stop"}:
            # Only the logical owner may publish the terminal state.
            self._terminal.append(event)
            return []
        if kind == "error":
            raise ValueError(
                "Attempt failures must be raised before public serialization."
            )
        if kind not in {
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        }:
            if kind != "ping":
                self._blocked_reason = (
                    "An unknown native event may contain source-owned references."
                )
            return [event]
        index = payload["index"]
        if kind == "content_block_start":
            block = payload["content_block"]
            fields = {
                "text": {"type", "text", "citations"},
                "thinking": {"type", "thinking", "signature"},
                "redacted_thinking": {"type", "data"},
                "tool_use": {"type", "id", "name", "input", "caller"},
            }.get(block.get("type"))
            if fields is not None and set(block) - fields:
                self._blocked_reason = (
                    "An unknown native block field has no continuation contract."
                )
            self._buffer.start(index, event, atomic=block.get("type") == "tool_use")
        else:
            self._buffer.append(
                index,
                event,
                complete=kind == "content_block_stop"
                and (
                    not self._buffer.atomic(index)
                    or event.item_completion is ItemCompletion.COMPLETE
                ),
            )
        return [out for queued in self._buffer.drain() for out in self._publish(queued)]

    def _publish(self, event: StreamEvent) -> list[StreamEvent]:
        assert self._origin is not None
        kind, payload = event.kind, event.payload
        upstream_index = payload["index"]
        if kind == "content_block_start":
            index = self._indexes.get(
                upstream_index,
                upstream_index if not self._recovered else self._next_index,
            )
            self._next_index = max(self._next_index, index + 1)
            self._indexes[upstream_index] = index
            body = deepcopy(payload["content_block"])
            if body.get("type") == "text":
                body["text"] = self._prefix.feed(body.get("text", ""))
                self._last_text_index = index
            block = _PublishedBlock(body, event.replay_origin or self._origin)
            self._blocks[index] = block
            if any(body.get(key) for key in ("text", "thinking", "data")):
                self._revision += 1
            return [
                StreamEvent(kind, {**payload, "index": index, "content_block": body})
            ]
        index = self._indexes[upstream_index]
        block = self._blocks[index]
        if event.replay_origin is not None:
            block.origin = event.replay_origin
        if kind == "content_block_delta":
            delta = dict(payload["delta"])
            field = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "signature",
                "input_json_delta": "partial_json",
            }.get(delta.get("type"))
            if field is not None:
                value = delta[field]
                if field == "text":
                    self._last_text_index = index
                    value = delta[field] = self._prefix.feed(value)
                    if not value:
                        return []
                block.parts.setdefault(field, []).append(value)
                if value:
                    self._revision += 1
            elif delta.get("type") == "citations_delta":
                block.body["citations"] = [
                    *(block.body.get("citations") or []),
                    deepcopy(delta["citation"]),
                ]
                self._revision += 1
            else:
                self._blocked_reason = (
                    "An unknown native delta cannot be reconstructed safely."
                )
            return [StreamEvent(kind, {**payload, "index": index, "delta": delta})]
        block.closed = True
        if block.body.get("type") == "tool_use":
            self._revision += 1
        return [StreamEvent(kind, {**payload, "index": index})]

    def interrupt(self) -> list[StreamEvent]:
        """Close only portable public blocks; discard the unpublished tail."""
        result = self._flush_prefix()
        if self.checkpoint.blocked_reason is not None:
            return result
        for index, block in self._blocks.items():
            if not block.closed:
                block.closed = True
                result.append(
                    StreamEvent(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": index},
                    )
                )
        self._buffer = OrderedStreamBuffer()
        return result

    def _flush_prefix(self) -> list[StreamEvent]:
        text = self._prefix.finish()
        if not text or self._last_text_index is None:
            return []
        index = self._last_text_index
        block = self._blocks[index]
        if block.closed:
            # A full block can finish while prefix matching still spans blocks.
            # Allocate a new plain-text block for the unmatched held suffix.
            index = self._next_index
            self._next_index += 1
            assert self._origin is not None
            block = _PublishedBlock({"type": "text", "text": ""}, self._origin)
            self._blocks[index] = block
            start = [
                StreamEvent(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": block.body,
                    },
                )
            ]
        else:
            start = []
        block.parts.setdefault("text", []).append(text)
        self._revision += 1
        return [
            *start,
            StreamEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
        ]

    def finish(self, *, salvage: bool = False) -> list[StreamEvent]:
        if self._completed:
            raise ValueError("Messages response already completed.")
        result = self.interrupt()
        if salvage:
            terminal = [
                StreamEvent(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    },
                ),
                StreamEvent("message_stop", {"type": "message_stop"}),
            ]
        else:
            terminal = self._terminal
        for event in terminal:
            if event.kind == "message_delta" and (self._recovered or salvage):
                event = StreamEvent(
                    event.kind, {**event.payload, "usage": self._logical_usage()}
                )
            result.append(event)
        self._completed = True
        return result

    def failure(self, failure: ExecutionFailure) -> list[StreamEvent]:
        self._completed = True
        return [StreamEvent("error", anthropic_failure_payload(failure))]

    def _logical_usage(self) -> dict[str, int]:
        output = 0
        for block in self._blocks.values():
            body = block.content()
            kind = body.get("type")
            if kind in {"text", "thinking"}:
                text = str(body.get("text" if kind == "text" else "thinking", ""))
                if text:
                    output += estimate_text_tokens(text) + 4
            elif kind == "tool_use":
                arguments = "".join(block.parts.get("partial_json", [])) or json.dumps(
                    body.get("input", {}), ensure_ascii=False
                )
                output += (
                    estimate_text_tokens(str(body.get("name", "")))
                    + estimate_text_tokens(arguments)
                    + 19
                )
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": output,
        }


class NativeMessagesCompletionWriter:
    """Retain an opaque nonstreaming Message without normalizing native fields."""

    def __init__(self, *, model: str) -> None:
        self._model = model
        self._message: dict[str, Any] | None = None
        self._blocked_reason: str | None = None
        self._required_origins: set[ReplayOrigin] = set()
        self._attempt_origins: set[ReplayOrigin] = set()

    @property
    def revision(self) -> int:
        return int(self._message is not None)

    @property
    def started(self) -> bool:
        return False

    @property
    def checkpoint(self) -> RecoveryCheckpoint:
        return RecoveryCheckpoint(
            "messages",
            required_origins=tuple(self._required_origins),
            blocked_reason=self._blocked_reason,
        )

    def begin_attempt(self) -> None:
        self._attempt_origins = self._required_origins.copy()
        self._explicit_stop = False
        self._message = None

    @property
    def allow_empty_completion(self) -> bool:
        return (
            self._message is not None
            and self._message.get("stop_reason") == "max_tokens"
        )

    def reject_attempt(self) -> None:
        self._required_origins = self._attempt_origins.copy()
        self._blocked_reason = None

    def dispatch(self, evidence: AttemptDispatch) -> None:
        self._required_origins.update(evidence.required_origins)
        if not evidence.replay_safe:
            self._blocked_reason = "A provider-owned operation may already have run."

    def feed(self, event: DecodedStreamEvent) -> list[StreamEvent]:
        self._required_origins.update(event.required_origins)
        if not event.replay_safe:
            self._blocked_reason = "A provider-owned operation may already have run."
        if event.completed:
            self._message = {**event.source.payload, "model": self._model}
        return []

    def interrupt(self) -> list[StreamEvent]:
        return []

    def finish(self, *, salvage: bool = False) -> list[NativeMessage]:
        assert not salvage and self._message is not None
        return [NativeMessage(self._message)]

    def failure(self, failure: ExecutionFailure) -> list[StreamEvent]:
        raise failure
