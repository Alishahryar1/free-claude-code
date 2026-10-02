"""One public Responses lifecycle spanning physical provider attempts."""

import json
import time
import uuid
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    preserve_responses_reasoning,
    readable_reasoning,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.recovery import (
    AttemptDispatch,
    AttemptFailure,
    RecoveryCheckpoint,
)
from free_claude_code.core.stream_events import (
    ExactPrefixFilter,
    ItemCompletion,
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
from free_claude_code.core.tool_adaptation import ResponsesToolEvents

from .chat_projection import ChatResponsesProjection
from .errors import ResponsesConversionError, openai_error_from_failure
from .ids import new_message_item_id, new_response_id
from .messages_projection import MessagesResponsesProjection
from .messages_usage import NativeMessagesUsage
from .models import OpenAIResponsesRequest
from .tool_search import is_client_search

_TERMINALS = {"response.completed", "response.incomplete"}
_TEXT_DELTAS = {"response.output_text.delta", "response.refusal.delta"}
_REASONING_DELTAS = {
    "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta",
}
_RESPONSE_FIELDS = {
    "background",
    "completed_at",
    "conversation",
    "created_at",
    "error",
    "id",
    "incomplete_details",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "moderation",
    "object",
    "output",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_diagnostics",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "status",
    "store",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "usage",
    "user",
}


def _client_call(item: dict[str, Any]) -> bool:
    return item.get("type") in {"function_call", "custom_tool_call"} or (
        item.get("type") == "tool_search_call" and is_client_search(item)
    )


@dataclass(slots=True)
class _PublishedItem:
    body: dict[str, Any]
    origin: ReplayOrigin
    parts: dict[int, dict[str, Any]] = field(default_factory=dict)
    done_parts: set[int] = field(default_factory=set)
    done_text: set[int] = field(default_factory=set)
    closed: bool = False

    def content(self) -> JsonObject:
        body = deepcopy(self.body)
        if self.parts:
            body["content"] = [
                deepcopy(self.parts[index]) for index in sorted(self.parts)
            ]
        return cast(JsonObject, body)


class ResponsesRecoveryWriter:
    """Retain public identity and committed items while replacing failed attempts."""

    def __init__(
        self,
        *,
        model: str,
        input_tokens: int,
        request: OpenAIResponsesRequest | None = None,
    ) -> None:
        self._request = request
        self._messages_projection: MessagesResponsesProjection | None = None
        self._chat_projection: ChatResponsesProjection | None = None
        self.messages_usage = NativeMessagesUsage()
        self._model = model
        self._input_tokens = input_tokens
        self._response: dict[str, Any] | None = None
        self._items: dict[int, _PublishedItem] = {}
        self._used_ids: set[str] = set()
        self._next_index = 0
        self._next_sequence = 0
        self._revision = 0
        self._recovered = False
        self._completed = False
        self._blocked_reason: str | None = None
        self._uncertain_operation = False
        self._native_reasoning_pending = False
        self._explicit_stop = False
        self._unpublished_tail = False
        self._opaque_events = False
        self._terminal: StreamEvent | None = None
        self._buffer = OrderedStreamBuffer()
        self._indexes: dict[int, int] = {}
        self._ids: dict[str, str] = {}
        self._origin: ReplayOrigin | None = None
        self._prefix = ExactPrefixFilter("")
        self._last_text: tuple[int, int] | None = None
        self._required_origins: set[ReplayOrigin] = set()
        self._attempt_origins: set[ReplayOrigin] = set()
        self._tool_events: ResponsesToolEvents | None = None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def started(self) -> bool:
        return self._response is not None

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
        for item in self._items.values():
            body = item.content()
            kind = body.get("type")
            if kind == "message":
                if not any(
                    part.get("text") or part.get("refusal") or part.get("annotations")
                    for part in body.get("content", [])
                    if isinstance(part, dict)
                ):
                    continue
                text.extend(
                    str(part.get("text", ""))
                    for part in body.get("content", [])
                    if isinstance(part, dict)
                )
            elif _client_call(body):
                tools = True
            elif kind == "reasoning":
                encrypted = body.get("encrypted_content")
                if encrypted:
                    required_origins.append(item.origin)
                if not item.closed and item.origin.protocol != "chat":
                    blocked = (
                        "Interrupted native reasoning has no complete replay state."
                    )
            else:
                blocked = (
                    "Committed native output has no supported continuation contract."
                )
            content.append(body)
        return RecoveryCheckpoint(
            "responses",
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
        self._ids = {}
        self._terminal = None
        self._origin = None
        self._last_text = None
        self._prefix = ExactPrefixFilter(self.checkpoint.text)
        self._recovered = self._recovered or self.started
        self._unpublished_tail = False
        self._opaque_events = False
        self._tool_events = None
        self._messages_projection = None
        self._chat_projection = None
        self.messages_usage = NativeMessagesUsage()

    @property
    def allow_empty_completion(self) -> bool:
        return self._explicit_stop or (
            self._terminal is not None and self._terminal.kind == "response.incomplete"
        )

    def reject_attempt(self) -> None:
        self._required_origins = self._attempt_origins.copy()
        self._uncertain_operation = False

    def dispatch(self, evidence: AttemptDispatch) -> None:
        self._required_origins.update(evidence.required_origins)
        if not evidence.replay_safe:
            self._uncertain_operation = True

    def feed(self, event: DecodedStreamEvent) -> Iterator[StreamEvent]:
        try:
            yield from self._feed(event)
        except ResponsesConversionError as error:
            raise AttemptFailure(
                ExecutionFailure(FailureKind.UPSTREAM, 502, str(error), False)
            ) from error

    def _feed(self, event: DecodedStreamEvent) -> Iterator[StreamEvent]:
        self._origin = event.origin
        self._native_reasoning_pending = event.native_reasoning_pending
        self._explicit_stop = self._explicit_stop or event.allow_empty_completion
        self._required_origins.update(event.required_origins)
        if not event.replay_safe:
            self._uncertain_operation = True
        if isinstance(observation := event.observation, ResponsesObservation):
            if observation.tools is not None and self._tool_events is None:
                self._tool_events = observation.tools.event_adapter()
            for observed in observation.events:
                payload = preserve_responses_reasoning(observed.payload, event.origin)
                changes = (
                    self._tool_events.feed(observed.kind, payload)
                    if self._tool_events is not None
                    else ((observed.kind, payload),)
                )
                for kind, data in changes:
                    yield from self._accept(
                        StreamEvent(kind, data, observed.item_completion)
                    )
            for snapshot in observation.snapshots:
                body = preserve_responses_reasoning(
                    {"item": snapshot.body}, event.origin
                )["item"]
                if observation.tools is not None:
                    body = observation.tools.restore_item(body)
                if isinstance(body, dict):
                    self._refresh_item(snapshot.index, body)
            return
        if isinstance(event.observation, ChatObservation):
            if self._chat_projection is None:
                self._chat_projection = ChatResponsesProjection(
                    self,
                    event.observation.tools.event_adapter()
                    if event.observation.tools is not None
                    else None,
                )
            yield from self._chat_projection.feed(event.observation)
            return
        if isinstance(event.observation, MessagesObservation):
            if self._messages_projection is None:
                self._messages_projection = MessagesResponsesProjection(self)
            try:
                yield from self._messages_projection.feed(event, event.observation)
            except NativeMessagesError as error:
                raise AttemptFailure(
                    ExecutionFailure(FailureKind.UPSTREAM, 502, str(error), True),
                    retry_allowed=True,
                ) from error

    def allocate_output_index(self) -> int:
        index = self._next_index
        self._next_index += 1
        self._indexes[index] = index
        return index

    def accept_event(self, event: StreamEvent) -> list[StreamEvent]:
        return self._accept(event)

    def start_response(self) -> list[StreamEvent]:
        if self._response is not None:
            return []
        request = self._request
        response: dict[str, Any] = {
            "id": new_response_id(),
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": self._model,
            "output": [],
            "error": None,
            "incomplete_details": None,
            "usage": None,
            "tools": request.tools or [] if request else [],
            "tool_choice": request.tool_choice or "auto" if request else "auto",
            "parallel_tool_calls": request.parallel_tool_calls
            if request and request.parallel_tool_calls is not None
            else True,
        }
        if request is not None:
            for key in (
                "instructions",
                "max_output_tokens",
                "temperature",
                "top_p",
                "metadata",
                "reasoning",
            ):
                value = getattr(request, key)
                if value is not None:
                    response[key] = value
        return self._accept(
            StreamEvent(
                "response.created", {"type": "response.created", "response": response}
            )
        )

    def complete_response(self, *, incomplete: bool, usage: JsonObject | None) -> None:
        kind = "response.incomplete" if incomplete else "response.completed"
        self._terminal = StreamEvent(
            kind,
            {
                "type": kind,
                "response": {
                    **(self._response or {}),
                    "status": "incomplete" if incomplete else "completed",
                    "usage": usage,
                    "incomplete_details": {"reason": "max_output_tokens"}
                    if incomplete
                    else None,
                },
            },
        )

    def _refresh_item(self, source_index: int, body: dict[str, Any]) -> None:
        """Update a published snapshot without emitting its lifecycle a second time."""
        index = self._indexes.get(source_index)
        if index is None:
            return
        item = self._items[index]
        body = deepcopy(body)
        body["id"] = item.body["id"]
        if body.get("type") == "message":
            for part_index, part in enumerate(body.get("content", [])):
                if isinstance(part, dict):
                    key = "refusal" if part.get("type") == "refusal" else "text"
                    item.parts[part_index] = {
                        **part,
                        key: item.parts.get(part_index, {}).get(key, ""),
                    }
        if body.get("type") == "message":
            body["content"] = []
        item.body.update(body)

    def _accept(self, event: StreamEvent) -> list[StreamEvent]:
        kind, payload = event.kind, event.payload
        if self._completed:
            raise ValueError("Responses event arrived after logical completion.")
        if kind in {"response.failed", "response.error", "error"}:
            raise ValueError(
                "Attempt failures must be raised before public serialization."
            )
        if kind in _TERMINALS:
            self._terminal = deepcopy(event)
            return []
        if kind in {"response.created", "response.in_progress", "response.queued"}:
            if set(payload["response"]) - _RESPONSE_FIELDS:
                self._blocked_reason = (
                    "Unknown native response state has no continuation contract."
                )
            if self._recovered:
                return []
            if self._response is None:
                self._response = deepcopy(payload["response"])
                self._response["model"] = self._model
            return [self._event(event)]
        if kind == "response.output_item.added":
            self._buffer.start(
                payload["output_index"], event, atomic=_client_call(payload["item"])
            )
        elif kind == "response.output_item.done" and not self._buffer.contains(
            payload["output_index"]
        ):
            self._buffer.start(
                payload["output_index"],
                event,
                atomic=_client_call(payload["item"]),
                complete=not _client_call(payload["item"])
                or event.item_completion is ItemCompletion.COMPLETE,
            )
        elif "output_index" in payload:
            index = payload["output_index"]
            if not self._buffer.contains(index):
                if self._items:
                    raise AttemptFailure(
                        ExecutionFailure(
                            FailureKind.UPSTREAM,
                            502,
                            "Provider output cannot be joined to the committed item lifecycle.",
                            True,
                        ),
                        retry_allowed=True,
                    )
                # The native relay historically preserves sparse or extended
                # lifecycles. Keep their successful output intact, but do not
                # claim that such a checkpoint can be reconstructed elsewhere.
                self._opaque_events = True
                self._blocked_reason = (
                    "Native output has no complete item lifecycle for continuation."
                )
                return [self._event(event)]
            if self._buffer.complete(index):
                self._blocked_reason = (
                    "A native event after item completion has source-owned references."
                )
                return self._publish(event)
            self._buffer.append(
                payload["output_index"],
                event,
                complete=kind == "response.output_item.done"
                and (
                    not self._buffer.atomic(index)
                    or event.item_completion is ItemCompletion.COMPLETE
                ),
            )
        else:
            if kind not in {"ping"}:
                self._blocked_reason = (
                    "An unknown native event may contain source-owned references."
                )
            return [self._event(event)]
        return [out for queued in self._buffer.drain() for out in self._publish(queued)]

    def _publish(self, event: StreamEvent, *, emit: bool = True) -> list[StreamEvent]:
        assert self._origin is not None
        kind, original = event.kind, event.payload
        payload = deepcopy(original)
        if set(payload) - {
            "type",
            "sequence_number",
            "response_id",
            "output_index",
            "item_id",
            "content_index",
            "summary_index",
            "item",
            "part",
            "delta",
            "text",
            "refusal",
            "arguments",
            "input",
            "logprobs",
            "annotation",
            "annotation_index",
        }:
            self._blocked_reason = (
                "An unknown native event field may contain source-owned references."
            )
        upstream_index = payload["output_index"]
        if kind == "response.output_item.added" or upstream_index not in self._indexes:
            index = self._indexes.get(
                upstream_index,
                upstream_index if not self._recovered else self._next_index,
            )
            self._next_index = max(self._next_index, index + 1)
            self._indexes[upstream_index] = index
            body = payload["item"]
            old_id = body.get("id")
            if isinstance(old_id, str):
                new_id = (
                    old_id
                    if old_id not in self._used_ids
                    else f"{old_id}_{uuid.uuid4().hex}"
                )
                self._used_ids.add(new_id)
                self._ids[old_id] = body["id"] = new_id
            item = _PublishedItem(deepcopy(body), event.replay_origin or self._origin)
            self._items[index] = item
            if body.get("type") == "message":
                for part_index, part in enumerate(body.get("content", [])):
                    body["content"][part_index] = self._public_part(
                        item, index, part_index, part
                    )
        else:
            index = self._indexes[upstream_index]
            item = self._items[index]
        if event.replay_origin is not None:
            item.origin = event.replay_origin
        payload["output_index"] = index
        if isinstance(payload.get("item_id"), str):
            payload["item_id"] = self._ids.get(payload["item_id"], payload["item_id"])
        part_index = payload.get("content_index", 0)
        if kind == "response.content_part.added":
            payload["part"] = self._public_part(
                item, index, part_index, payload["part"]
            )
        elif kind in _TEXT_DELTAS:
            key = "refusal" if kind == "response.refusal.delta" else "text"
            raw = payload["delta"]
            value = raw if key == "refusal" else self._prefix.feed(raw)
            if key == "text":
                self._last_text = (index, part_index)
            if not value:
                return []
            payload["delta"] = value
            part = item.parts.setdefault(
                part_index,
                {"type": "refusal", "refusal": ""}
                if key == "refusal"
                else {"type": "output_text", "text": "", "annotations": []},
            )
            part[key] = part.get(key, "") + value
            self._revision += 1
        elif kind in _REASONING_DELTAS:
            self._revision += bool(payload.get("delta"))
            # Preserve readable Chat reasoning even when no final item arrived.
            key = "summary" if "summary" in kind else "content"
            parts = item.body.setdefault(key, [])
            if not parts:
                parts.append(
                    {
                        "type": "summary_text"
                        if key == "summary"
                        else "reasoning_text",
                        "text": "",
                    }
                )
            parts[-1]["text"] += payload["delta"]
        elif kind in {"response.output_text.done", "response.refusal.done"}:
            item.done_text.add(part_index)
            key = "refusal" if kind == "response.refusal.done" else "text"
            part = {
                **item.parts.get(
                    part_index,
                    {"type": "refusal" if key == "refusal" else "output_text"},
                ),
                key: payload[key],
            }
            payload[key] = self._public_part(item, index, part_index, part)[key]
        elif kind == "response.content_part.done":
            item.done_parts.add(part_index)
            payload["part"] = self._public_part(
                item, index, part_index, payload["part"]
            )
        elif kind == "response.output_item.done":
            body = payload["item"]
            old_id = body.get("id")
            if isinstance(old_id, str):
                body["id"] = self._ids.get(old_id, old_id)
            if body.get("type") == "message":
                for part_index, part in enumerate(body.get("content", [])):
                    body["content"][part_index] = self._public_part(
                        item, index, part_index, part
                    )
            item.body = deepcopy(body)
            if body.get("type") == "message":
                item.body["content"] = []
            item.closed = True
            if _client_call(body):
                self._revision += 1
        elif kind not in {
            "response.output_item.added",
            "response.reasoning_text.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
        }:
            self._blocked_reason = (
                "A native event has no supported continuation contract."
            )
        body = item.body
        fields = {
            "message": {"type", "id", "role", "status", "content", "phase"},
            "reasoning": {
                "type",
                "id",
                "status",
                "content",
                "summary",
                "encrypted_content",
            },
            "function_call": {
                "type",
                "id",
                "status",
                "call_id",
                "name",
                "namespace",
                "arguments",
            },
            "custom_tool_call": {
                "type",
                "id",
                "status",
                "call_id",
                "name",
                "namespace",
                "input",
            },
            "tool_search_call": {
                "type",
                "id",
                "status",
                "call_id",
                "execution",
                "arguments",
            },
        }.get(body.get("type"))
        if fields is not None and set(body) - fields:
            self._blocked_reason = (
                "An unknown native item field has no continuation contract."
            )
        return [self._event(StreamEvent(kind, payload))] if emit else []

    def _public_part(
        self, item: _PublishedItem, index: int, part_index: int, part: dict[str, Any]
    ) -> dict[str, Any]:
        """Keep source metadata beside the text this writer actually published."""
        part = deepcopy(part)
        kind = part.get("type")
        if kind not in {"output_text", "refusal"} or set(part) - {
            "type",
            "text",
            "refusal",
            "annotations",
            "logprobs",
        }:
            self._blocked_reason = (
                "Native content has no supported continuation contract."
            )
        key = "refusal" if kind == "refusal" else "text"
        part[key] = item.parts.get(part_index, {}).get(key, "")
        item.parts[part_index] = deepcopy(part)
        if key == "text":
            self._last_text = (index, part_index)
        return part

    def _event(self, event: StreamEvent) -> StreamEvent:
        payload = deepcopy(event.payload)
        sequence = payload.get("sequence_number")
        if not self._recovered and isinstance(sequence, int):
            self._next_sequence = max(self._next_sequence, sequence)
        payload["sequence_number"] = self._next_sequence
        self._next_sequence += 1
        if self._response is not None:
            if "response_id" in payload:
                payload["response_id"] = self._response["id"]
            if isinstance(response := payload.get("response"), dict):
                response["id"] = self._response["id"]
                response["model"] = self._model
                if self._recovered:
                    response["created_at"] = self._response.get("created_at")
        if isinstance(response := payload.get("response"), dict):
            response["model"] = self._model
            for field_name in ("created_at", "completed_at"):
                timestamp = response.get(field_name)
                if isinstance(timestamp, float) and timestamp.is_integer():
                    response[field_name] = int(timestamp)
        return StreamEvent(event.kind, payload)

    def interrupt(self) -> list[StreamEvent]:
        self._unpublished_tail = self._unpublished_tail or self._buffer.pending
        result: list[StreamEvent] = []
        held = self._prefix.finish()
        if held and self._last_text is not None:
            index, part_index = self._last_text
            item = self._items[index]
            if not item.closed and part_index not in item.done_text:
                part = item.parts.setdefault(
                    part_index, {"type": "output_text", "text": "", "annotations": []}
                )
                part["text"] = part.get("text", "") + held
                self._revision += 1
                result.append(
                    self._event(
                        StreamEvent(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": index,
                                "item_id": item.body["id"],
                                "content_index": part_index,
                                "delta": held,
                            },
                        )
                    )
                )
            else:
                result.extend(self._append_held_text(held))
        if self.checkpoint.blocked_reason is not None:
            return result
        for index, item in self._items.items():
            if item.closed:
                continue
            item_id = item.body["id"]
            for part_index, part in item.parts.items():
                if part_index not in item.done_text:
                    key = "refusal" if part.get("type") == "refusal" else "text"
                    kind = (
                        "response.refusal.done"
                        if key == "refusal"
                        else "response.output_text.done"
                    )
                    result.append(
                        self._event(
                            StreamEvent(
                                kind,
                                {
                                    "type": kind,
                                    "output_index": index,
                                    "item_id": item_id,
                                    "content_index": part_index,
                                    key: part.get(key, ""),
                                },
                            )
                        )
                    )
                if part_index not in item.done_parts:
                    result.append(
                        self._event(
                            StreamEvent(
                                "response.content_part.done",
                                {
                                    "type": "response.content_part.done",
                                    "output_index": index,
                                    "item_id": item_id,
                                    "content_index": part_index,
                                    "part": part,
                                },
                            )
                        )
                    )
            item.body["status"] = "completed"
            item.closed = True
            result.append(
                self._event(
                    StreamEvent(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": index,
                            "item": item.content(),
                        },
                    )
                )
            )
        self._buffer = OrderedStreamBuffer()
        return result

    def _append_held_text(self, text: str) -> list[StreamEvent]:
        assert self._origin is not None
        index = self._next_index
        self._next_index += 1
        item_id = new_message_item_id()
        body = {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        part = {"type": "output_text", "text": text, "annotations": []}
        self._items[index] = _PublishedItem(body, self._origin, {0: part})
        self._revision += 1
        return [
            self._event(StreamEvent(kind, {"type": kind, **payload}))
            for kind, payload in (
                ("response.output_item.added", {"output_index": index, "item": body}),
                (
                    "response.content_part.added",
                    {
                        "output_index": index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": {**part, "text": ""},
                    },
                ),
                (
                    "response.output_text.delta",
                    {
                        "output_index": index,
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": text,
                    },
                ),
            )
        ]

    def finish(self, *, salvage: bool = False) -> list[StreamEvent]:
        if self._completed:
            raise ValueError("Responses response already completed.")
        result = self.interrupt()
        if salvage:
            response = {
                **(self._response or {}),
                "status": "completed",
                "error": None,
                "incomplete_details": None,
            }
            terminal = StreamEvent(
                "response.completed",
                {"type": "response.completed", "response": response},
            )
        else:
            if self._terminal is None:
                raise ValueError(
                    "Responses attempt completed without its terminal payload."
                )
            terminal = deepcopy(self._terminal)
        if not self._opaque_events or self._unpublished_tail:
            response = terminal.payload["response"]
            response["output"] = [item.content() for item in self._items.values()]
            if self._recovered or salvage:
                response["usage"] = self._logical_usage()
        result.append(self._event(terminal))
        self._completed = True
        return result

    def failure(self, failure: ExecutionFailure) -> list[StreamEvent]:
        response = {
            **(self._response or {"id": new_response_id(), "object": "response"}),
            "status": "failed",
            "model": self._model,
            "error": openai_error_from_failure(failure),
            "output": [item.content() for item in self._items.values()],
        }
        self._completed = True
        return [
            self._event(
                StreamEvent(
                    "response.failed", {"type": "response.failed", "response": response}
                )
            )
        ]

    def _logical_usage(self) -> dict[str, int]:
        output = 0
        for item in self._items.values():
            body = item.content()
            if _client_call(body):
                arguments = body.get("arguments", body.get("input", ""))
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                output += (
                    estimate_text_tokens(str(body.get("name", "")))
                    + estimate_text_tokens(arguments)
                    + 19
                )
            else:
                text = (
                    "".join(text for text, _ in readable_reasoning(body))
                    if body.get("type") == "reasoning"
                    else "".join(
                        str(part.get("text", part.get("refusal", "")))
                        for part in body.get("content", [])
                        if isinstance(part, dict)
                    )
                )
                if text:
                    output += estimate_text_tokens(text) + 4
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": output,
            "total_tokens": self._input_tokens + output,
        }
