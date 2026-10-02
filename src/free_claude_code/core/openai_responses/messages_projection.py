"""Project decoded Messages blocks into the request's Responses writer."""

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any

from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.stream_events import StreamEvent
from free_claude_code.core.stream_observations import (
    DecodedStreamEvent,
    MessagesObservation,
)

from .ids import new_message_item_id, new_reasoning_item_id, tool_item_id_prefix
from .items import message_item, reasoning_item
from .projection import ResponsesWriter
from .streaming.event_builders import ResponseEventBuilder


@dataclass(frozen=True, slots=True)
class _Binding:
    index: int
    item: dict[str, Any]


class MessagesResponsesProjection:
    """Keep only source-to-public bindings, with no content or response ledger."""

    def __init__(self, writer: ResponsesWriter) -> None:
        self._writer = writer
        self._bindings: dict[int, _Binding] = {}
        self._events = ResponseEventBuilder()

    def feed(
        self, event: DecodedStreamEvent, observation: MessagesObservation
    ) -> list[StreamEvent]:
        kind, body = event.source.kind, event.source.payload
        if kind == "ping":
            return []
        if kind == "message_start":
            self._writer.messages_usage.update(body["message"].get("usage"))
            return self._writer.start_response()
        if kind == "message_delta":
            self._writer.messages_usage.update(
                body.get("usage"),
                final=body.get("delta", {}).get("stop_reason") is not None,
            )
            return []
        if kind == "message_stop":
            if event.stop_reason not in {
                "end_turn",
                "stop_sequence",
                "tool_use",
                "max_tokens",
                "refusal",
            }:
                raise NativeMessagesError(
                    f"Responses cannot represent native stop reason {event.stop_reason!r}."
                )
            self._writer.complete_response(
                incomplete=event.stop_reason == "max_tokens",
                usage=self._writer.messages_usage.payload(require_final=True),
            )
            return []
        source_index = body["index"]
        if kind == "content_block_start":
            changes = self._start(source_index, body["content_block"], observation)
        elif kind == "content_block_delta":
            changes = self._delta(source_index, body["delta"])
        else:
            changes = self._finish(source_index, observation, event.origin)
        return [
            output for change in changes for output in self._writer.accept_event(change)
        ]

    def _start(
        self, source_index: int, block: dict[str, Any], observation: MessagesObservation
    ) -> list[StreamEvent]:
        index = self._writer.allocate_output_index()
        kind = block["type"]
        if kind == "text":
            if set(block) - {"type", "text", "citations"} or block.get("citations"):
                raise NativeMessagesError(
                    "Responses cannot represent native text extensions."
                )
            item = message_item(new_message_item_id(), "", "in_progress")
        elif kind in {"thinking", "redacted_thinking"}:
            item = reasoning_item(new_reasoning_item_id(), "", "in_progress")
        elif kind == "tool_use":
            if set(block) - {"type", "id", "name", "input", "caller"}:
                raise NativeMessagesError(
                    "Responses cannot represent native tool extensions."
                )
            if block.get("caller") not in (None, {"type": "direct"}):
                raise NativeMessagesError(
                    "Responses cannot represent a server-managed tool caller."
                )
            identity = observation.tool_identities.get(block["name"])
            if identity is None:
                raise NativeMessagesError("Native output used an unknown tool name.")
            custom = identity.kind == "custom"
            item = {
                "id": f"{tool_item_id_prefix(identity.kind)}{uuid.uuid4().hex}",
                "type": "custom_tool_call" if custom else "function_call",
                "status": "in_progress",
                "call_id": block["id"],
                "name": identity.name,
                "input" if custom else "arguments": "",
            }
            if identity.namespace:
                item["namespace"] = identity.namespace
        else:
            raise NativeMessagesError(
                f"Responses cannot represent native block type {kind!r}."
            )
        self._bindings[source_index] = _Binding(index, item)
        changes = [self._events.output_item_added(index, item)]
        if kind == "text":
            changes.append(self._events.content_part_added(item["id"], index))
        text = block.get("text" if kind == "text" else "thinking", "")
        if text:
            changes.append(
                (
                    self._events.output_text_delta
                    if kind == "text"
                    else self._events.reasoning_text_delta
                )(item["id"], index, text)
            )
        return changes

    def _delta(self, source_index: int, delta: dict[str, Any]) -> list[StreamEvent]:
        binding = self._bindings[source_index]
        kind = delta["type"]
        if kind == "text_delta":
            return [
                self._events.output_text_delta(
                    binding.item["id"], binding.index, delta["text"]
                )
            ]
        if kind == "thinking_delta":
            return [
                self._events.reasoning_text_delta(
                    binding.item["id"], binding.index, delta["thinking"]
                )
            ]
        if kind in {"signature_delta", "input_json_delta"}:
            return []
        raise NativeMessagesError(
            "Responses cannot represent the native content delta."
        )

    def _finish(
        self, source_index: int, observation: MessagesObservation, origin: ReplayOrigin
    ) -> list[StreamEvent]:
        completed = observation.completed
        assert completed is not None
        binding = self._bindings.pop(source_index)
        item_id, index = binding.item["id"], binding.index
        body = completed.body
        changes: list[StreamEvent] = []
        if body["type"] == "text":
            text = str(body["text"])
            item = message_item(item_id, text, "completed")
            changes.extend(
                (
                    self._events.output_text_done(item_id, index, text),
                    self._events.content_part_done(item_id, index, text),
                )
            )
        elif body["type"] in {"thinking", "redacted_thinking"}:
            text = str(body.get("thinking", ""))
            item = reasoning_item(item_id, text, "completed")
            item["encrypted_content"] = encode_replay(ReplayRecord(origin, body))
            if not text:
                item.pop("content")
            else:
                changes.append(self._events.reasoning_text_done(item_id, index, text))
        else:
            arguments = completed.tool_arguments
            assert arguments is not None
            custom = binding.item["type"] == "custom_tool_call"
            if custom:
                try:
                    wrapper = json.loads(arguments)
                except (ValueError, RecursionError) as error:
                    raise NativeMessagesError(
                        "Invalid native custom tool wrapper."
                    ) from error
                if (
                    not isinstance(wrapper, dict)
                    or set(wrapper) != {"input"}
                    or not isinstance(wrapper["input"], str)
                ):
                    raise NativeMessagesError(
                        "Native custom tool input must contain exactly one text input."
                    )
                arguments = wrapper["input"]
            item = {
                **binding.item,
                "status": "completed",
                "input" if custom else "arguments": arguments,
            }
            if arguments:
                changes.append(
                    (
                        self._events.custom_tool_call_input_delta
                        if custom
                        else self._events.function_call_arguments_delta
                    )(item_id, index, arguments)
                )
            changes.append(
                (
                    self._events.custom_tool_call_input_done
                    if custom
                    else self._events.function_call_arguments_done
                )(item_id, index, arguments)
            )
        changes.append(self._events.output_item_done(index, item))
        return [
            replace(change, item_completion=completed.completion) for change in changes
        ]
