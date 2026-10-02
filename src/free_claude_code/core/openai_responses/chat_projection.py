"""Map Chat source observations onto the request's Responses writer."""

import uuid
from collections.abc import Iterator
from dataclasses import replace

from free_claude_code.core.chat_observations import ChatChange, responses_usage
from free_claude_code.core.history_replay import encode_replay
from free_claude_code.core.stream_events import ItemCompletion, StreamEvent
from free_claude_code.core.stream_observations import ChatObservation
from free_claude_code.core.tool_adaptation import ResponsesToolEvents

from .ids import new_message_item_id, new_reasoning_item_id
from .items import message_item, reasoning_item
from .projection import ResponsesWriter
from .streaming.event_builders import ResponseEventBuilder


class ChatResponsesProjection:
    def __init__(
        self, writer: ResponsesWriter, tools: ResponsesToolEvents | None
    ) -> None:
        self._writer = writer
        self._events = ResponseEventBuilder()
        self._tool_events = tools
        self._text: tuple[int, str] | None = None
        self._thinking: tuple[int, str] | None = None
        self._tools: dict[int, tuple[int, str]] = {}

    def feed(self, observation: ChatObservation) -> Iterator[StreamEvent]:
        for change in observation.changes:
            if change.kind == "start":
                yield from (self._writer.start_response())
                continue
            if change.kind == "complete":
                assert change.usage is not None
                self._writer.complete_response(
                    incomplete=change.text in {"length", "max_tokens"},
                    usage=responses_usage(change.usage),
                )
                continue
            for event in self._change(change):
                changes = (
                    self._tool_events.feed(event.kind, event.payload)
                    if self._tool_events is not None
                    else ((event.kind, event.payload),)
                )
                for kind, body in changes:
                    yield from (
                        self._writer.accept_event(
                            StreamEvent(
                                kind, body, event.item_completion, event.replay_origin
                            )
                        )
                    )

    def _start(self, *, reasoning: bool) -> list[StreamEvent]:
        if self._thinking is not None if reasoning else self._text is not None:
            return []
        index = self._writer.allocate_output_index()
        item_id = new_reasoning_item_id() if reasoning else new_message_item_id()
        if reasoning:
            self._thinking = (index, item_id)
            return [
                self._events.output_item_added(
                    index, reasoning_item(item_id, "", "in_progress")
                )
            ]
        self._text = (index, item_id)
        return [
            self._events.output_item_added(
                index, {**message_item(item_id, "", "in_progress"), "content": []}
            ),
            self._events.content_part_added(item_id, index),
        ]

    def _change(self, change: ChatChange) -> list[StreamEvent]:
        kind = change.kind
        if kind == "text.start":
            return self._start(reasoning=False)
        if kind in {"reasoning.start", "reasoning_record.start"}:
            return self._start(reasoning=True)
        if kind in {"text.delta", "reasoning.delta"}:
            reasoning = kind == "reasoning.delta"
            events = self._start(reasoning=reasoning)
            binding = self._thinking if reasoning else self._text
            assert binding is not None
            index, item_id = binding
            events.append(
                (
                    self._events.reasoning_text_delta
                    if reasoning
                    else self._events.output_text_delta
                )(item_id, index, change.text)
            )
            return events
        if kind == "text.stop":
            binding, self._text = self._text, None
            if binding is None:
                return []
            index, item_id = binding
            return [
                self._events.output_text_done(item_id, index, change.text),
                self._events.content_part_done(item_id, index, change.text),
                self._events.output_item_done(
                    index, message_item(item_id, change.text, "completed")
                ),
            ]
        if kind in {"reasoning.stop", "reasoning_record.complete"}:
            events = (
                self._start(reasoning=True)
                if self._thinking is None and change.record is not None
                else []
            )
            binding, self._thinking = self._thinking, None
            if binding is None:
                return events
            index, item_id = binding
            item = reasoning_item(item_id, change.text, "completed")
            if change.record is not None:
                item["encrypted_content"] = encode_replay(change.record)
                if not change.text:
                    item.pop("content")
            if change.text:
                events.append(
                    self._events.reasoning_text_done(item_id, index, change.text)
                )
            events.append(
                replace(
                    self._events.output_item_done(index, item),
                    replay_origin=change.record.origin
                    if change.record is not None
                    else None,
                )
            )
            return events
        if kind == "tool.start":
            tool = change.tool
            assert tool is not None
            index, item_id = (
                self._writer.allocate_output_index(),
                f"fc_{uuid.uuid4().hex}",
            )
            self._tools[change.tool_index] = (index, item_id)
            return [
                self._events.output_item_added(
                    index,
                    {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": tool.tool_id,
                        "name": tool.name,
                        "arguments": "",
                    },
                )
            ]
        if kind == "tool.stop":
            tool = change.tool
            assert tool is not None
            index, item_id = self._tools.pop(change.tool_index)
            events = []
            if tool.arguments:
                events.append(
                    self._events.function_call_arguments_delta(
                        item_id, index, tool.arguments
                    )
                )
            events.extend(
                (
                    self._events.function_call_arguments_done(
                        item_id, index, tool.arguments
                    ),
                    self._events.output_item_done(
                        index,
                        {
                            "id": item_id,
                            "type": "function_call",
                            "status": "completed",
                            "call_id": tool.tool_id,
                            "name": tool.name,
                            "arguments": tool.arguments,
                        },
                    ),
                )
            )
            return [
                StreamEvent(event.kind, event.payload, ItemCompletion.COMPLETE)
                for event in events
            ]
        return []
