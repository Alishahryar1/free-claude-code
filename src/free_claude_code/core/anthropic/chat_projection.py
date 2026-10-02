"""Map Chat source observations onto the request's Messages writer."""

from collections.abc import Iterator
from typing import Any

from free_claude_code.core.chat_observations import ChatChange
from free_claude_code.core.history_replay import (
    AssociatedReplayRecord,
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.stream_events import ItemCompletion, StreamEvent
from free_claude_code.core.stream_observations import ChatObservation

from .projection import MessagesWriter


class ChatMessagesProjection:
    def __init__(self, writer: MessagesWriter) -> None:
        self._writer = writer
        self._text: int | None = None
        self._thinking: int | None = None
        self._tools: dict[int, int] = {}
        self._anchors: set[str] = set()
        self._pending_replay: list[ReplayRecord | AssociatedReplayRecord] = []

    def feed(self, observation: ChatObservation) -> Iterator[StreamEvent]:
        for change in observation.changes:
            yield from self._change(change)

    def _event(
        self,
        kind: str,
        index: int,
        *,
        completion: ItemCompletion | None = None,
        replay_origin: ReplayOrigin | None = None,
        **body: Any,
    ) -> list[StreamEvent]:
        return self._writer.accept_event(
            StreamEvent(
                kind, {"type": kind, "index": index, **body}, completion, replay_origin
            )
        )

    def _start(self, kind: str, **fields: Any) -> tuple[int, list[StreamEvent]]:
        index = self._writer.allocate_block_index()
        return index, self._event(
            "content_block_start", index, content_block={"type": kind, **fields}
        )

    def _stop(self, index: int | None, *, tool: bool = False) -> list[StreamEvent]:
        return (
            []
            if index is None
            else self._event(
                "content_block_stop",
                index,
                completion=ItemCompletion.COMPLETE if tool else None,
            )
        )

    def _opaque(
        self, record: ReplayRecord | AssociatedReplayRecord
    ) -> list[StreamEvent]:
        index = self._writer.allocate_block_index()
        events = self._event(
            "content_block_start",
            index,
            content_block={"type": "redacted_thinking", "data": encode_replay(record)},
            replay_origin=record.origin,
        )
        return [*events, *self._stop(index)]

    def _signature(
        self, record: ReplayRecord | AssociatedReplayRecord
    ) -> list[StreamEvent]:
        assert self._thinking is not None
        events = self._event(
            "content_block_delta",
            self._thinking,
            delta={"type": "signature_delta", "signature": encode_replay(record)},
            replay_origin=record.origin,
        )
        events.extend(self._stop(self._thinking))
        self._thinking = None
        return events

    def _change(self, change: ChatChange) -> list[StreamEvent]:
        kind = change.kind
        if kind == "start":
            return self._writer.start_message()
        if kind in {"text.start", "reasoning.start", "reasoning.delta"}:
            reasoning = kind != "text.start"
            index = self._thinking if reasoning else self._text
            events: list[StreamEvent] = []
            if index is None:
                index, events = self._start(
                    "thinking" if reasoning else "text",
                    **({"thinking": ""} if reasoning else {"text": ""}),
                )
                if reasoning:
                    self._thinking = index
                else:
                    self._text = index
            if kind == "reasoning.delta":
                events.extend(
                    self._event(
                        "content_block_delta",
                        index,
                        delta={"type": "thinking_delta", "thinking": change.text},
                    )
                )
            return events
        if kind == "text.delta":
            events = self._change(ChatChange("text.start"))
            assert self._text is not None
            return [
                *events,
                *self._event(
                    "content_block_delta",
                    self._text,
                    delta={"type": "text_delta", "text": change.text},
                ),
            ]
        if kind in {"text.stop", "reasoning.stop"}:
            index = self._text if kind == "text.stop" else self._thinking
            if kind == "text.stop":
                self._text = None
            else:
                self._thinking = None
            return self._stop(index)
        if kind == "tool.start":
            tool = change.tool
            assert tool is not None
            fields: dict[str, Any] = {
                "id": tool.tool_id,
                "name": tool.name,
                "input": {},
            }
            if tool.extra_content:
                fields["extra_content"] = tool.extra_content
            index, events = self._start("tool_use", **fields)
            self._tools[change.tool_index] = index
            return events
        if kind == "tool.delta":
            return self._event(
                "content_block_delta",
                self._tools[change.tool_index],
                delta={"type": "input_json_delta", "partial_json": change.text},
            )
        if kind == "tool.stop":
            return self._stop(self._tools.pop(change.tool_index), tool=True)
        if kind == "reasoning_record.pause":
            record = change.record
            assert record is not None
            if change.group_id in self._anchors or (
                self._thinking is None and self._tools
            ):
                return []
            self._anchors.add(change.group_id)
            anchor = AssociatedReplayRecord(
                record.origin, record.native, change.group_id, "anchor"
            )
            if self._thinking is not None:
                return self._signature(anchor)
            events = self._stop(self._text)
            self._text = None
            return [*events, *self._opaque(anchor)]
        if kind == "reasoning_record.complete":
            record = change.record
            assert record is not None
            if change.group_id in self._anchors:
                self._anchors.remove(change.group_id)
                self._pending_replay.append(
                    AssociatedReplayRecord(
                        record.origin, record.native, change.group_id, "final"
                    )
                )
            elif self._thinking is not None:
                return self._signature(record)
            else:
                self._pending_replay.append(record)
            return []
        if kind == "replay.flush":
            pending, self._pending_replay = self._pending_replay, []
            return [event for record in pending for event in self._opaque(record)]
        if kind == "complete":
            usage = change.usage
            assert usage is not None
            self._writer.accept_event(
                StreamEvent(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": change.text, "stop_sequence": None},
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            **usage.anthropic_fields,
                        },
                    },
                )
            )
            self._writer.accept_event(
                StreamEvent("message_stop", {"type": "message_stop"})
            )
        return []
