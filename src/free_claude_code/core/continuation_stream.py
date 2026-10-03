"""Map continuation events into an existing public response lifecycle."""

from copy import deepcopy
from typing import Any
from uuid import uuid4, uuid5

from .delivered_response import DeliveredResponse


class ContinuationStream:
    """Own only recovery-time mappings; ordinary events remain untouched."""

    def __init__(self, delivered: DeliveredResponse) -> None:
        self.delivered = delivered
        self.active = False
        self.handoff = False
        self._offset = 0
        self._namespace = uuid4()
        self._boundary: list[dict[str, Any]] = []
        self._expected = ""
        self._candidate = ""
        self._candidate_event: dict[str, Any] | None = None
        self._prefix_pending = False
        self._initial_progress = 0
        self._prefix_items: list[dict[str, Any]] = []
        self._trimmed = 0
        self._text_target: tuple[int, int] | None = None

    @property
    def made_progress(self) -> bool:
        pending_visible_text = (
            self._text_target is not None
            and self._text_target[0] in self.delivered.items
            and bool(self._candidate[self._overlap_size() :])
        )
        return self.delivered.progress > self._initial_progress or pending_visible_text

    def begin(self, *, handoff: bool) -> None:
        self.active = True
        self.handoff = handoff
        self._offset = max(self.delivered.items, default=-1) + 1
        self._namespace = uuid4()
        self._boundary = self.delivered.closing_events()
        self._expected = self.delivered.snapshot().text
        self._candidate = ""
        self._candidate_event = None
        self._prefix_pending = bool(self._expected)
        self._initial_progress = self.delivered.progress
        self._trimmed = 0
        self._text_target = None
        finalized = deepcopy(self.delivered.items)
        for event in self._boundary:
            if event["type"] == "response.output_item.done":
                finalized[event["output_index"]] = deepcopy(event["item"])
        self._prefix_items = [item for _, item in sorted(finalized.items())]

    def boundary(self) -> list[dict[str, Any]]:
        events, self._boundary = self._boundary, []
        return [self._sequence(event) for event in events]

    def _sequence(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.delivered.wire_api == "responses":
            self.delivered.sequence += 1
            event["sequence_number"] = self.delivered.sequence
        return event

    def _item_id(self, value: str) -> str:
        return f"{value.split('_')[0]}_{uuid5(self._namespace, value).hex}"

    def _trim_item(self, item: dict[str, Any], index: int) -> None:
        if self._text_target is None or self._text_target[0] != index:
            return
        position = self._text_target[1]
        parts = item.get("content", [])
        if position < len(parts):
            parts[position]["text"] = parts[position].get("text", "")[self._trimmed :]

    def prepare(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.active:
            return [payload]
        data = deepcopy(payload)
        kind = data.get("type", "")
        if kind in {"message_start", "response.created", "response.in_progress"}:
            return []
        if not self.handoff:
            if isinstance(data.get("index"), int):
                data["index"] += self._offset
            if isinstance(data.get("output_index"), int):
                data["output_index"] += self._offset
            if isinstance(data.get("item_id"), str):
                data["item_id"] = self._item_id(data["item_id"])
            item = data.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                item["id"] = self._item_id(item["id"])

        if "response_id" in data:
            data["response_id"] = self.delivered.header.get("id", data["response_id"])

        if self._is_text_delta(data):
            return self._text(data)

        if (
            kind == "content_block_start"
            and data.get("content_block", {}).get("type") == "text"
            and data["content_block"].get("text")
        ):
            text = data["content_block"]["text"]
            data["content_block"]["text"] = ""
            return [
                self._sequence(data),
                *self._text(
                    {
                        "type": "content_block_delta",
                        "index": data["index"],
                        "delta": {"type": "text_delta", "text": text},
                    }
                ),
            ]

        before: list[dict[str, Any]] = []
        terminal = kind in {
            "message_stop",
            "error",
            "response.completed",
            "response.incomplete",
            "response.failed",
        } or (
            kind == "message_delta"
            and data.get("delta", {}).get("stop_reason") is not None
        )
        target = self._text_target
        text_closed = target is not None and (
            (kind == "content_block_stop" and data.get("index") == target[0])
            or (
                kind == "response.output_item.done"
                and data.get("output_index") == target[0]
            )
            or (
                kind in {"response.output_text.done", "response.content_part.done"}
                and (data.get("output_index"), data.get("content_index", 0)) == target
            )
        )
        if self._prefix_pending and (terminal or text_closed):
            before = self._flush_candidate()

        if self.delivered.wire_api == "messages":
            if kind == "message_delta" and "usage" in data:
                data["usage"] = {"output_tokens": self.delivered.output_tokens()}
        else:
            index = data.get("output_index")
            target_part = self._text_target == (index, data.get("content_index", 0))
            if kind == "response.output_text.done" and target_part:
                data["text"] = data.get("text", "")[self._trimmed :]
            if (
                kind == "response.content_part.done"
                and data.get("part", {}).get("type") == "output_text"
                and target_part
            ):
                data["part"]["text"] = data["part"].get("text", "")[self._trimmed :]
            if kind == "response.output_item.done" and isinstance(index, int):
                self._trim_item(data["item"], index)
            response = data.get("response")
            if isinstance(response, dict):
                response["id"] = self.delivered.header.get("id", response.get("id"))
                if "created_at" in self.delivered.header:
                    response["created_at"] = self.delivered.header["created_at"]
                if self.delivered.unsafe_reason == "retention_limit":
                    if self.handoff:
                        response["output"] = deepcopy(self._prefix_items)
                    else:
                        for position, item in enumerate(response.get("output", [])):
                            if isinstance(item.get("id"), str):
                                item["id"] = self._item_id(item["id"])
                            self._trim_item(item, position + self._offset)
                        response["output"] = [
                            *deepcopy(self._prefix_items),
                            *response.get("output", []),
                        ]
                else:
                    response["output"] = [
                        deepcopy(item)
                        for _, item in sorted(self.delivered.items.items())
                    ]
                response["usage"] = self.delivered.usage(response.get("output"))
        return [*before, self._sequence(data)]

    @staticmethod
    def _is_text_delta(data: dict[str, Any]) -> bool:
        return data.get("type") == "response.output_text.delta" or (
            data.get("type") == "content_block_delta"
            and data.get("delta", {}).get("type") == "text_delta"
        )

    @staticmethod
    def _get_text(data: dict[str, Any]) -> str:
        return (
            data["delta"]
            if isinstance(data.get("delta"), str)
            else data["delta"]["text"]
        )

    @staticmethod
    def _set_text(data: dict[str, Any], text: str) -> None:
        if isinstance(data.get("delta"), str):
            data["delta"] = text
        else:
            data["delta"]["text"] = text

    def _text(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        text = self._get_text(data)
        if not text:
            return []
        if self._prefix_pending:
            self._candidate += text
            self._candidate_event = data
            if self._text_target is None:
                self._text_target = (
                    data.get("output_index", data.get("index", 0)),
                    data.get("content_index", 0),
                )
            # A substring may still grow into a longer suffix match. Wait until
            # it diverges or this text part ends before choosing the longest one.
            if self._candidate in self._expected:
                return []
            self._trimmed = self._overlap_size()
            text = self._candidate[self._trimmed :]
            self._candidate = ""
            self._candidate_event = None
            self._prefix_pending = False
            self._set_text(data, text)
        if not text:
            return []
        return [self._sequence(data)]

    def _overlap_size(self) -> int:
        for size in range(min(len(self._expected), len(self._candidate)), 0, -1):
            if self._expected.endswith(self._candidate[:size]):
                return size
        return 0

    def _flush_candidate(self) -> list[dict[str, Any]]:
        self._prefix_pending = False
        if not self._candidate or self._candidate_event is None:
            return []
        data = self._candidate_event
        self._trimmed = self._overlap_size()
        text = self._candidate[self._trimmed :]
        self._set_text(data, text)
        self._candidate = ""
        self._candidate_event = None
        return [self._sequence(data)] if text else []
