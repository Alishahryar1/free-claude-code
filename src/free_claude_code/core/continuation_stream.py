"""Map continuation events into an existing public response lifecycle."""

from copy import deepcopy
from typing import Any, Literal
from uuid import uuid4, uuid5

from .continuation_overlap import ContinuationOverlap
from .delivered_response import (
    DeliveredResponse,
    client_call,
    responses_text_constraint,
)
from .failures import ExecutionFailure, FailureKind
from .gemini.response import gemini_finished, gemini_parts


class ContinuationStream:
    """Own only recovery-time mappings; ordinary events remain untouched."""

    def __init__(self, delivered: DeliveredResponse) -> None:
        self.delivered = delivered
        self.active = False
        self.handoff = False
        self._offset = 0
        self._namespace = uuid4()
        self._boundary: list[dict[str, Any]] = []
        self._expected: dict[Literal["text", "thinking"], str] = {}
        self._overlaps: list[ContinuationOverlap] = []
        self._initial_progress = 0
        self._prefix_items: list[dict[str, Any]] = []

    @property
    def made_progress(self) -> bool:
        return self.delivered.progress > self._initial_progress

    def begin(self, *, handoff: bool) -> None:
        self.active = True
        self.handoff = handoff
        self._offset = max(self.delivered.items, default=-1) + 1
        self._boundary = self.delivered.closing_events()
        prefix = self.delivered.snapshot().continuation
        assert prefix is not None
        self._expected = {
            channel: prefix.readable_text(channel) for channel in ("text", "thinking")
        }
        self.restart_attempt()
        finalized = deepcopy(self.delivered.items)
        for event in self._boundary:
            if event["type"] == "response.output_item.done":
                finalized[event["output_index"]] = deepcopy(event["item"])
        self._prefix_items = [item for _, item in sorted(finalized.items())]

    def restart_attempt(self) -> None:
        """Replace an invisible attempt without losing its finalized prefix."""
        self._namespace = uuid4()
        self._overlaps = [
            ContinuationOverlap(expected, channel=channel)
            for channel, expected in self._expected.items()
        ]
        self._initial_progress = self.delivered.progress

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

    def prepare(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.active:
            return [payload]
        if (
            self.delivered.wire_api == "responses"
            and payload.get("type")
            not in {"response.failed", "response.error", "error"}
            and responses_text_constraint(payload)
        ):
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message="The continuation returned text with a native output contract that cannot be reconstructed.",
                retryable=False,
            )
        if payload.get("type") in {
            "message_start",
            "response.created",
            "response.in_progress",
        }:
            return []
        if self.delivered.wire_api == "chat":
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                if (
                    choice.get("logprobs") is not None
                    or choice.get("index") != 0
                    or any(
                        key
                        not in {
                            "role",
                            "content",
                            "reasoning",
                            "reasoning_content",
                            "reasoning_details",
                            "tool_calls",
                        }
                        and value is not None
                        for key, value in delta.items()
                    )
                ):
                    raise ExecutionFailure(
                        kind=FailureKind.UPSTREAM,
                        status_code=502,
                        message="Chat continuation returned an unsupported native output contract.",
                        retryable=False,
                    )
        events = [deepcopy(payload)]
        if not self.handoff:
            for overlap in self._overlaps:
                events = [ready for event in events for ready in overlap.feed(event)]
        return [self._map(event) for event in events]

    def _map(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.delivered.wire_api == "chat":
            for key in ("id", "model", "created"):
                if key in self.delivered.header and key in data:
                    data[key] = self.delivered.header[key]
            for choice in data.get("choices", []):
                for detail in (choice.get("delta") or {}).get("reasoning_details", []):
                    if isinstance(detail.get("id"), str):
                        detail["id"] = self._item_id(detail["id"])
            return data
        if self.delivered.wire_api == "gemini":
            for key in ("responseId", "modelVersion"):
                if key in self.delivered.header:
                    data[key] = self.delivered.header[key]
            return data
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
        response = data.get("response")
        if isinstance(response, dict):
            response["id"] = self.delivered.header.get("id", response.get("id"))
            if "created_at" in self.delivered.header:
                response["created_at"] = self.delivered.header["created_at"]
            if not self.handoff:
                for item in response.get("output", []):
                    if isinstance(item.get("id"), str):
                        item["id"] = self._item_id(item["id"])
        return self._sequence(data)

    def finalize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Finish the response only after preceding frames have been published."""
        if not self.active:
            return data
        if self.delivered.wire_api == "chat":
            if (
                any(
                    choice.get("finish_reason") is not None
                    for choice in data.get("choices", [])
                )
                and not self.made_progress
                and not any(
                    any(
                        (choice.get("delta") or {}).get(key)
                        for key in (
                            "content",
                            "reasoning",
                            "reasoning_content",
                            "reasoning_details",
                            "tool_calls",
                        )
                    )
                    for choice in data.get("choices", [])
                )
            ):
                raise ExecutionFailure(
                    kind=FailureKind.UPSTREAM,
                    status_code=502,
                    message="The provider continuation completed without adding output.",
                    retryable=False,
                )
            data.pop("usage", None)
            return data
        if self.delivered.wire_api == "gemini":
            if (
                gemini_finished(data)
                and not self.handoff
                and not self.made_progress
                and not any(
                    part.get("text") or part.get("functionCall")
                    for part in gemini_parts(data)
                )
            ):
                raise ExecutionFailure(
                    kind=FailureKind.UPSTREAM,
                    status_code=502,
                    message="The provider continuation completed without adding output.",
                    retryable=False,
                )
            # After overlap removal, keep an empty terminal valid for SDK chat
            # history. This adds no output and cannot satisfy the progress check.
            for candidate in data.get("candidates", []):
                if candidate.get("finishReason") and not candidate.get(
                    "content", {}
                ).get("parts"):
                    candidate["content"] = {"role": "model", "parts": [{"text": ""}]}
            # Per-attempt usage cannot describe a synthesized multi-attempt answer.
            data.pop("usageMetadata", None)
            return data
        kind = data.get("type")
        response = data.get("response")
        output = response.get("output", []) if isinstance(response, dict) else []
        terminal = kind in {"response.completed", "response.incomplete", "message_stop"}
        terminal |= (
            kind == "message_delta"
            and data.get("delta", {}).get("stop_reason") is not None
        )
        if (
            terminal
            and not self.handoff
            and not (self.made_progress or self._has_output(output))
        ):
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message="The provider continuation completed without adding output.",
                retryable=False,
            )
        if isinstance(response, dict):
            response["output"] = [
                *deepcopy(self._prefix_items),
                *([] if self.handoff else output),
            ]
            response["usage"] = self.delivered.usage(response["output"])
        elif kind == "message_delta" and "usage" in data:
            data["usage"] = {"output_tokens": self.delivered.output_tokens()}
        return data

    @staticmethod
    def _has_output(items: list[dict[str, Any]]) -> bool:
        return any(
            client_call(item)
            or any(
                part.get("text") or part.get("refusal")
                for part in [*(item.get("content") or []), *(item.get("summary") or [])]
            )
            for item in items
        )
