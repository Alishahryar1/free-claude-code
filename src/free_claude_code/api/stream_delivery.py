"""Observe public delivery and retain one response envelope across hidden retries."""

import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import simplejson

from free_claude_code.core.async_iterators import try_close_async_iterator
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)


class DeliveryObservedStream(AsyncIterator[str]):
    """Bind observation for each read/close, including prefetch in another task."""

    def __init__(self, body: AsyncIterator[str], state: StreamDeliveryState) -> None:
        self._body = body
        self._state = state

    def __aiter__(self) -> DeliveryObservedStream:
        return self

    async def __anext__(self) -> str:
        with bind_stream_delivery(self._state):
            return await anext(self._body)

    async def aclose(self) -> None:
        with bind_stream_delivery(self._state):
            error = await try_close_async_iterator(self._body)
        if error is not None:
            raise error


def frame_data(frame: str) -> dict[str, Any] | None:
    """Read only SSE data, retaining decimal precision for envelope edits."""
    text = "\n".join(
        line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:")
    )
    try:
        value: object = simplejson.loads(text, use_decimal=True)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def replace_frame_data(frame: str, payload: Mapping[str, object]) -> str:
    """Replace data lines while preserving other SSE fields and line endings."""
    lines = re.split(r"(\r\n|\r|\n)", frame)
    replacement = "data: " + simplejson.dumps(
        payload, use_decimal=True, ensure_ascii=False
    )
    parts: list[str] = []
    replaced = False
    for index in range(0, len(lines), 2):
        line = lines[index]
        ending = lines[index + 1] if index + 1 < len(lines) else ""
        if line.startswith("data:"):
            if replaced:
                continue
            line = replacement
            replaced = True
        parts.extend((line, ending))
    return "".join(parts)


class PublicStreamEnvelope:
    """Own published metadata without reconstructing content or tool arguments."""

    def __init__(
        self, state: StreamDeliveryState, *, wire_api: Literal["messages", "responses"]
    ) -> None:
        self.state = state
        self._wire_api = wire_api
        self._revision = state.attempt_revision
        self._published_starts: set[str] = set()
        self._seen_starts: set[str] = set()
        self._replacement = False
        self._public_id: str | None = None
        self._created_at: object = None
        self._sequence: int = -1
        self._offset: int | None = None
        self._initial_usage: dict[str, Any] = {}
        self.start_frame = ""

    def synchronize(self) -> None:
        if self._revision == self.state.attempt_revision:
            return
        self._revision = self.state.attempt_revision
        self._replacement = bool(self._published_starts)
        self._seen_starts.clear()
        self._offset = None
        self._initial_usage.clear()

    def publish(self, frame: str) -> str | None:
        """Normalize only a replacement envelope and mark actually released frames."""
        self.synchronize()
        payload = frame_data(frame)
        if payload is None:
            if any(
                line.strip() and not line.startswith(":") for line in frame.splitlines()
            ):
                self.state.release_content()
            return frame

        raw_kind = payload.get("type")
        kind = raw_kind if isinstance(raw_kind, str) else ""
        metadata = self._metadata(kind, payload)
        start = kind in {"message_start", "response.created", "response.in_progress"}
        if start and metadata:
            if kind == "message_start":
                message = payload.get("message")
                if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                    self._initial_usage = dict(message["usage"])
            if kind not in self._seen_starts:
                self._seen_starts.add(kind)
                if self._replacement and kind in self._published_starts:
                    return None
            self._published_starts.add(kind)
            if not self.start_frame:
                self.start_frame = frame
                value = payload.get(
                    "message" if self._wire_api == "messages" else "response"
                )
                if isinstance(value, dict):
                    self._public_id = value.get("id")
                    self._created_at = value.get("created_at")

        changed = False
        if self._replacement and self._wire_api == "responses":
            response = payload.get("response")
            if isinstance(response, dict):
                if (
                    self._public_id is not None
                    and response.get("id") != self._public_id
                ):
                    response["id"] = self._public_id
                    changed = True
                if (
                    self._created_at is not None
                    and response.get("created_at") != self._created_at
                ):
                    response["created_at"] = self._created_at
                    changed = True
            if (
                "response_id" in payload
                and self._public_id is not None
                and payload["response_id"] != self._public_id
            ):
                payload["response_id"] = self._public_id
                changed = True
        if self._replacement and kind == "message_delta":
            delta = payload.get("delta")
            usage = payload.get("usage")
            if (
                isinstance(delta, dict)
                and delta.get("stop_reason") is not None
                and isinstance(usage, dict)
            ):
                for field in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    if (
                        usage.get(field) is None
                        and self._initial_usage.get(field) is not None
                    ):
                        usage[field] = self._initial_usage[field]
                        changed = True

        number = payload.get("sequence_number")
        if isinstance(number, int) and not isinstance(number, bool):
            if self._replacement:
                if self._offset is None:
                    self._offset = max(0, self._sequence + 1 - number)
                if self._offset:
                    payload["sequence_number"] = number + self._offset
                    changed = True
                number += self._offset
            self._sequence = max(self._sequence, number)
        if not metadata:
            self.state.release_content()
        return replace_frame_data(frame, payload) if changed else frame

    def _metadata(self, kind: str, payload: dict[str, Any]) -> bool:
        if kind == "ping":
            return True
        if self._wire_api == "messages" and kind == "message_start":
            message = payload.get("message")
            return (
                isinstance(message, dict)
                and message.get("content", []) == []
                and message.get("stop_reason") is None
            )
        if self._wire_api == "responses" and kind in {
            "response.created",
            "response.in_progress",
        }:
            response = payload.get("response")
            return (
                isinstance(response, dict)
                and response.get("output", []) == []
                and response.get("status") in (None, "queued", "in_progress")
            )
        return False
