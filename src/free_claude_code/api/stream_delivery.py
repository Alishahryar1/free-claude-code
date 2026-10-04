"""Own public projection, tool release, recovery mapping and wire termination."""

import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from copy import deepcopy
from typing import Any, Literal

import simplejson

from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.async_iterators import try_close_async_iterator
from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)

from .classifier_projection import ClassifierProjection
from .stream_frames import frame_data, replace_frame_data
from .tool_call_buffer import ToolCallBuffer


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


class PublicResponseStream(AsyncIterator[str]):
    """Publish one response while providers retain their physical request loops."""

    def __init__(
        self,
        body: AsyncIterator[str],
        state: StreamDeliveryState,
        *,
        wire_api: Literal["messages", "responses"],
        first_chunk: str,
        terminal_frame: Callable[[str, str, BaseException], str] | None,
        terminal_failure_observer: Callable[[BaseException], None] | None,
        hide_reasoning: bool = False,
    ) -> None:
        self._body = body
        self.state = state
        self._wire_api = wire_api
        self._revision = state.attempt_revision
        self._hide_reasoning = hide_reasoning
        self._decoder = AnthropicSSEDecoder()
        self._projection = ClassifierProjection() if hide_reasoning else None
        self._tools = ToolCallBuffer(wire_api)
        self._delivered = DeliveredResponse(wire_api)
        self._continuation = ContinuationStream(self._delivered)
        state.bind_prefix(self._delivered.snapshot)
        self._published_starts: set[str] = set()
        self._seen_starts: set[str] = set()
        self._replacement = False
        self._public_id: str | None = None
        self._created_at: object = None
        self._sequence = -1
        self._offset: int | None = None
        self._pending_starts: list[str] = []
        self.start_frame = ""
        self._initial_chunk = first_chunk
        self._latest_chunk = ""
        self._terminal_frame = terminal_frame
        self._terminal_failure_observer = terminal_failure_observer
        self._done = False
        self._closed = False
        self._events = self._iterate()

    def __aiter__(self) -> PublicResponseStream:
        return self

    async def __anext__(self) -> str:
        if self._closed or self._done:
            raise StopAsyncIteration
        try:
            while True:
                frame, synthetic = await anext(self._events)
                frame = self._finalize_frame(frame)
                value = self._publish(frame, synthetic=synthetic)
                if value is not None:
                    self._latest_chunk = value
                    return value
        except StopAsyncIteration:
            self._done = True
            raise
        except (Exception, BaseExceptionGroup) as exc:
            self._done = True
            self._pending_starts.clear()
            self._tools = ToolCallBuffer(self._wire_api)
            failure = find_execution_failure(exc) or exc
            if self._terminal_frame is None:
                raise
            if self._terminal_failure_observer is not None:
                self._terminal_failure_observer(failure)
            frame = self._terminal_frame(
                self.start_frame or self._initial_chunk, self._latest_chunk, failure
            )
            frame = self._finalize_frame(frame)
            return self._publish(frame, synthetic=True) or frame

    def _finalize_frame(self, frame: str) -> str:
        payload = frame_data(frame)
        if payload is None:
            return frame
        kind = payload.get("type")
        if kind == "response.failed":
            response = payload.get("response")
            if not isinstance(response, dict):
                return frame
            # Failure snapshots may be based on the provider's empty start frame.
            # Use only retained public evidence, independently of a failed mapper.
            if self._delivered.retention_exhausted:
                response["output"] = []
                response["usage"] = None
                error = response.get("error")
                error = error if isinstance(error, dict) else {}
                message = error.get("message") or "Provider response failed."
                response["error"] = {
                    **error,
                    "message": f"{message}\n\nThe partial answer was already streamed. "
                    "A complete final snapshot is unavailable.",
                }
            elif self._continuation.active:
                response["output"] = [
                    deepcopy(item) for _, item in sorted(self._delivered.items.items())
                ]
                response["usage"] = self._delivered.usage(response["output"])
            else:
                return frame
        elif not self._continuation.active or kind in {"error", "response.error"}:
            return frame
        else:
            payload = self._continuation.finalize(payload)
        return replace_frame_data(frame, payload)

    async def _iterate(self) -> AsyncGenerator[tuple[str, bool]]:
        while True:
            try:
                chunk = await anext(self._body)
            except StopAsyncIteration:
                for frame in self._synchronize():
                    yield frame, True
                if self._continuation.handoff:
                    for payload in self._delivered.handoff_events():
                        for frame in self._prepare(self._format(payload)):
                            yield frame, True
                else:
                    for frame in self._decoder.finish_frames():
                        for ready in self._process(frame):
                            yield ready, False
                return
            for frame in self._synchronize():
                yield frame, True
            for frame in self._decoder.feed_frames(chunk):
                for ready in self._process(frame):
                    yield ready, False

    def _synchronize(self) -> Iterator[str]:
        transition = self.state.transition
        if self._revision == self.state.attempt_revision and transition is None:
            return
        self._revision = self.state.attempt_revision
        self.state.transition = None
        self._decoder = AnthropicSSEDecoder()
        self._tools = ToolCallBuffer(self._wire_api)
        self._projection = ClassifierProjection() if self._hide_reasoning else None
        self._seen_starts.clear()
        self._pending_starts.clear()
        self._offset = None
        if transition is not None:
            self._continuation.begin(handoff=transition == "handoff")
        elif self._continuation.active:
            self._continuation.restart_attempt()
        self._replacement = (
            bool(self._published_starts) and not self._continuation.active
        )
        for event in self._continuation.boundary():
            yield self._format(event)

    def _process(self, frame: str) -> Iterator[str]:
        if self._projection is not None:
            projected = self._projection.feed(frame)
            if projected is None:
                return
            frame = projected
        for released in self._tools.feed(frame):
            yield from self._prepare(released)

    def _prepare(self, frame: str) -> Iterator[str]:
        if not self._continuation.active:
            yield frame
            return
        payload = frame_data(frame)
        if payload is None:
            yield frame
            return
        for item in self._continuation.prepare(payload):
            yield (
                replace_frame_data(frame, item)
                if item.get("type") == payload.get("type")
                else self._format(item)
            )

    @staticmethod
    def _format(payload: dict[str, Any]) -> str:
        return f"event: {payload['type']}\ndata: {simplejson.dumps(payload, use_decimal=True)}\n\n"

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = self._done = True
        await self._events.aclose()
        self._pending_starts.clear()
        self._decoder = AnthropicSSEDecoder()
        self._tools = ToolCallBuffer(self._wire_api)
        self.state.bind_prefix(None)
        self._delivered = DeliveredResponse(self._wire_api)
        self._continuation = ContinuationStream(self._delivered)
        self.start_frame = self._initial_chunk = self._latest_chunk = ""
        error = await try_close_async_iterator(self._body)
        if error is not None:
            raise error

    def _release_content(self, frame: str, *, synthetic: bool) -> str:
        self.state.release_content(synthetic=synthetic)
        for start in self._pending_starts:
            self._record(start, synthetic=synthetic)
        self._record(frame, synthetic=synthetic)
        if not self._pending_starts:
            return frame
        self.start_frame = self._pending_starts[0]
        self._published_starts.add("message_start")
        prefix = "".join(self._pending_starts)
        self._pending_starts.clear()
        return prefix + frame

    def _record(self, frame: str, *, synthetic: bool) -> None:
        payload = frame_data(frame)
        if payload is not None:
            self._delivered.observe(payload, count_progress=not synthetic)
        elif any(line.startswith("data:") for line in frame.splitlines()):
            self._delivered.unsafe_reason = "unknown_event"

    def _publish(self, frame: str, *, synthetic: bool = False) -> str | None:
        """Normalize only a replacement envelope and mark actually released frames."""
        payload = frame_data(frame)
        if payload is None:
            if any(
                line.strip() and not line.startswith(":")
                for line in re.split(r"\r\n|\r|\n", frame)
            ):
                return self._release_content(frame, synthetic=synthetic)
            return frame

        raw_kind = payload.get("type")
        kind = raw_kind if isinstance(raw_kind, str) else ""
        metadata = self._metadata(kind, payload)
        if self._wire_api == "messages":
            if kind == "message_start" and metadata and not self.state.content_released:
                self._pending_starts.append(frame)
                return None
            if kind == "error":
                self._pending_starts.clear()
        start = kind in {"message_start", "response.created", "response.in_progress"}
        if start and metadata:
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
        frame = replace_frame_data(frame, payload) if changed else frame
        if metadata:
            self._record(frame, synthetic=synthetic)
            return frame
        return self._release_content(frame, synthetic=synthetic)

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
