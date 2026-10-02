"""Scripted decoded candidates for application/API tests without provider HTTP."""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from free_claude_code.core.anthropic.native_stream import NativeMessagesStreamState
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.async_iterators import CleanupBudget
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.openai_responses import ResponsesSourceState
from free_claude_code.core.recovery import (
    AttemptDispatch,
    AttemptFailure,
    RecoveryCheckpoint,
)
from free_claude_code.core.stream_events import RequestOutcome, StreamEvent
from free_claude_code.core.stream_observations import (
    DecodedStreamEvent,
    MessagesObservation,
    ResponsesObservation,
)
from free_claude_code.core.trace import close_stream_input


def progress_frame(text: str) -> str:
    return StreamEvent("fixture.progress", {"value": text}).serialize()


class ScriptedAttempt:
    def __init__(self, source: AsyncIterator[DecodedStreamEvent]) -> None:
        self.source = source
        self.dispatch: AttemptDispatch | None = None
        self.read_task: asyncio.Task[DecodedStreamEvent] | None = None

    async def read(self) -> DecodedStreamEvent:
        self.dispatch = AttemptDispatch()

        async def advance():
            return await anext(self.source)

        self.read_task = asyncio.create_task(advance())
        return await asyncio.shield(self.read_task)

    async def resolve_failure(
        self, failure: AttemptFailure, *, can_correct: bool
    ) -> AttemptFailure:
        return failure

    async def aclose(self, *, active_error: BaseException | None) -> None:
        async def close():
            if self.read_task is not None:
                if not self.read_task.done():
                    self.read_task.cancel()
                with suppress(Exception, asyncio.CancelledError):
                    await self.read_task
            await close_stream_input(
                self.source,
                owner="fixture",
                source="test",
                preserved_error=active_error,
            )

        await CleanupBudget().run(close())


class ScriptedProvider:
    """Tests supply a wire transcript; only this fixture decodes it into the port."""

    @asynccontextmanager
    async def open_messages(self, request: Any, **kwargs: Any):
        candidate = _ScriptedCandidate(
            self.stream_messages(request, **kwargs), "messages"
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()

    @asynccontextmanager
    async def open_responses(self, request: Any, **kwargs: Any):
        candidate = _ScriptedCandidate(
            self.stream_responses(request, **kwargs), "responses"
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()

    @asynccontextmanager
    async def open_native_messages(self, request: Any, **kwargs: Any):
        candidate = _ScriptedCandidate(
            self.stream_native_messages(request, **kwargs), "messages"
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()

    def stream_messages(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        raise AssertionError("Unexpected Messages call")

    def stream_responses(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        raise AssertionError("Unexpected Responses call")

    def stream_native_messages(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        raise AssertionError("Unexpected native Messages call")


class _ScriptedCandidate:
    can_attempt = False

    def __init__(self, source: AsyncIterator[str], protocol: str) -> None:
        self.source = source
        self.origin = ReplayOrigin(
            "fixture",
            "messages" if protocol == "messages" else "responses",
            "fixture",
            "fixture",
            "fixture",
        )
        self.error: BaseException | None = None
        self.closed = False

    def open_attempt(self, checkpoint: RecoveryCheckpoint, *, wait_for_recovery: bool):
        checkpoint.require_origin(self.origin)
        return ScriptedAttempt(self._read())

    def finish(self, failure: ExecutionFailure | None) -> None:
        self.error = failure or self.error

    async def suspend(self) -> None:
        pass

    async def _read(self):
        decoder = AnthropicSSEDecoder()
        native_source = NativeMessagesStreamState(permissive=True)
        responses_source = ResponsesSourceState()
        framed = False
        try:
            async for chunk in self.source:
                if not chunk:
                    yield DecodedStreamEvent(self.origin, StreamEvent("ping", {}))
                    continue
                if not framed and not chunk.startswith(("event:", "data:", ":")):
                    event = StreamEvent("fixture.progress", {"value": chunk})
                    yield DecodedStreamEvent(
                        self.origin,
                        event,
                        observation=MessagesObservation()
                        if self.origin.protocol == "messages"
                        else ResponsesObservation(events=(event,)),
                        progress=True,
                    )
                    continue
                framed = True
                frames = decoder.feed(chunk)
                for frame in frames:
                    event = StreamEvent(
                        frame.event or frame.data.get("type", ""), frame.data
                    )
                    if event.kind in {"response.failed", "error"}:
                        error = frame.data.get("response", frame.data).get("error", {})
                        kind, status = {
                            "rate_limit_error": (FailureKind.RATE_LIMIT, 429),
                            "timeout_error": (FailureKind.TIMEOUT, 504),
                            "overloaded_error": (FailureKind.OVERLOADED, 529),
                            "authentication_error": (FailureKind.AUTHENTICATION, 401),
                            "permission_error": (FailureKind.PERMISSION, 403),
                            "invalid_request_error": (FailureKind.INVALID_REQUEST, 400),
                        }.get(error.get("type"), (FailureKind.UPSTREAM, 502))
                        if error.get("code") == "context_length_exceeded":
                            kind, status = FailureKind.CONTEXT_WINDOW_EXCEEDED, 400
                        raise ExecutionFailure(
                            kind,
                            status,
                            error.get("message", "Scripted failure"),
                            False,
                        )
                    if self.origin.protocol == "responses":
                        observation = responses_source.observe(event)
                    else:
                        completed = (
                            native_source.accept(event.kind, event.payload)
                            if event.payload.get("type") == event.kind
                            else None
                        )
                        observation = MessagesObservation(completed)
                    yield DecodedStreamEvent(
                        self.origin,
                        event,
                        observation=observation,
                        progress=event.kind
                        not in {
                            "ping",
                            "message_start",
                            "response.created",
                            "response.in_progress",
                        },
                        outcome=RequestOutcome.SUCCESS
                        if event.kind
                        in {"message_stop", "response.completed", "response.incomplete"}
                        else None,
                    )
            yield DecodedStreamEvent(
                self.origin,
                StreamEvent("fixture.completed", {}),
                outcome=RequestOutcome.SUCCESS,
            )
        except ExecutionFailure as error:
            self.error = error
            raise AttemptFailure(error) from error
        finally:
            self.error = self.error or sys.exception()
            await self.aclose()

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        await close_stream_input(
            self.source,
            owner="scripted_candidate",
            source="test",
            preserved_error=self.error,
        )
