"""Recovery across candidates with one response writer and progress deadline."""

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, suppress
from dataclasses import dataclass, replace
from time import monotonic

from free_claude_code.core.anthropic.recovery_stream import (
    MessagesRecoveryWriter,
    NativeMessagesCompletionWriter,
)
from free_claude_code.core.async_iterators import complete_cleanup
from free_claude_code.core.failures import (
    ExecutionFailure,
    FailureKind,
    find_execution_failure,
)
from free_claude_code.core.openai_responses import (
    ResponsesRecoveryWriter,
)
from free_claude_code.core.recovery import (
    AttemptFailure,
    CandidateIncompatible,
    RecoveryCheckpoint,
)
from free_claude_code.core.request_outcomes import (
    record_request_exception,
    record_request_route,
)
from free_claude_code.core.stream_events import DecodedStreamEvent
from free_claude_code.core.trace import close_stream_input, trace_event

from .errors import InvalidRequestError
from .ports import ProviderCandidate
from .routing import ProviderModelTarget

type RecoveryWriter = (
    MessagesRecoveryWriter | ResponsesRecoveryWriter | NativeMessagesCompletionWriter
)
type CandidateOpener = Callable[
    [int, ProviderModelTarget],
    Awaitable[AbstractAsyncContextManager[ProviderCandidate]],
]
type FallbackObserver = Callable[
    [ProviderModelTarget, ProviderModelTarget, ExecutionFailure, int], None
]
type SelectionObserver = Callable[[ProviderModelTarget, int], None]


@dataclass(slots=True)
class _OpenedCandidate:
    index: int
    target: ProviderModelTarget
    candidate: ProviderCandidate
    resources: AsyncExitStack
    closed: bool = False


class RecoveryCoordinator:
    """Select, correct and continue attempts without parsing protocol payloads."""

    def __init__(
        self,
        *,
        candidates: tuple[ProviderModelTarget, ...],
        opener: CandidateOpener,
        writer: RecoveryWriter,
        progress_timeout_seconds: float,
        timeout_failure: Callable[[str], ExecutionFailure],
        request_id: str,
        on_fallback: FallbackObserver | None = None,
        on_selected: SelectionObserver | None = None,
    ) -> None:
        self._candidates = candidates
        self._opener = opener
        self._writer = writer
        self._timeout_seconds = progress_timeout_seconds
        self._timeout_failure = timeout_failure
        self._request_id = request_id
        self._on_fallback = on_fallback
        self._on_selected = on_selected
        self._cursor = 0
        self._recovering = False
        self._deadline = 0.0
        self._last_failure: ExecutionFailure | None = None
        self._last_incompatibility: CandidateIncompatible | None = None
        self._opened: list[_OpenedCandidate] = []
        self._provider_id = candidates[0].provider_id

    async def _bounded[T](self, operation: Awaitable[T]) -> T:
        timeout = asyncio.timeout_at(self._deadline)
        try:
            async with timeout:
                result = await operation
        except TimeoutError as error:
            if not timeout.expired():
                raise
            raise self._timeout_failure(self._provider_id) from error
        if timeout.expired():
            raise self._timeout_failure(self._provider_id)
        return result

    def _checkpoint(self) -> RecoveryCheckpoint:
        return replace(self._writer.checkpoint, recovering=self._recovering)

    async def _open_next(
        self, checkpoint: RecoveryCheckpoint
    ) -> _OpenedCandidate | None:
        while self._cursor < len(self._candidates):
            index = self._cursor
            self._cursor += 1
            target = self._candidates[index]
            resources = AsyncExitStack()
            retained = False
            try:
                # Runtime initialization has its own shared wait allowance.
                initialization_started = monotonic()
                try:
                    context = await self._opener(index, target)
                finally:
                    self._deadline += monotonic() - initialization_started
                candidate = await self._bounded(resources.enter_async_context(context))
                await self._bounded(
                    candidate.prepare(
                        replace(
                            checkpoint, recovering=index > 0 or checkpoint.recovering
                        )
                    )
                )
                opened = _OpenedCandidate(index, target, candidate, resources)
                self._opened.append(opened)
                retained = True
                return opened
            except CandidateIncompatible as error:
                self._last_incompatibility = error
                trace_event(
                    stage="execution",
                    event="free_claude_code.model_fallback.skipped",
                    source="application",
                    request_id=self._request_id,
                    provider_model_ref=target.provider_model_ref,
                    reason=str(error),
                )
            except ExecutionFailure as failure:
                # The shared deadline is terminal even while preparing a target.
                if asyncio.get_running_loop().time() >= self._deadline:
                    raise
                if self._last_failure is None:
                    self._last_failure = failure
                record_request_route(target.provider_id, target.provider_model)
            finally:
                if not retained:
                    await complete_cleanup(
                        self._close_resources(resources, sys.exception())
                    )
        return None

    async def _close(
        self, opened: _OpenedCandidate, failure: ExecutionFailure | None
    ) -> None:
        if opened.closed:
            return
        opened.closed = True
        opened.candidate.finish(failure)
        await complete_cleanup(self._close_resources(opened.resources, failure))

    async def _close_resources(
        self, resources: AsyncExitStack, error: BaseException | None
    ) -> None:
        await self._bounded(
            close_stream_input(
                resources,
                owner="recovery_candidate",
                source="application",
                preserved_error=error,
            )
        )

    async def stream(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        self._deadline = loop.time() + self._timeout_seconds
        current: _OpenedCandidate | None = None
        try:
            current = await self._open_next(self._checkpoint())
            if current is None:
                if self._last_failure is not None:
                    raise self._last_failure
                raise InvalidRequestError(
                    str(self._last_incompatibility)
                    if self._last_incompatibility
                    else "No configured candidate can preserve the required request features."
                )
            while current is not None:
                self._provider_id = current.target.provider_id
                record_request_route(
                    current.target.provider_id, current.target.provider_model
                )
                checkpoint = self._checkpoint()
                self._writer.begin_attempt()
                revision = self._writer.revision
                selected = False
                completed = False
                attempt_failure: AttemptFailure | None = None
                incompatible: CandidateIncompatible | None = None
                source = current.candidate.stream_attempt(
                    checkpoint,
                    wait_for_recovery=self._cursor >= len(self._candidates),
                    can_correct=lambda revision=revision: (
                        self._writer.revision == revision
                        and self._checkpoint().blocked_reason is None
                    ),
                )
                try:
                    while True:
                        try:
                            event = await self._bounded(
                                _next_event(source, self._deadline)
                            )
                        except StopAsyncIteration:
                            break
                        if event.progress:
                            self._deadline = loop.time() + self._timeout_seconds
                        events = self._writer.feed(event)
                        if events and not selected and current.index > 0:
                            selected = True
                            if self._on_selected is not None:
                                self._on_selected(current.target, current.index)
                        for output in events:
                            consumer_started = loop.time()
                            yield output.serialize()
                            self._deadline += loop.time() - consumer_started
                        if event.completed:
                            completed = True
                            break
                except AttemptFailure as error:
                    attempt_failure = error
                except CandidateIncompatible as error:
                    incompatible = error
                except ExceptionGroup as group:
                    failure = find_execution_failure(group)
                    if failure is None:
                        raise
                    attempt_failure = AttemptFailure(failure)
                finally:
                    active_error = sys.exception()
                    closing = close_stream_input(
                        source,
                        owner="recovery_coordinator",
                        source="application",
                        preserved_error=active_error or attempt_failure,
                    )
                    if isinstance(
                        active_error, (asyncio.CancelledError, GeneratorExit)
                    ):
                        await complete_cleanup(self._bounded(closing))
                    else:
                        await self._bounded(closing)

                if incompatible is not None:
                    trace_event(
                        stage="execution",
                        event="free_claude_code.model_fallback.skipped",
                        source="application",
                        request_id=self._request_id,
                        provider_model_ref=current.target.provider_model_ref,
                        reason=str(incompatible),
                    )
                    await self._close(current, self._last_failure)
                    current = await self._open_next(self._checkpoint())
                    if current is None:
                        if self._last_failure is not None:
                            raise self._last_failure
                        raise InvalidRequestError(str(incompatible))
                    continue

                if attempt_failure is not None and attempt_failure.request_rejected:
                    self._writer.reject_attempt()
                if attempt_failure is not None and attempt_failure.corrected:
                    # A correction is authorized only before new committed content.
                    for output in self._writer.interrupt():
                        consumer_started = loop.time()
                        yield output.serialize()
                        self._deadline += loop.time() - consumer_started
                    continue
                for output in self._writer.interrupt():
                    consumer_started = loop.time()
                    yield output.serialize()
                    self._deadline += loop.time() - consumer_started

                if completed:
                    if (
                        checkpoint.content
                        and self._writer.revision == revision
                        and not self._writer.allow_empty_completion
                    ):
                        attempt_failure = AttemptFailure(
                            ExecutionFailure(
                                FailureKind.UPSTREAM,
                                502,
                                "Provider continuation produced no new output.",
                                True,
                            ),
                            retry_allowed=current.candidate.can_attempt,
                        )
                    else:
                        if not selected and current.index > 0 and self._on_selected:
                            self._on_selected(current.target, current.index)
                        await self._close(current, None)
                        for output in self._writer.finish():
                            yield output.serialize()
                        return
                if attempt_failure is None:
                    raise RuntimeError(
                        "Provider attempt ended without a completion or failure."
                    )
                failure = attempt_failure.failure
                self._last_failure = failure
                self._recovering = True
                checkpoint = self._checkpoint()
                if checkpoint.blocked_reason is not None:
                    raise failure
                if checkpoint.published_tools:
                    await self._close(current, None)
                    for output in self._writer.finish(salvage=True):
                        yield output.serialize()
                    return

                next_candidate = await self._open_next(checkpoint)
                if next_candidate is not None:
                    await self._close(current, failure)
                    if self._on_fallback is not None:
                        self._on_fallback(
                            current.target,
                            next_candidate.target,
                            failure,
                            next_candidate.index,
                        )
                    current = next_candidate
                    continue
                if current.candidate.can_attempt and (
                    attempt_failure.deferred or attempt_failure.retry_allowed
                ):
                    continue
                raise self._last_failure
        except ExecutionFailure as failure:
            self._last_failure = failure
            if current is not None:
                current.candidate.finish(failure)
            record_request_exception(failure)
            trace_event(
                stage="execution",
                event="free_claude_code.execution.failed",
                source="application",
                request_id=self._request_id,
                provider_id=self._provider_id,
                failure_kind=failure.kind.value,
                status_code=failure.status_code,
                provider_retryable=failure.retryable,
            )
            if not self._writer.started:
                raise
            for output in self._writer.failure(failure):
                yield output.serialize()
        finally:
            primary_error = sys.exception() or self._last_failure
            cancelled = False
            for opened in reversed(self._opened):
                if opened.closed:
                    continue
                try:
                    opened.closed = True
                    await complete_cleanup(
                        self._close_resources(opened.resources, primary_error)
                    )
                except asyncio.CancelledError:
                    cancelled = True
                except Exception as error:
                    if primary_error is None:
                        raise
                    trace_event(
                        stage="execution",
                        event="provider.candidate.close_failed",
                        source="application",
                        request_id=self._request_id,
                        close_exc_type=type(error).__name__,
                        preserved_exc_type=type(primary_error).__name__,
                    )
            if cancelled:
                raise asyncio.CancelledError


async def _next_event(
    source: AsyncIterator[DecodedStreamEvent], deadline: float
) -> DecodedStreamEvent:
    async def advance() -> DecodedStreamEvent:
        return await anext(source)

    task = asyncio.create_task(advance())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()

        async def finish_read() -> None:
            async with asyncio.timeout_at(deadline):
                await task

        with suppress(Exception, asyncio.CancelledError):
            await complete_cleanup(finish_read())
        raise
