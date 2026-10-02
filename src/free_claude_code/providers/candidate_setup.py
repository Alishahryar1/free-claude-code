"""A candidate owns asynchronous setup before its concrete transport is known."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager

from free_claude_code.application.ports import ProviderCandidate
from free_claude_code.core.async_iterators import complete_cleanup
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.recovery import AttemptFailure, RecoveryCheckpoint
from free_claude_code.core.stream_events import DecodedStreamEvent

from .admission import ProviderExecution
from .failure_policy import ProviderRecoveryDeferred, classify_provider_failure
from .http import maybe_await_aclose


class CandidateSetup:
    """Retain setup progress and discovery budget across pre-dispatch deferrals."""

    def __init__(
        self,
        open_candidate: Callable[
            [bool], Awaitable[AbstractAsyncContextManager[ProviderCandidate]]
        ],
        *,
        provider_name: str,
        request_id: str | None,
        discovery: ProviderExecution | None = None,
    ) -> None:
        self._open_candidate = open_candidate
        self._provider_name = provider_name
        self._request_id = request_id
        self._discovery = discovery
        self._candidate: ProviderCandidate | None = None
        self._resources = AsyncExitStack()
        self._closed = False

    @asynccontextmanager
    async def open(self) -> AsyncIterator[ProviderCandidate]:
        try:
            yield self
        finally:
            await self.aclose()

    @property
    def can_attempt(self) -> bool:
        return not self._closed and (
            self._candidate is None or self._candidate.can_attempt
        )

    async def prepare(self, checkpoint: RecoveryCheckpoint) -> None:
        if self._candidate is not None:
            await self._candidate.prepare(checkpoint)

    async def stream_attempt(
        self,
        checkpoint: RecoveryCheckpoint,
        *,
        wait_for_recovery: bool,
        can_correct: Callable[[], bool],
        on_rejected: Callable[[], None] | None = None,
    ) -> AsyncIterator[DecodedStreamEvent]:
        try:
            if self._candidate is None:
                context = await self._open_candidate(wait_for_recovery)
                self._candidate = await self._resources.enter_async_context(context)
        except ProviderRecoveryDeferred as error:
            raise AttemptFailure(
                classify_provider_failure(
                    error.last_error,
                    provider_name=self._provider_name,
                    read_timeout_s=None,
                    request_id=self._request_id,
                ),
                deferred=True,
            ) from error
        except ExecutionFailure as error:
            raise AttemptFailure(error) from error
        stream = self._candidate.stream_attempt(
            checkpoint,
            wait_for_recovery=wait_for_recovery,
            can_correct=can_correct,
            on_rejected=on_rejected,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await maybe_await_aclose(stream)

    def finish(self, failure: ExecutionFailure | None) -> None:
        if self._candidate is not None:
            self._candidate.finish(failure)

    async def suspend(self) -> None:
        if self._candidate is not None:
            await self._candidate.suspend()
        if self._discovery is not None:
            await self._discovery.suspend()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await complete_cleanup(self._resources.aclose())
        finally:
            if self._discovery is not None:
                await self._discovery.aclose()
