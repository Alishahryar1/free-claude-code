"""Provider-specific initialization with a retained discovery allowance."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from free_claude_code.application.ports import CandidateInitializer, ProviderCandidate
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.recovery import AttemptFailure

from .admission import ProviderExecution
from .failure_policy import ProviderRecoveryDeferred, classify_provider_failure


@asynccontextmanager
async def deferred_candidate(
    initialize: Callable[
        [bool], Awaitable[AbstractAsyncContextManager[ProviderCandidate]]
    ],
    *,
    provider_name: str,
    request_id: str | None,
    discovery: ProviderExecution | None = None,
) -> AsyncIterator[CandidateInitializer]:
    """The request's candidate session owns the initialized context and its leases."""

    @asynccontextmanager
    async def open_candidate(
        wait_for_recovery: bool,
    ) -> AsyncIterator[ProviderCandidate]:
        try:
            context = await initialize(wait_for_recovery)
            async with context as candidate:
                yield candidate
        except ProviderRecoveryDeferred as error:
            raise AttemptFailure(
                classify_provider_failure(
                    error.last_error,
                    provider_name=provider_name,
                    read_timeout_s=None,
                    request_id=request_id,
                ),
                deferred=True,
            ) from error
        except ExecutionFailure as error:
            raise AttemptFailure(error) from error

    try:
        yield CandidateInitializer(open_candidate)
    finally:
        if discovery is not None:
            await discovery.aclose()
