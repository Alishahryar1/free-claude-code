"""Strict automatic policy execution using existing provider ports and failures."""

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.request_outcomes import record_request_route
from free_claude_code.core.stream_delivery import bind_strict_routing
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.core.trace import close_stream_input

from .routing import ProviderModelCandidate, ResolvedModelRoute

type CandidateOpener = Callable[
    [int, ProviderModelCandidate, ContinuationSeed | None],
    Awaitable[AsyncIterator[str]],
]


async def strict_policy_candidates(
    resolved: ResolvedModelRoute, open_candidate: CandidateOpener, *, request_id: str
) -> AsyncIterator[str]:
    assert resolved.policy_deadline_seconds is not None
    assert resolved.policy_first_event_timeout_seconds is not None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + resolved.policy_deadline_seconds

    def check_deadline(limit: float) -> None:
        # A timeout at an already elapsed deadline can otherwise complete a
        # synchronous buffered read before its cancellation callback runs.
        if loop.time() >= limit:
            raise TimeoutError

    def timeout_failure() -> ExecutionFailure:
        return ExecutionFailure(
            FailureKind.TIMEOUT,
            504,
            f"Automatic routing deadline expired. Request ID: {request_id}",
            True,
        )

    candidates = (ProviderModelCandidate(resolved.primary), *resolved.fallbacks[:1])
    for index, candidate in enumerate(candidates):
        stream: AsyncIterator[str] | None = None
        forwarded = False
        failure: ExecutionFailure | None = None
        target = candidate.target
        record_request_route(target.provider_id, target.provider_model)
        try:
            first_deadline = min(
                deadline, loop.time() + resolved.policy_first_event_timeout_seconds
            )
            check_deadline(first_deadline)
            with bind_strict_routing(True):
                async with asyncio.timeout_at(first_deadline):
                    stream = await open_candidate(index, candidate, None)
                    check_deadline(first_deadline)
                    while True:
                        check_deadline(first_deadline)
                        first = await anext(stream)
                        check_deadline(first_deadline)
                        if first:
                            break
                        await asyncio.sleep(0)
            forwarded = True
            yield first
            while True:
                check_deadline(deadline)
                with bind_strict_routing(True):
                    async with asyncio.timeout_at(deadline):
                        chunk = await anext(stream)
                check_deadline(deadline)
                yield chunk
        except StopAsyncIteration:
            if loop.time() >= deadline:
                failure = timeout_failure()
            elif forwarded:
                return
            else:
                failure = ExecutionFailure(
                    FailureKind.UPSTREAM,
                    503,
                    "Provider returned no stream events.",
                    True,
                )
        except TimeoutError:
            failure = timeout_failure()
        except ExecutionFailure as error:
            failure = error
        finally:
            if stream is not None:
                with bind_strict_routing(True):
                    await close_stream_input(
                        stream,
                        owner="routing_policy",
                        source="application",
                        preserved_error=sys.exception() or failure,
                    )
        assert failure is not None
        if (
            forwarded
            or index + 1 >= len(candidates)
            or loop.time() >= deadline
            or not failure.retryable
            or failure.kind
            not in {
                FailureKind.RATE_LIMIT,
                FailureKind.OVERLOADED,
                FailureKind.TIMEOUT,
                FailureKind.UPSTREAM,
                FailureKind.UNAVAILABLE,
            }
        ):
            raise failure
