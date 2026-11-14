"""Execute ordered model candidates while preserving safe response continuity."""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.request_outcomes import record_request_route
from free_claude_code.core.stream_delivery import StreamDeliveryState
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.core.trace import close_stream_input

from ..routing import ProviderModelCandidate
from .decision import CandidateFailure, Fail, Fallback, Handoff, next_step
from .traces import trace_fallback_selected, trace_fallback_started
from .types import CandidateStreamOpener, WireApi


async def stream_model_candidates(
    candidates: tuple[ProviderModelCandidate, ...],
    open_candidate: CandidateStreamOpener,
    *,
    get_delivery: Callable[[], StreamDeliveryState | None],
    request_id: str,
    wire_api: WireApi,
    generation_id: int | None = None,
) -> AsyncGenerator[StreamOutput]:
    """Run candidates sequentially, closing each before a safe model transition.

    Delivery is acquired on first iteration, after the response consumer binds it.
    The opener owns protocol-specific preparation; this iterator owns its streams.
    """
    delivery = get_delivery()
    if delivery is None:
        raise RuntimeError("Generation requires a response delivery context")
    continuation: ContinuationSeed | None = None
    for index, candidate in enumerate(candidates):
        if index:
            # A replacement can fail before its provider opens an attempt.
            delivery.begin_attempt()
        target = candidate.target
        record_request_route(target.provider_id, target.provider_model)
        provider_stream: AsyncIterator[StreamOutput] | None = None
        candidate_committed = False
        candidate_failure: ExecutionFailure | None = None
        try:
            try:
                provider_stream = await open_candidate(index, candidate, continuation)
            except ExecutionFailure as failure:
                candidate_failure = failure

            if provider_stream is None and candidate_failure is None:
                raise TypeError("provider stream method must return an async iterator")
            while provider_stream is not None:
                try:
                    chunk = await anext(provider_stream)
                except StopAsyncIteration:
                    break
                except ExecutionFailure as failure:
                    candidate_failure = failure
                    break
                if not any(chunk.frames) and not chunk.completed_calls:
                    await asyncio.sleep(0)
                    continue
                if not candidate_committed:
                    candidate_committed = True
                    if index > 0:
                        trace_fallback_selected(
                            request_id=request_id,
                            generation_id=generation_id,
                            wire_api=wire_api,
                            selected=target,
                            candidate_index=index + 1,
                            candidate_count=len(candidates),
                        )
                yield chunk
        finally:
            if provider_stream is not None:
                active_error = sys.exception()
                preserved_error = active_error or candidate_failure
                await close_stream_input(
                    provider_stream,
                    owner="model_fallback",
                    source="application",
                    preserved_error=preserved_error,
                )

        if candidate_failure is None:
            return
        step = next_step(
            CandidateFailure(
                context=candidate_failure.stream_context,
                retryable=candidate_failure.retryable,
                produced_output=candidate_committed,
            ),
            content_released=delivery.content_released,
            read_prefix=lambda: delivery.prefix,
            seed=continuation,
            has_next=index + 1 < len(candidates),
        )
        match step:
            case Fail():
                raise candidate_failure
            case Handoff():
                delivery.begin_continuation(handoff=True)
                return
            case Fallback(seed=seed, action=action, reason=reason):
                continuation = seed
                if continuation is not None:
                    delivery.begin_continuation()
                trace_fallback_started(
                    request_id=request_id,
                    generation_id=generation_id,
                    wire_api=wire_api,
                    failed=target,
                    selected=candidates[index + 1].target,
                    failure=candidate_failure,
                    candidate_index=index + 2,
                    candidate_count=len(candidates),
                    recovery_action=action,
                    recovery_reason=reason,
                )
