"""Execute ordered model candidates while preserving safe response continuity."""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Literal

from loguru import logger

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.request_outcomes import record_request_route
from free_claude_code.core.stream_delivery import StreamDeliveryState
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import (
    ContinuationSeed,
    RecoveryAction,
    assess_recovery,
)
from free_claude_code.core.trace import close_stream_input, trace_event

from .routing import ProviderModelCandidate, ProviderModelTarget

WireApi = Literal["messages", "responses", "gemini", "chat"]
CandidateStreamOpener = Callable[
    [int, ProviderModelCandidate, ContinuationSeed | None],
    Awaitable[AsyncIterator[StreamOutput]],
]


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
                        _trace_fallback_selected(
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
        recovery_action = "continue" if continuation is not None else "restart"
        recovery_reason = "no_candidate_output"
        context = candidate_failure.stream_context
        if context is None:
            if delivery.content_released:
                raise candidate_failure
        else:
            assessment = assess_recovery(
                content_released=delivery.content_released,
                prefix=delivery.prefix,
                source=context.source,
                retryable=candidate_failure.retryable,
            )
            if assessment.action is RecoveryAction.STOP:
                raise candidate_failure
            if assessment.action is RecoveryAction.HANDOFF:
                delivery.begin_continuation(handoff=True)
                return
            continuation = assessment.seed
            recovery_action = assessment.action.value
            recovery_reason = assessment.reason
        if index + 1 >= len(candidates):
            raise candidate_failure
        if continuation is not None:
            delivery.begin_continuation()
        next_target = candidates[index + 1].target
        _trace_fallback_started(
            request_id=request_id,
            generation_id=generation_id,
            wire_api=wire_api,
            failed=target,
            selected=next_target,
            failure=candidate_failure,
            candidate_index=index + 2,
            candidate_count=len(candidates),
            recovery_action=recovery_action,
            recovery_reason=recovery_reason,
        )


def _trace_fallback_started(
    *,
    request_id: str,
    generation_id: int | None,
    wire_api: WireApi,
    failed: ProviderModelTarget,
    selected: ProviderModelTarget,
    failure: ExecutionFailure,
    candidate_index: int,
    candidate_count: int,
    recovery_action: str,
    recovery_reason: str | None,
) -> None:
    fields: dict[str, object] = {
        "stage": "execution",
        "event": "free_claude_code.model_fallback.started",
        "source": "application",
        "request_id": request_id,
        "wire_api": wire_api,
        "from_provider_model_ref": failed.provider_model_ref,
        "to_provider_model_ref": selected.provider_model_ref,
        "from_api_key_number": failed.api_key_number,
        "to_api_key_number": selected.api_key_number,
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
        "failure_kind": failure.kind.value,
        "status_code": failure.status_code,
        "provider_retryable": failure.retryable,
        "recovery_action": recovery_action,
        "recovery_reason": recovery_reason,
    }
    if generation_id is not None:
        fields["generation_id"] = generation_id
    trace_event(**fields)
    logger.info(
        "Model fallback: request_id={} from={} to={} candidate={}/{} "
        "failure_kind={} status_code={}",
        request_id,
        failed.provider_model_ref,
        selected.provider_model_ref,
        candidate_index,
        candidate_count,
        failure.kind.value,
        failure.status_code,
    )


def _trace_fallback_selected(
    *,
    request_id: str,
    generation_id: int | None,
    wire_api: WireApi,
    selected: ProviderModelTarget,
    candidate_index: int,
    candidate_count: int,
) -> None:
    fields: dict[str, object] = {
        "stage": "execution",
        "event": "free_claude_code.model_fallback.selected",
        "source": "application",
        "request_id": request_id,
        "wire_api": wire_api,
        "selected_provider_model_ref": selected.provider_model_ref,
        "api_key_number": selected.api_key_number,
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
    }
    if generation_id is not None:
        fields["generation_id"] = generation_id
    trace_event(**fields)
