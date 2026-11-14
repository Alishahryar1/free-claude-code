"""The attempt loop every streaming provider transport shares."""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, assert_never

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.trace import close_stream_input, trace_event
from free_claude_code.providers.admission import (
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.continuation import SourceRecoveryState
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.request_correction import (
    Rejection,
    RequestCorrector,
)
from free_claude_code.providers.stream_failure_resolution import (
    Continue,
    Fail,
    Handoff,
    Retry,
    StreamFailure,
    StreamFailureResolver,
)

from .attempt import AttemptContext, TransportAttempt


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """Everything one provider request's attempts need."""

    body: JsonObject
    operation_kind: ProviderOperationKind
    execution: ProviderExecution
    resolver: StreamFailureResolver
    corrector: Callable[[], RequestCorrector]
    start: Callable[[AttemptContext], TransportAttempt]
    provider_name: str
    transport: str
    gateway_model: str
    sent_fields: Mapping[str, Any] = field(default_factory=dict)
    sent_payload: Callable[[], Mapping[str, Any]] | None = None


async def run_attempts(plan: AttemptPlan) -> AsyncGenerator[StreamOutput]:
    """Stream the first attempt that succeeds, or raise the terminal failure.

    Credentials resolve before admission opens an attempt, so a credential
    failure is never counted as a provider call. Every failure of a resolvable
    attempt goes through the resolver, whose decision this loop applies.
    """
    execution = plan.execution
    body = plan.body
    operation_kind = plan.operation_kind
    corrections = plan.corrector()
    trace_event(
        plan.sent_payload,
        stage="provider",
        event="provider.request.sent",
        source="provider",
        provider=plan.provider_name,
        request_id=execution.request_id,
        execution_id=execution.execution_id,
        gateway_model=plan.gateway_model,
        downstream_model=body.get("model"),
        transport=plan.transport,
        **plan.sent_fields,
    )
    source: SourceRecoveryState | None = None
    while execution.can_attempt:
        source = SourceRecoveryState(execution, previous=source)
        attempt = plan.start(AttemptContext(body, operation_kind, corrections, source))
        scope: ProviderAttemptScope | None = None
        stream: AsyncIterator[StreamOutput] | None = None
        try:
            await attempt.prepare()
            scope = ProviderAttemptScope(
                await execution.open_attempt(operation_kind),
                provider_name=plan.provider_name,
                request_id=execution.request_id,
            )
            stream = attempt.stream(scope)
            async for output in stream:
                yield output
            trace_event(
                stage="provider",
                event="provider.response.completed",
                source="provider",
                provider=plan.provider_name,
                request_id=execution.request_id,
                transport=plan.transport,
                **attempt.completed_fields(),
            )
            return
        except asyncio.CancelledError, GeneratorExit:
            raise
        except Exception as raw_error:
            if not attempt.resolvable(raw_error):
                raise
            if not _unsafe(raw_error):
                for output in attempt.abandon(raw_error):
                    yield output
            resolution = await plan.resolver.resolve(
                StreamFailure(
                    error=attempt.resolution_error(raw_error),
                    scope=scope,
                    body=attempt.body,
                    operation_kind=operation_kind,
                    normal_stop_seen=attempt.normal_stop_seen,
                    source=source,
                    propose_correction=partial(
                        corrections.correct,
                        Rejection(raw_error, attempt.body, attempt.sent_body),
                    ),
                    failure_payload=partial(attempt.failure_payload, raw_error),
                )
            )
            match resolution:
                case Retry(body=next_body):
                    body = next_body
                case Continue(body=next_body):
                    body = next_body
                    operation_kind = ProviderOperationKind.CONTINUATION
                    corrections = plan.corrector()
                case Handoff():
                    for output in attempt.handoff():
                        yield output
                    return
                case Fail(error=failure) if failure is raw_error:
                    raise
                case Fail(error=failure):
                    raise failure from raw_error
                case _:
                    assert_never(resolution)
        finally:
            active_error = sys.exception()
            try:
                # A consumer that closes early leaves the transport's reader
                # suspended; finish its cleanup before admission is released.
                if stream is not None:
                    await close_stream_input(
                        stream,
                        owner="attempt_loop",
                        source="provider",
                        preserved_error=active_error,
                    )
            finally:
                if scope is not None:
                    await scope.aclose(active_error=active_error)
    if execution.last_failure is not None:
        raise execution.last_failure
    raise RuntimeError(f"{plan.transport} execution ended without a terminal result")


def _unsafe(error: Exception) -> bool:
    """A failure whose source is unsafe owes the client no buffered frames."""
    return bool(
        isinstance(error, ExecutionFailure)
        and error.stream_context
        and error.stream_context.source.unsafe_reason
    )
