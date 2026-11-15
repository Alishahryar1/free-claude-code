"""The attempt loop every streaming provider transport shares.

This is the shell around ``stream_failure_resolution``: it makes every
admission, credential and delivery call, and asks the pure rules what each
answer means. The order is fixed: repairs (a credential refresh or a corrected
body), then admission's verdict, then a clean restart, then public recovery of
delivered content, and finally the terminal failure.
"""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.trace import close_stream_input, trace_event
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderFailureDecision,
    ProviderOperationKind,
)
from free_claude_code.providers.canonical_failures import ProviderFailureOverride
from free_claude_code.providers.continuation import (
    SourceRecoveryState,
    public_recovery,
    public_stream_failure,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.failure_classification import classify
from free_claude_code.providers.failure_reporting import finalize_provider_failure
from free_claude_code.providers.history_negotiation import RequestHistory
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.recovery_errors import underlying_provider_error
from free_claude_code.providers.request_correction import (
    Correction,
    NextHistoryMode,
    Rejection,
    RequestCorrector,
)
from free_claude_code.providers.stream_failure_resolution import (
    NextStep,
    Repair,
    Stage,
    is_hidden_stop,
    next_step,
    repairs,
)

from .attempt import AttemptContext, TransportAttempt


@dataclass(frozen=True, slots=True)
class FailureReporting:
    """How one transport reports the failures it cannot recover."""

    read_timeout_s: float
    failure_override: ProviderFailureOverride | None = None
    # Request errors that must reach the caller exactly as raised.
    raise_unfinalized: tuple[type[Exception], ...] = ()
    # Report the upstream error body to clients that speak the upstream protocol.
    upstream_evidence: bool = False
    trace_error_message: bool = False
    report_terminal_error: Callable[[Exception], None] | None = None


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """Everything one provider request's attempts need."""

    body: JsonObject
    operation_kind: ProviderOperationKind
    execution: ProviderExecution
    continuation: ContinuationRequest
    corrector: Callable[[], RequestCorrector]
    start: Callable[[AttemptContext], TransportAttempt]
    reporting: FailureReporting
    provider_name: str
    transport: str
    gateway_model: str
    # The credential owner a rejected attempt may refresh, once per request.
    refreshable: RequestEndpoint | None = None
    sent_fields: Mapping[str, Any] = field(default_factory=dict)
    sent_payload: Callable[[], Mapping[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class _Retry:
    body: JsonObject


@dataclass(frozen=True, slots=True)
class _Continue:
    body: JsonObject


@dataclass(frozen=True, slots=True)
class _Handoff:
    pass


@dataclass(frozen=True, slots=True)
class _Fail:
    error: Exception


type _Resolution = _Retry | _Continue | _Handoff | _Fail


async def run_attempts(plan: AttemptPlan) -> AsyncGenerator[StreamOutput]:
    """Stream the first attempt that succeeds, or raise the terminal failure."""
    shell = _Shell(plan)
    execution = plan.execution
    body = plan.body
    operation_kind = plan.operation_kind
    corrections = plan.corrector()
    history = RequestHistory()
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
        attempt = plan.start(AttemptContext(body, operation_kind, history, source))
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
            shell.trace("provider.response.completed", attempt.completed_fields())
            return
        except asyncio.CancelledError, GeneratorExit:
            raise
        except Exception as raw_error:
            if not attempt.resolvable(raw_error):
                raise
            if not _unsafe(raw_error):
                for output in attempt.abandon(raw_error):
                    yield output
            resolution = await shell.resolve(
                raw_error,
                attempt,
                scope,
                corrections=corrections,
                history=history,
                operation_kind=operation_kind,
                source=source,
            )
            match resolution:
                case _Retry(body=next_body):
                    body = next_body
                case _Continue(body=next_body):
                    body = next_body
                    operation_kind = ProviderOperationKind.CONTINUATION
                    corrections = plan.corrector()
                    history = RequestHistory()
                case _Handoff():
                    for output in attempt.handoff():
                        yield output
                    return
                case _Fail(error=failure) if failure is raw_error:
                    raise
                case _Fail(error=failure):
                    raise failure from raw_error
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


class _Shell:
    """Carry out the effects of one request's failure decisions."""

    def __init__(self, plan: AttemptPlan) -> None:
        self._plan = plan
        self._refreshed = False

    async def resolve(
        self,
        raw_error: Exception,
        attempt: TransportAttempt,
        scope: ProviderAttemptScope | None,
        *,
        corrections: RequestCorrector,
        history: RequestHistory,
        operation_kind: ProviderOperationKind,
        source: SourceRecoveryState,
    ) -> _Resolution:
        plan = self._plan
        execution = plan.execution
        delivery = execution.delivery
        content_released = None if delivery is None else delivery.content_released
        error = attempt.resolution_error(raw_error)
        stage = Stage.BEFORE_ATTEMPT if scope is None else Stage.OPEN
        normal_stop_seen = attempt.normal_stop_seen

        if scope is not None:
            found = repairs(
                stage=stage,
                auth_status=classify(error).auth_status,
                can_refresh=(
                    plan.refreshable is not None and plan.refreshable.can_refresh
                ),
                refreshed=self._refreshed,
                content_released=content_released,
                attempt_content_released=(
                    delivery is not None and delivery.attempt_content_released
                ),
                normal_stop_seen=normal_stop_seen,
                continuing=operation_kind is ProviderOperationKind.CONTINUATION,
            )
            if (
                Repair.REFRESH in found
                and plan.refreshable is not None
                and await _authorize(execution, scope.attempt, error)
            ):
                self._refreshed = True
                plan.refreshable.request_refresh()
                return _Retry(attempt.body)
            if Repair.CORRECT in found:
                rejection = Rejection(
                    raw_error,
                    attempt.body,
                    attempt.sent_body,
                    history_source=history.source,
                    history_paths=history.paths,
                )
                corrected = self._correct(corrections.correct(rejection), history)
                if corrected is not None and await _authorize(
                    execution, scope.attempt, error
                ):
                    body, step, note = corrected
                    self._corrected(step, note)
                    return _Retry(body)

        admission: ProviderFailureDecision | None = None
        if scope is not None and not scope.attempt.accepted:
            if is_hidden_stop(
                normal_stop_seen=normal_stop_seen, content_released=content_released
            ):
                await scope.attempt.accept()
            else:
                admission = await scope.fail(
                    error, provider_failure_override=plan.reporting.failure_override
                )
                error = admission.error

        retryable = (
            admission.retryable
            if admission is not None
            else classify(error).retryable_mid_stream
        )
        step = next_step(
            stage=stage,
            admission_retry_allowed=(
                None if admission is None else admission.retry_allowed
            ),
            retryable=retryable,
            content_released=content_released,
            attempts_remaining=execution.attempts_remaining,
            normal_stop_seen=normal_stop_seen,
        )
        if step is NextStep.RETRY:
            self.trace(
                "provider.recovery.early_retry",
                {
                    "attempts_started": execution.attempts_started,
                    "max_attempts": execution.max_attempts,
                },
            )
            return _Retry(attempt.body)
        if step is NextStep.RECOVER:
            if scope is not None and content_released:
                # Free the attempt before public recovery reuses the execution.
                # A terminal failure leaves cleanup to the loop, which closes
                # with the error the caller actually receives.
                await scope.aclose(active_error=error)
            recovered = public_recovery(
                execution,
                body=attempt.body,
                request=plan.continuation,
                retryable=retryable,
                normal_stop_seen=normal_stop_seen,
                source=source,
            )
            if recovered is not None:
                if recovered.body is None:
                    return _Handoff()
                return _Continue(recovered.body)
        return _Fail(self._terminal(error, raw_error, attempt, source))

    def _correct(
        self, decision: Correction | NextHistoryMode | None, history: RequestHistory
    ) -> tuple[JsonObject, str, str | None] | None:
        """Carry out a correction decision: the body to send, and why.

        What the rejection taught is applied whether or not admission allows
        the corrected retry: the evidence came from the upstream either way.
        """
        match decision:
            case None:
                return None
            case NextHistoryMode():
                body = history.next_mode()
                if body is None:
                    return None
                return body, "history_mode", "moved history replay to the next mode"
            case Correction(body=body, step=step, note=note, lesson=lesson):
                if lesson is not None:
                    lesson.apply()
                return body, step, note

    def _corrected(self, step: str, note: str | None) -> None:
        plan = self._plan
        self.trace("provider.request.corrected", {"step": step, "note": note})
        logger.warning(
            "{}_STREAM: correcting the request after an upstream rejection: {} ({})",
            plan.provider_name,
            note or step,
            step,
        )

    def _terminal(
        self,
        error: Exception,
        raw_error: Exception,
        attempt: TransportAttempt,
        source: SourceRecoveryState,
    ) -> Exception:
        plan = self._plan
        reporting = plan.reporting
        if isinstance(error, reporting.raise_unfinalized):
            return error
        reported = underlying_provider_error(error)
        if reporting.report_terminal_error is not None:
            reporting.report_terminal_error(reported)
        finalized = finalize_provider_failure(
            error,
            provider_name=plan.provider_name,
            read_timeout_s=reporting.read_timeout_s,
            request_id=plan.execution.request_id,
            provider_failure_override=reporting.failure_override,
            upstream_evidence=reporting.upstream_evidence,
        )
        fields: dict[str, Any] = {
            "exc_type": type(reported).__name__,
            "failure_kind": finalized.kind.value,
            "status_code": finalized.status_code,
            "provider_retryable": finalized.retryable,
        }
        if reporting.trace_error_message:
            fields["error_message"] = finalized.message
        self.trace("provider.response.error", fields)
        if plan.execution.delivery is None:
            return finalized
        return public_stream_failure(
            finalized,
            source=source,
            body=attempt.body,
            protocol=plan.continuation.protocol,
            normal_stop_seen=attempt.normal_stop_seen,
            responses_failure_payload=attempt.failure_payload(raw_error, finalized),
        )

    def trace(self, event: str, fields: Mapping[str, Any]) -> None:
        plan = self._plan
        trace_event(
            stage="provider",
            event=event,
            source="provider",
            provider=plan.provider_name,
            request_id=plan.execution.request_id,
            transport=plan.transport,
            **fields,
        )


async def _authorize(
    execution: ProviderExecution, attempt: ProviderAttempt, error: Exception
) -> bool:
    """Admission's permission to repeat this attempt with a repaired request."""
    if attempt.accepted:
        return execution.can_attempt
    return await attempt.correct(error) is ProviderCorrectionAction.RETRY


def _unsafe(error: Exception) -> bool:
    """A failure whose source is unsafe owes the client no buffered frames."""
    return bool(
        isinstance(error, ExecutionFailure)
        and error.stream_context
        and error.stream_context.source.unsafe_reason
    )
