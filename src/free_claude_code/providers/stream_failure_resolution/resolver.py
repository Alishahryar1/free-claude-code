"""Resolve one failed stream attempt into a single next action.

The order is fixed: request recovery (credential refresh or a corrected
body), then admission's verdict, then a clean restart, then public recovery of
delivered content, and finally the terminal failure.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderFailureDecision,
    ProviderOperationKind,
)
from free_claude_code.providers.canonical_failures import ProviderFailureOverride
from free_claude_code.providers.continuation import (
    SourceRecoveryState,
    public_recovery,
    public_stream_failure,
)
from free_claude_code.providers.failure_classification import classify
from free_claude_code.providers.failure_reporting import finalize_provider_failure
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.recovery_errors import underlying_provider_error
from free_claude_code.providers.request_recovery import RequestRecovery

from .decision import NextStep, is_hidden_stop, next_step
from .resolution import Continue, Fail, Handoff, Retry, StreamFailureResolution


@dataclass(frozen=True, slots=True)
class StreamFailurePolicy:
    """How one transport reports the failures it cannot recover."""

    provider_name: str
    read_timeout_s: float
    request_id: str | None
    failure_override: ProviderFailureOverride | None = None
    # False leaves the error for an outer owner to finalize.
    finalize: bool = True
    # Request errors that must reach the caller exactly as raised.
    raise_unfinalized: tuple[type[Exception], ...] = ()
    trace_error_message: bool = False
    report_terminal_error: Callable[[Exception], None] | None = None


@dataclass(frozen=True, slots=True)
class StreamFailure:
    """One attempt's failure and the evidence it left behind."""

    error: Exception
    scope: ProviderAttemptScope | None
    body: JsonObject
    operation_kind: ProviderOperationKind
    normal_stop_seen: bool
    source: SourceRecoveryState
    propose_correction: Callable[[], JsonObject | None]
    failure_payload: Callable[[ExecutionFailure], JsonObject | None] | None = None


class StreamFailureResolver:
    """Own the failure rules of one request's stream attempts."""

    def __init__(
        self,
        policy: StreamFailurePolicy,
        *,
        recovery: RequestRecovery,
        continuation: ContinuationRequest,
    ) -> None:
        self._policy = policy
        self._recovery = recovery
        self._continuation = continuation

    async def resolve(self, failure: StreamFailure) -> StreamFailureResolution:
        error = failure.error
        scope = failure.scope
        execution = self._recovery.execution
        delivery = execution.delivery
        content_released = None if delivery is None else delivery.content_released

        if scope is not None:
            corrected = await self._recovery.retry_request(
                error,
                classify(error).auth_status,
                scope.attempt,
                failure.body,
                operation_kind=failure.operation_kind,
                normal_stop_seen=failure.normal_stop_seen,
                propose_correction=failure.propose_correction,
            )
            if corrected is not None:
                return Retry(corrected)

        admission: ProviderFailureDecision | None = None
        if scope is not None and not scope.attempt.accepted:
            if is_hidden_stop(
                normal_stop_seen=failure.normal_stop_seen,
                content_released=content_released,
            ):
                await scope.attempt.accept()
            else:
                admission = await scope.fail(
                    error, provider_failure_override=self._policy.failure_override
                )
                error = admission.error

        retryable = (
            admission.retryable
            if admission is not None
            else classify(error).retryable_mid_stream
        )
        step = next_step(
            attempted=scope is not None,
            admission_retry_allowed=(
                None if admission is None else admission.retry_allowed
            ),
            retryable=retryable,
            content_released=content_released,
            attempts_remaining=execution.attempts_remaining,
            normal_stop_seen=failure.normal_stop_seen,
        )
        if step is NextStep.RETRY:
            self._trace(
                "provider.recovery.early_retry",
                {
                    "attempts_started": execution.attempts_started,
                    "max_attempts": execution.max_attempts,
                },
            )
            return Retry(failure.body)

        if scope is not None and content_released:
            # Free the attempt before public recovery reuses the execution. A
            # terminal failure leaves cleanup to the transport, which closes
            # with the error the caller actually receives.
            await scope.aclose(active_error=error)
        recovered = public_recovery(
            execution,
            body=failure.body,
            request=self._continuation,
            retryable=retryable,
            normal_stop_seen=failure.normal_stop_seen,
            source=failure.source,
        )
        if recovered is not None:
            return Handoff() if recovered.body is None else Continue(recovered.body)
        return Fail(self._terminal(error, failure))

    def _terminal(self, error: Exception, failure: StreamFailure) -> Exception:
        policy = self._policy
        if not policy.finalize or isinstance(error, policy.raise_unfinalized):
            return error
        reported = underlying_provider_error(error)
        if policy.report_terminal_error is not None:
            policy.report_terminal_error(reported)
        finalized = finalize_provider_failure(
            error,
            provider_name=policy.provider_name,
            read_timeout_s=policy.read_timeout_s,
            request_id=policy.request_id,
            provider_failure_override=policy.failure_override,
        )
        fields: dict[str, Any] = {
            "exc_type": type(reported).__name__,
            "failure_kind": finalized.kind.value,
            "status_code": finalized.status_code,
            "provider_retryable": finalized.retryable,
        }
        if policy.trace_error_message:
            fields["error_message"] = finalized.message
        self._trace("provider.response.error", fields)
        if self._recovery.execution.delivery is None:
            return finalized
        return public_stream_failure(
            finalized,
            source=failure.source,
            body=failure.body,
            protocol=self._continuation.protocol,
            normal_stop_seen=failure.normal_stop_seen,
            responses_failure_payload=(
                failure.failure_payload(finalized)
                if failure.failure_payload is not None
                else None
            ),
        )

    def _trace(self, event: str, fields: dict[str, Any]) -> None:
        trace_event(
            stage="provider",
            event=event,
            source="provider",
            provider=self._policy.provider_name,
            request_id=self._policy.request_id,
            transport=self._continuation.protocol,
            **fields,
        )
