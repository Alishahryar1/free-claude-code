"""One failed stream attempt resolves to exactly one next action."""

from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from loguru import logger

from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.trace import TRACE_PAYLOAD_BINDING
from free_claude_code.providers.admission import (
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.continuation import PublicRecovery, SourceRecoveryState
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.request_recovery import RequestRecovery
from free_claude_code.providers.stream_failure_resolution import (
    Continue,
    Fail,
    Handoff,
    Retry,
    StreamFailure,
    StreamFailurePolicy,
    StreamFailureResolver,
)
from free_claude_code.providers.stream_failure_resolution import (
    resolver as resolver_module,
)
from free_claude_code.providers.stream_failure_resolution.decision import (
    NextStep,
    next_step,
)
from tests.providers.support import immediate_admission

BODY: JsonObject = {"model": "upstream", "messages": []}
TIMEOUT = httpx.ReadTimeout("quiet upstream")
REJECTED = ExecutionFailure(FailureKind.INVALID_REQUEST, 400, "bad request", False)


@dataclass(frozen=True, slots=True)
class Row:
    attempted: bool = True
    admission: bool | None = None
    retryable: bool = True
    released: bool | None = False
    attempts_remaining: int = 1
    normal_stop: bool = False


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (Row(attempted=False), NextStep.RECOVER),
        (Row(admission=True), NextStep.RETRY),
        (Row(admission=False), NextStep.RECOVER),
        # Admission's verdict is final even when delivery evidence would allow it.
        (Row(admission=False, retryable=True), NextStep.RECOVER),
        (Row(), NextStep.RETRY),
        (Row(retryable=False), NextStep.RECOVER),
        (Row(released=True), NextStep.RECOVER),
        (Row(released=None), NextStep.RECOVER),
        (Row(attempts_remaining=0), NextStep.RECOVER),
        (Row(normal_stop=True), NextStep.RECOVER),
    ],
)
def test_next_step(row: Row, expected: NextStep) -> None:
    assert (
        next_step(
            attempted=row.attempted,
            admission_retry_allowed=row.admission,
            retryable=row.retryable,
            content_released=row.released,
            attempts_remaining=row.attempts_remaining,
            normal_stop_seen=row.normal_stop,
        )
        is expected
    )


@pytest.fixture
def traces() -> Iterator[list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        payload = message.record["extra"].get(TRACE_PAYLOAD_BINDING)
        if isinstance(payload, dict):
            payloads.append(payload)

    handle = logger.add(sink, level="DEBUG")
    try:
        yield payloads
    finally:
        logger.remove(handle)


def _names(traces: list[dict[str, Any]]) -> list[str]:
    return [
        payload["event"]
        for payload in traces
        if payload["event"]
        in {"provider.recovery.early_retry", "provider.response.error"}
    ]


@dataclass(slots=True)
class Harness:
    execution: ProviderExecution
    delivery: StreamDeliveryState
    resolver: StreamFailureResolver

    async def scope(self, *, accepted: bool = False) -> ProviderAttemptScope:
        attempt = await self.execution.open_attempt(ProviderOperationKind.GENERATION)
        if accepted:
            await attempt.accept()
        return ProviderAttemptScope(attempt, provider_name="TEST", request_id="r")

    def failure(
        self,
        error: Exception,
        scope: ProviderAttemptScope | None,
        *,
        normal_stop_seen: bool = False,
        correction: JsonObject | None = None,
    ) -> StreamFailure:
        return StreamFailure(
            error=error,
            scope=scope,
            body=BODY,
            operation_kind=ProviderOperationKind.GENERATION,
            normal_stop_seen=normal_stop_seen,
            source=SourceRecoveryState(self.execution),
            propose_correction=lambda: correction,
        )


@asynccontextmanager
async def _harness(*, max_attempts: int = 3, **policy: Any) -> AsyncGenerator[Harness]:
    delivery = StreamDeliveryState()
    with bind_stream_delivery(delivery):
        execution = immediate_admission(max_attempts=max_attempts).start_execution(
            request_id="r"
        )
    resolver = StreamFailureResolver(
        StreamFailurePolicy(
            provider_name="TEST", read_timeout_s=1, request_id="r", **policy
        ),
        recovery=RequestRecovery(execution),
        continuation=ContinuationRequest("chat"),
    )
    yield Harness(execution, delivery, resolver)


@pytest.mark.asyncio
async def test_retryable_failure_before_acceptance_retries_by_admission(traces):
    async with _harness() as harness:
        scope = await harness.scope()
        resolution = await harness.resolver.resolve(harness.failure(TIMEOUT, scope))
        await scope.aclose(active_error=None)

    assert resolution == Retry(BODY)
    assert _names(traces) == ["provider.recovery.early_retry"]


@pytest.mark.asyncio
async def test_rejected_failure_is_finalized_once(traces):
    async with _harness() as harness:
        scope = await harness.scope()
        resolution = await harness.resolver.resolve(harness.failure(REJECTED, scope))

    assert isinstance(resolution, Fail)
    assert isinstance(resolution.error, ExecutionFailure)
    assert resolution.error.status_code == 400
    assert _names(traces) == ["provider.response.error"]
    error_trace = next(p for p in traces if p["event"] == "provider.response.error")
    assert error_trace["transport"] == "chat"
    assert "error_message" not in error_trace


@pytest.mark.asyncio
async def test_exhausted_budget_is_final_before_acceptance(traces):
    async with _harness(max_attempts=1) as harness:
        scope = await harness.scope()
        resolution = await harness.resolver.resolve(harness.failure(TIMEOUT, scope))

    assert isinstance(resolution, Fail)
    assert _names(traces) == ["provider.response.error"]


@pytest.mark.asyncio
async def test_hidden_stop_accepts_and_never_restarts():
    async with _harness() as harness:
        scope = await harness.scope()
        resolution = await harness.resolver.resolve(
            harness.failure(TIMEOUT, scope, normal_stop_seen=True)
        )

    assert scope.attempt.accepted
    assert isinstance(resolution, Fail)


@pytest.mark.asyncio
async def test_accepted_undelivered_attempt_restarts_cleanly():
    async with _harness() as harness:
        scope = await harness.scope(accepted=True)
        resolution = await harness.resolver.resolve(harness.failure(TIMEOUT, scope))
        await scope.aclose(active_error=None)

    assert resolution == Retry(BODY)


@pytest.mark.asyncio
async def test_failure_without_an_attempt_is_never_retried():
    async with _harness() as harness:
        resolution = await harness.resolver.resolve(harness.failure(TIMEOUT, None))

    assert isinstance(resolution, Fail)


@pytest.mark.asyncio
async def test_request_correction_precedes_admission():
    corrected: JsonObject = {**BODY, "corrected": True}
    async with _harness() as harness:
        scope = await harness.scope()
        resolution = await harness.resolver.resolve(
            harness.failure(REJECTED, scope, correction=corrected)
        )
        await scope.aclose(active_error=None)

    assert resolution == Retry(corrected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy", [{"finalize": False}, {"raise_unfinalized": (ExecutionFailure,)}]
)
async def test_unfinalized_errors_reach_the_caller_unchanged(policy, traces):
    async with _harness(**policy) as harness:
        scope = await harness.scope()
        resolution = await harness.resolver.resolve(harness.failure(REJECTED, scope))

    assert resolution == Fail(REJECTED)
    assert _names(traces) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recovered", "expected"),
    [
        (PublicRecovery({"continued": True}), Continue({"continued": True})),
        (PublicRecovery(), Handoff()),
    ],
)
async def test_delivered_content_recovers_publicly(monkeypatch, recovered, expected):
    monkeypatch.setattr(resolver_module, "public_recovery", lambda *_, **__: recovered)
    async with _harness() as harness:
        harness.delivery.release_content()
        scope = await harness.scope(accepted=True)
        resolution = await harness.resolver.resolve(harness.failure(TIMEOUT, scope))

    assert resolution == expected


@pytest.mark.asyncio
async def test_terminal_report_and_message_trace_are_opt_in(traces):
    reported: list[Exception] = []
    async with _harness(
        report_terminal_error=reported.append, trace_error_message=True
    ) as harness:
        scope = await harness.scope()
        await harness.resolver.resolve(harness.failure(REJECTED, scope))

    assert reported == [REJECTED]
    error_trace = next(p for p in traces if p["event"] == "provider.response.error")
    assert error_trace["error_message"]
