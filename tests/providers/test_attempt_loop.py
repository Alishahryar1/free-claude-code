"""The attempt loop every streaming provider transport shares."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest
from loguru import logger

from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import (
    ContinuationSeed,
    DeliveredPrefix,
    SourceRecoverySnapshot,
    StreamFailureContext,
)
from free_claude_code.core.trace import TRACE_PAYLOAD_BINDING
from free_claude_code.providers.admission import (
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.attempt_loop import (
    AttemptContext,
    AttemptPlan,
    TransportAttempt,
    run_attempts,
)
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.request_correction import (
    CorrectionStep,
    Rejection,
    RequestCorrector,
)
from free_claude_code.providers.request_recovery import RequestRecovery
from free_claude_code.providers.stream_failure_resolution import (
    StreamFailurePolicy,
    StreamFailureResolver,
)
from tests.providers.support import immediate_admission

BODY: JsonObject = {"model": "m", "n": 0, "messages": []}


class _Unresolvable(Exception):
    pass


class _Unfinalized(Exception):
    pass


def _failure(status: int = 503, *, unsafe: bool = False) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=status,
        message=f"failed with {status}",
        retryable=status >= 500,
        stream_context=(
            StreamFailureContext(SourceRecoverySnapshot(unsafe_reason="native"))
            if unsafe
            else None
        ),
    )


@dataclass(frozen=True)
class Step:
    """How one scripted attempt behaves."""

    prepare_error: Exception | None = None
    frames: tuple[str, ...] = ()
    error: Exception | None = None


@dataclass
class Record:
    bodies: list[JsonObject] = field(default_factory=list)
    kinds: list[ProviderOperationKind] = field(default_factory=list)
    attempts_at_prepare: list[int] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)

    def events(self, name: str) -> list[dict[str, Any]]:
        return [payload for payload in self.traces if payload["event"] == name]

    @property
    def outcomes(self) -> list[str]:
        return [
            payload["outcome"] for payload in self.events("provider.attempt.resolved")
        ]


@pytest.fixture
def record() -> Iterator[Record]:
    result = Record()

    def sink(message: Any) -> None:
        payload = message.record["extra"].get(TRACE_PAYLOAD_BINDING)
        if isinstance(payload, dict):
            result.traces.append(payload)

    handle = logger.add(sink, level="DEBUG")
    try:
        yield result
    finally:
        logger.remove(handle)


async def _run(
    record: Record,
    *steps: Step,
    correction: CorrectionStep | None = None,
    max_attempts: int = 3,
    delivery: StreamDeliveryState | None = None,
) -> tuple[list[str], BaseException | None]:
    admission = immediate_admission(max_attempts=max_attempts)
    with bind_stream_delivery(delivery) if delivery else nullcontext():
        execution = admission.start_execution(request_id="loop", model="m")
    script = iter(steps)

    class Scripted(TransportAttempt):
        def __init__(self, context: AttemptContext) -> None:
            super().__init__(context)
            self.streamed = False
            self.step = next(script)
            record.bodies.append(context.body)
            record.kinds.append(context.operation_kind)

        async def prepare(self) -> None:
            record.attempts_at_prepare.append(execution.attempts_started)
            if self.step.prepare_error is not None:
                raise self.step.prepare_error
            self.sent_body = {**self.body, "sent": True}

        async def stream(
            self, scope: ProviderAttemptScope
        ) -> AsyncIterator[StreamOutput]:
            self.streamed = True
            for frame in self.step.frames:
                if not scope.attempt.accepted:
                    await scope.attempt.accept()
                yield StreamOutput((frame,))
            if self.step.error is not None:
                raise self.step.error

        def resolvable(self, error: Exception) -> bool:
            return not isinstance(error, _Unresolvable)

        def abandon(self, error: Exception) -> tuple[StreamOutput, ...]:
            return (StreamOutput(("abandoned",)),) if self.streamed else ()

    frames: list[str] = []
    raised: BaseException | None = None
    try:
        async for output in run_attempts(_plan(execution, Scripted, correction)):
            frames.extend(output.frames)
    except Exception as error:
        raised = error
    return frames, raised


def _plan(
    execution: ProviderExecution,
    start: Callable[[AttemptContext], TransportAttempt],
    correction: CorrectionStep | None = None,
) -> AttemptPlan:
    return AttemptPlan(
        body=BODY,
        operation_kind=ProviderOperationKind.GENERATION,
        execution=execution,
        resolver=StreamFailureResolver(
            StreamFailurePolicy(
                provider_name="TEST",
                read_timeout_s=30,
                request_id="loop",
                raise_unfinalized=(_Unfinalized,),
            ),
            recovery=RequestRecovery(execution, endpoint=None),
            continuation=ContinuationRequest("messages"),
        ),
        corrector=lambda: RequestCorrector(
            "messages", steps=(correction,) if correction is not None else ()
        ),
        start=start,
        provider_name="TEST",
        transport="messages",
        gateway_model="public",
    )


@pytest.mark.asyncio
async def test_success_prepares_before_admission_and_traces_its_lifecycle(record):
    frames, raised = await _run(record, Step(frames=("a", "b")))

    assert (frames, raised) == (["a", "b"], None)
    assert record.attempts_at_prepare == [0]
    assert record.outcomes == ["accepted"]
    sent = record.events("provider.request.sent")
    assert [(event["transport"], event["downstream_model"]) for event in sent] == [
        ("messages", "m")
    ]
    assert len(record.events("provider.response.completed")) == 1


@pytest.mark.asyncio
async def test_preparation_failure_opens_no_attempt_and_is_terminal(record):
    frames, raised = await _run(record, Step(prepare_error=_failure(503)), Step())

    assert isinstance(raised, ExecutionFailure) and raised.status_code == 503
    assert frames == []
    assert record.attempts_at_prepare == [0]
    assert record.outcomes == []


@pytest.mark.asyncio
async def test_retryable_failure_before_acceptance_retries_the_same_body(record):
    frames, raised = await _run(record, Step(error=_failure(503)), Step(frames=("ok",)))

    assert (frames, raised) == (["abandoned", "ok"], None)
    assert record.bodies == [BODY] * 2
    assert record.attempts_at_prepare == [0, 1]
    assert record.outcomes == ["retryable_failure", "accepted"]


@pytest.mark.asyncio
async def test_corrected_rejection_retries_with_the_corrected_body(record):
    rejections: list[Rejection] = []

    def propose(rejection: Rejection) -> JsonObject:
        rejections.append(rejection)
        return {**rejection.body, "n": 1}

    frames, raised = await _run(
        record,
        Step(error=_failure(400)),
        Step(frames=("ok",)),
        correction=CorrectionStep("fix", propose),
    )

    assert (frames, raised) == (["abandoned", "ok"], None)
    assert record.bodies == [BODY, {**BODY, "n": 1}]
    assert [rejection.sent_body for rejection in rejections] == [{**BODY, "sent": True}]


@pytest.mark.asyncio
async def test_exhausted_attempts_raise_the_resolved_failure(record):
    frames, raised = await _run(
        record, Step(error=_failure(503)), Step(error=_failure(503)), max_attempts=2
    )

    assert isinstance(raised, ExecutionFailure) and raised.status_code == 503
    assert frames == ["abandoned", "abandoned"]
    assert len(record.events("provider.response.error")) == 1


@pytest.mark.asyncio
async def test_unsafe_failure_owes_no_abandoned_frames(record):
    frames, raised = await _run(
        record, Step(frames=("a",), error=_failure(503, unsafe=True)), max_attempts=1
    )

    assert isinstance(raised, ExecutionFailure)
    assert frames == ["a"]


@pytest.mark.asyncio
async def test_unresolvable_error_is_raised_without_resolution(record):
    error = _Unresolvable("local check")
    frames, raised = await _run(record, Step(error=error), Step(frames=("ok",)))

    assert raised is error
    assert frames == []
    assert record.events("provider.response.error") == []
    assert len(record.bodies) == 1


@pytest.mark.asyncio
async def test_unfinalized_failure_is_raised_as_itself(record):
    error = _Unfinalized("caller error")
    _, raised = await _run(record, Step(error=error), max_attempts=1)

    assert raised is error
    assert raised.__cause__ is None


def _delivered(text: str) -> StreamDeliveryState:
    state = StreamDeliveryState()
    state.bind_prefix(
        lambda: DeliveredPrefix(
            text=text,
            thinking="",
            has_calls=False,
            eligible=True,
            can_handoff=True,
            unsafe_reason=None,
            continuation=ContinuationSeed(
                "messages", ({"type": "text", "text": text},)
            ),
        )
    )
    state.release_content()
    return state


@pytest.mark.asyncio
async def test_early_close_finishes_transport_cleanup_before_releasing_admission():
    events: list[str] = []

    class Resource:
        async def aclose(self) -> None:
            events.append("scope resource closed")

    class Held(TransportAttempt):
        async def stream(
            self, scope: ProviderAttemptScope
        ) -> AsyncIterator[StreamOutput]:
            scope.retain(Resource())
            await scope.attempt.accept()
            try:
                yield StreamOutput(("a",))
                yield StreamOutput(("b",))
            finally:
                await asyncio.sleep(0)
                events.append("transport cleanup finished")

    execution = immediate_admission().start_execution(request_id="close", model="m")
    outputs = run_attempts(_plan(execution, Held))
    assert (await anext(outputs)).frames == ("a",)
    await outputs.aclose()

    assert events == ["transport cleanup finished", "scope resource closed"]
