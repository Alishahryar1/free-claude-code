"""Candidate setup, admission and rejection cross real provider boundaries."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from free_claude_code.application.recovery import RecoveryCoordinator
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.recovery_stream import MessagesRecoveryWriter
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesRecoveryWriter,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.recovery import (
    AttemptFailure,
    CandidateIncompatible,
    RecoveryCheckpoint,
)
from free_claude_code.core.stream_events import (
    DecodedStreamEvent,
    RequestOutcome,
    StreamEvent,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecutionState,
    ProviderOperationKind,
)
from free_claude_code.providers.anthropic import AnthropicProvider
from free_claude_code.providers.candidate_setup import CandidateSetup
from free_claude_code.providers.failure_policy import ProviderRecoveryDeferred
from free_claude_code.providers.stream_candidate import StreamCandidate
from tests.providers.support import make_provider_config, stream_responses
from tests.providers.test_anthropic_messages_transport import _events
from tests.providers.test_history_transports import _events_for
from tests.providers.test_opencode import (
    _catalog_payload,
    _provider_with_wire_transports,
)
from tests.providers.test_request_recovery_transitions import (
    _history_error,
    _request,
    _stream,
    _transport,
)


def _timeout(_):
    return ExecutionFailure(FailureKind.TIMEOUT, 504, "No provider progress", False)


def _admission(**kwargs):
    return ProviderAdmissionController(
        provider_name="lifetime",
        rate_limit=1000,
        max_concurrency=1,
        max_attempts=3,
        base_delay=0,
        max_delay=0,
        jitter=0,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_rejection_cannot_authorize_removing_original_native_history(protocol):
    async with _transport(
        protocol,
        lambda ordinal: (
            (400, _history_error(protocol))
            if ordinal == 1
            else (200, _events_for(protocol))
        ),
    ) as (provider, endpoint, bodies, wires, admission):
        with pytest.raises(ExecutionFailure):
            _ = [
                event async for event in _stream(provider, endpoint, _request(protocol))
            ]
        assert len(bodies) == 1 and "opaque-original" in json.dumps(bodies[0])
        assert all(wire.close_calls == 1 for wire in wires)
        assert admission._active_attempts == 0


@pytest.mark.asyncio
async def test_cold_same_provider_metadata_does_not_wait_behind_failed_generation():
    calls = []

    def handle(request):
        body = json.loads(request.content) if request.method == "POST" else {}
        calls.append((request.method, request.url.path, body.get("model")))
        if request.method == "GET":
            return httpx.Response(200, json={"id": request.url.path.rsplit("/", 1)[1]})
        if body["model"] == "first":
            return httpx.Response(503, json={"error": {"type": "overloaded_error"}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in _events("Recovered")
            ),
        )

    admission = _admission()
    provider = AnthropicProvider(
        make_provider_config("key", "https://provider.invalid/v1"),
        admission=admission,
        transport=httpx.MockTransport(handle),
    )
    try:
        await provider.model_record("first")

        async def opener(index, target):
            return provider.open_messages(
                MessagesRequest.model_validate(
                    {
                        "model": target.provider_model,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )

        stream = RecoveryCoordinator(
            candidates=tuple(
                ProviderModelTarget("anthropic", model, f"anthropic/{model}")
                for model in ("first", "second")
            ),
            opener=opener,
            writer=MessagesRecoveryWriter(model="public", input_tokens=0),
            progress_timeout_seconds=0.5,
            timeout_failure=_timeout,
            request_id="cold-recovery",
        ).stream()
        output = "".join([chunk async for chunk in stream])
        assert "Recovered" in output and "event: error" not in output
        assert calls == [
            ("GET", "/v1/models/first", None),
            ("POST", "/v1/messages", "first"),
            ("GET", "/v1/models/second", None),
            ("POST", "/v1/messages", "second"),
        ]
        assert admission._active_attempts == 0
        assert admission._episode is None
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("hosted", [False, True])
async def test_explicit_auth_rejection_allows_one_refresh_with_hosted_tools(hosted):
    def respond(ordinal):
        return (
            (401, {"type": "authentication_error", "message": "expired"})
            if ordinal == 1
            else (200, _events_for("responses"))
        )

    async with _transport("responses", respond) as (
        provider,
        endpoint,
        bodies,
        wires,
        admission,
    ):
        request = OpenAIResponsesRequest(
            model="upstream",
            input="hi",
            tools=[{"type": "web_search"}] if hosted else None,
        )

        async def opener(index, target):
            return provider.open_responses(
                request,
                input_tokens=0,
                request_id="rejection",
                response_model="public",
                reasoning=ReasoningPolicy.provider_default(),
                endpoint_context=endpoint,
            )

        stream = RecoveryCoordinator(
            candidates=(ProviderModelTarget("test", "upstream", "test/upstream"),),
            opener=opener,
            writer=ResponsesRecoveryWriter(model="public", input_tokens=0),
            progress_timeout_seconds=3,
            timeout_failure=_timeout,
            request_id="rejection",
        ).stream()
        output = "".join([chunk async for chunk in stream])
        assert "response.completed" in output and "response.failed" not in output
        assert len(bodies) == 2 and endpoint.calls == [False, True]
        assert bodies[0] == bodies[1]
        assert admission._active_attempts == 0
        assert all(wire.close_calls == 1 for wire in wires)


@pytest.mark.asyncio
async def test_discovery_cancellation_after_failure_releases_membership():
    admission = _admission()
    execution = admission.start_execution()

    async def fail_once():
        task = asyncio.current_task()
        assert task is not None
        asyncio.get_running_loop().call_soon(task.cancel)
        raise httpx.ReadError("interrupted discovery")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(
            execution.run_call(
                fail_once, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
            )
        )
    assert execution.state is ProviderExecutionState.ABANDONED
    assert execution.attempts_started == 1
    assert admission._active_attempts == 0
    assert admission._episode is not None
    assert admission._episode.leader is None and not admission._episode.waiters


@pytest.mark.asyncio
async def test_suspension_keeps_budget_and_cooldown_for_later_resumption():
    admission = _admission()
    execution = admission.start_execution()
    attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
    error = httpx.ReadError("lost upstream")
    await attempt.fail(error)
    await attempt.aclose()
    episode = admission._episode
    assert episode is not None
    ready_at = episode.ready_at
    await execution.suspend()
    assert execution.state is ProviderExecutionState.ACTIVE
    assert execution.attempts_remaining == 2
    assert episode.ready_at == ready_at and episode.last_error is error
    assert episode.leader is None and not episode.waiters
    resumed = await execution.open_attempt(ProviderOperationKind.GENERATION)
    assert execution.attempts_started == 2
    await resumed.accept()
    await resumed.aclose()
    await execution.aclose()


@pytest.mark.asyncio
async def test_ambiguous_hosted_disconnect_cannot_refresh_or_regenerate():
    async with _transport(
        "responses", lambda _: (200, _events_for("responses")[:1])
    ) as (provider, endpoint, bodies, wires, admission):
        request = OpenAIResponsesRequest(
            model="model", input="hello", tools=[{"type": "web_search"}]
        )
        output = "".join(
            [
                event
                async for event in stream_responses(
                    provider,
                    request,
                    endpoint_context=endpoint,
                    request_id="ambiguous",
                    response_model="public",
                    reasoning=ReasoningPolicy.provider_default(),
                )
            ]
        )
        assert "response.failed" in output and "response.completed" not in output
        assert len(bodies) == 1 and endpoint.calls == [False]
        assert admission._active_attempts == 0
        assert all(wire.close_calls == 1 for wire in wires)


@pytest.mark.asyncio
async def test_discovery_deferral_keeps_same_execution_until_resumed():
    admission = _admission()
    leader = admission.start_execution()
    first = await leader.open_attempt(ProviderOperationKind.GENERATION)
    await first.fail(httpx.ReadError("shared failure"))
    await first.aclose()
    discovery = admission.start_execution()
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return "metadata"

    with pytest.raises(ProviderRecoveryDeferred):
        await discovery.run_call(
            fetch,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
            wait_for_recovery=False,
        )
    assert (
        discovery.state is ProviderExecutionState.ACTIVE
        and discovery.attempts_started == 0
    )
    await leader.suspend()
    assert (
        await discovery.run_call(
            fetch, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )
        == "metadata"
    )
    assert calls == 1 and discovery.attempts_started == 1
    assert discovery.state is ProviderExecutionState.SUCCEEDED
    assert admission._active_attempts == 0 and admission._episode is None
    await leader.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_cold_opencode_candidate_retains_discovery_across_repeated_deferral(
    resume,
):
    provider, requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    admission = provider._admission
    leader = admission.start_execution()
    attempt = await leader.open_attempt(ProviderOperationKind.GENERATION)
    await attempt.fail(httpx.ReadError("shared provider unavailable"))
    await attempt.aclose()
    discovery = None
    try:
        async with provider.open_messages(
            MessagesRequest(
                model="chat-selector", messages=[{"role": "user", "content": "hello"}]
            )
        ) as candidate:
            assert isinstance(candidate, CandidateSetup)
            discovery = candidate._discovery
            assert discovery is not None
            checkpoint = RecoveryCheckpoint("messages", recovering=True)
            for _ in range(3):
                with pytest.raises(AttemptFailure) as failure:
                    _ = [
                        event
                        async for event in candidate.stream_attempt(
                            checkpoint,
                            wait_for_recovery=False,
                            can_correct=lambda: False,
                        )
                    ]
                assert failure.value.deferred
                await candidate.suspend()
                assert (
                    candidate._discovery is discovery
                    and discovery.attempts_started == 0
                )
                assert not provider._catalog._lock.locked()
                assert not requests and not catalog_requests
            if resume:
                await leader.suspend()
                events = [
                    event
                    async for event in candidate.stream_attempt(
                        checkpoint, wait_for_recovery=True, can_correct=lambda: False
                    )
                ]
                assert events[-1].completed
                assert (
                    len(requests)
                    == len(catalog_requests)
                    == discovery.attempts_started
                    == 1
                )
        assert discovery.state in {
            ProviderExecutionState.SUCCEEDED,
            ProviderExecutionState.ABANDONED,
        }
        assert admission._active_attempts == 0
    finally:
        await leader.aclose()
        await provider.cleanup()
    assert admission._episode is None or (
        admission._episode.leader is None and not admission._episode.waiters
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("refreshed_incompatible", [False, True])
async def test_deferred_target_is_resumed_before_predecessor_without_recreating_budgets(
    refreshed_incompatible,
):
    admissions = [_admission() for _ in range(3)]
    blocker = admissions[1].start_execution()
    blocked_attempt = await blocker.open_attempt(ProviderOperationKind.GENERATION)
    await blocked_attempt.fail(httpx.ReadError("B is gated"))
    await blocked_attempt.aclose()
    opened, reads, finalized, selected = [], [], [], []

    class Candidate(StreamCandidate):
        def __init__(self, index):
            super().__init__(
                admission=admissions[index],
                provider_name=str(index),
                protocol="messages",
                read_timeout_s=1,
                request_id="deferred-stack",
            )
            self.index = index

        def _build_body(self, checkpoint):
            if self.index == 2:
                raise CandidateIncompatible("C cannot represent this request")
            self.body = {"model": str(self.index)}

        async def _prepare_endpoint(self):
            if self.index == 1 and refreshed_incompatible:
                raise CandidateIncompatible("B refreshed to an incompatible endpoint")
            self.sent_body = self.body
            return ReplayOrigin(str(self.index), "messages", "", "", str(self.index))

        async def _read(self, scope):
            reads.append(self.index)
            if self.index == 0 and reads.count(0) == 1:
                raise httpx.ReadError("A failed")
            await scope.attempt.accept()
            assert self.origin is not None
            for payload in _events("Recovered"):
                kind = payload["type"]
                assert isinstance(kind, str)
                event = StreamEvent(kind, payload)
                yield DecodedStreamEvent(
                    self.origin,
                    event,
                    (event,),
                    progress=event.kind == "content_block_delta",
                    outcome=RequestOutcome.SUCCESS
                    if event.kind == "message_stop"
                    else None,
                )

        async def aclose(self):
            if self.index == 0 and not refreshed_incompatible:
                assert 1 in reads, (
                    "A must stay resumable until B invokes its physical call"
                )
            finalized.append(self.index)
            await super().aclose()

    candidates = [Candidate(index) for index in range(3)]

    @asynccontextmanager
    async def context(index):
        opened.append(index)
        if index == 1:
            assert admissions[0]._active_attempts == 0
            assert admissions[0]._episode.leader is None
            assert candidates[0].execution.state is ProviderExecutionState.ACTIVE
        if index == 2:
            await blocker.aclose()
        try:
            yield candidates[index]
        finally:
            await candidates[index].aclose()

    async def opener(index, target):
        return context(index)

    coordinator = RecoveryCoordinator(
        candidates=tuple(
            ProviderModelTarget(str(i), str(i), f"{i}/{i}") for i in range(3)
        ),
        opener=opener,
        writer=MessagesRecoveryWriter(model="public", input_tokens=0),
        progress_timeout_seconds=2,
        timeout_failure=_timeout,
        request_id="deferred-stack",
        on_selected=lambda target, index: selected.append(index),
    )
    output = "".join([chunk async for chunk in coordinator.stream()])
    assert "Recovered" in output and "event: error" not in output
    assert opened == [0, 1, 2]
    assert reads == ([0, 0] if refreshed_incompatible else [0, 1])
    assert selected == ([] if refreshed_incompatible else [1])
    assert sorted(finalized) == [0, 1, 2]
    assert candidates[0].execution.attempts_started == (
        2 if refreshed_incompatible else 1
    )
    assert candidates[1].execution.attempts_started == 1
    assert candidates[2].execution.attempts_started == 0
    assert all(admission._active_attempts == 0 for admission in admissions)
