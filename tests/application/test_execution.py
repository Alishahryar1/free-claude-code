"""Application-owned provider execution contracts."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
    RoutedResponsesRequest,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningCapability, ReasoningPolicy


class FakeProvider:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, object]] = []
        self.stream_close_calls = 0

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            yield "event: message_stop\ndata: {}\n\n"
        finally:
            self.stream_close_calls += 1

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        raise AssertionError("Messages test provider received a Responses request")
        yield ""

    def _record_stream_call(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str | None,
        reasoning: ReasoningPolicy,
    ) -> None:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )


class ResponsesFakeProvider:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, object]] = []
        self.stream_close_calls = 0

    async def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        raise AssertionError("Responses test provider received a Messages request")
        yield ""

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        try:
            yield "event: response.completed\ndata: {}\n\n"
        finally:
            self.stream_close_calls += 1


class WaitStep:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()


class EmptyForever:
    pass


EMPTY_FOREVER = EmptyForever()


class DelayStep:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds


type StreamStep = str | WaitStep | DelayStep | EmptyForever | BaseException


class ControlledProvider(FakeProvider):
    def __init__(self, steps: list[StreamStep]) -> None:
        super().__init__()
        self._steps = steps

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            for step in self._steps:
                if isinstance(step, str):
                    yield step
                elif isinstance(step, WaitStep):
                    step.started.set()
                    await step.release.wait()
                elif isinstance(step, DelayStep):
                    await asyncio.sleep(step.seconds)
                elif isinstance(step, EmptyForever):
                    while True:
                        yield ""
                else:
                    raise step
        finally:
            self.stream_close_calls += 1


class FailingValidationProvider(FakeProvider):
    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        raise ValueError("invalid provider request")


class ApplicationErrorValidationProvider(FakeProvider):
    def __init__(self, error: InvalidRequestError) -> None:
        super().__init__()
        self._error = error

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        raise self._error


class FailingStreamConstructionProvider(FakeProvider):
    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream construction failed")


class ExecutionFailureStreamConstructionProvider(FakeProvider):
    def __init__(self, failure: ExecutionFailure) -> None:
        super().__init__()
        self._failure = failure

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        del request, input_tokens, request_id, response_model, reasoning
        raise self._failure


class CloseControlledProvider(FakeProvider):
    def __init__(
        self,
        failure: ExecutionFailure,
        *,
        close_delay_seconds: float = 0.0,
        close_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._failure = failure
        self._close_delay_seconds = close_delay_seconds
        self._close_error = close_error
        self._read = False

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        return self

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if self._read:
            raise StopAsyncIteration
        self._read = True
        raise self._failure

    async def aclose(self) -> None:
        self.stream_close_calls += 1
        if self._close_delay_seconds:
            await asyncio.sleep(self._close_delay_seconds)
        if self._close_error is not None:
            raise self._close_error


def _target(provider_id: str, provider_model: str) -> ProviderModelTarget:
    return ProviderModelTarget(
        provider_id=provider_id,
        provider_model=provider_model,
        provider_model_ref=f"{provider_id}/{provider_model}",
    )


def _routed_request(
    *fallbacks: ProviderModelTarget,
) -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModelRoute(
            original_model="gateway-model",
            primary=_target("provider", "provider-model"),
            fallbacks=fallbacks,
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


@pytest.mark.asyncio
async def test_fallback_receives_its_own_cached_capability_in_stream():
    seen = []

    class MetadataProvider(ControlledProvider):
        async def stream_messages(self, request, input_tokens=0, **kwargs):
            seen.append(("stream", request.model, kwargs["model_info"]))
            async for event in super().stream_messages(request, input_tokens, **kwargs):
                yield event

    primary_info = ProviderModelInfo(
        "provider/provider-model", reasoning_capability=ReasoningCapability.NONE
    )
    fallback_info = ProviderModelInfo(
        "fallback/fallback-model", reasoning_capability=ReasoningCapability.REQUIRED
    )
    primary = MetadataProvider(
        [ExecutionFailure(FailureKind.UNAVAILABLE, 503, "unavailable", True)]
    )
    fallback = MetadataProvider(["verdict"])
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        lambda name: providers[name],
        progress_timeout_seconds=10,
        model_infos=(primary_info, fallback_info),
    )
    output = [
        event
        async for event in executor.stream_messages(
            _routed_request(_target("fallback", "fallback-model")),
            raw_log_payload={},
            request_id="capability-fallback",
        )
    ]
    assert output == ["verdict"]
    assert seen == [
        ("stream", "provider-model", primary_info),
        ("stream", "fallback-model", fallback_info),
    ]


def _routed_responses_request(
    *fallbacks: ProviderModelTarget,
) -> RoutedResponsesRequest:
    request = OpenAIResponsesRequest(
        model="provider-model",
        input=[{"role": "user", "content": "hello"}],
        reasoning={"effort": "high"},
    )
    return RoutedResponsesRequest(
        request=request,
        resolved=ResolvedModelRoute(
            original_model="gateway-model",
            primary=_target("provider", "provider-model"),
            fallbacks=fallbacks,
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


def _executor_stream(
    provider: FakeProvider,
    *,
    timeout_seconds: float,
    request_id: str,
) -> AsyncIterator[str]:
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
        progress_timeout_seconds=timeout_seconds,
    )
    return executor.stream_messages(
        _routed_request(),
        raw_log_payload={},
        request_id=request_id,
    )


@pytest.mark.asyncio
async def test_executor_routes_native_responses_without_messages_conversion() -> None:
    provider = ResponsesFakeProvider()
    routed = _routed_responses_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        responses_token_counter=lambda _request: 23,
    )

    stream = executor.stream_responses(
        routed,
        raw_log_payload=request.model_dump(mode="json"),
        request_id="req_responses_application",
    )

    assert provider.stream_calls == []
    assert [chunk async for chunk in stream] == [
        "event: response.completed\ndata: {}\n\n"
    ]
    assert provider.stream_calls == [
        {
            "request": request,
            "input_tokens": 23,
            "request_id": "req_responses_application",
            "response_model": "gateway-model",
            "reasoning": ReasoningPolicy.on(),
        }
    ]
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_executor_uses_structural_provider_port_and_defers_stream_startup() -> (
    None
):
    provider = FakeProvider()
    routed = _routed_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream_messages(
        routed,
        raw_log_payload=request.model_dump(),
        request_id="req_application",
    )

    assert provider.stream_calls == []
    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert provider.stream_calls == [
        {
            "request": request,
            "input_tokens": 17,
            "request_id": "req_application",
            "response_model": "gateway-model",
            "reasoning": ReasoningPolicy.on(),
        }
    ]
    assert provider.stream_close_calls == 1


def _execution_failure(
    message: str,
    *,
    retryable: bool = True,
) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message=message,
        retryable=retryable,
    )


@pytest.mark.asyncio
async def test_primary_success_never_resolves_fallback() -> None:
    primary = FakeProvider()
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "fallback-model")),
        raw_log_payload={},
        request_id="req_primary_success",
    )

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert resolved_ids == ["provider"]
    assert fallback.stream_calls == []


@pytest.mark.asyncio
async def test_retryable_preframe_failure_selects_fallback_after_closing_primary() -> (
    None
):
    order: list[str] = []
    primary = ControlledProvider([_execution_failure("primary overloaded")])
    fallback = ControlledProvider(["fallback-frame"])

    def resolve(provider_id: str) -> FakeProvider:
        if provider_id == "fallback":
            assert primary.stream_close_calls == 1
            order.append("fallback-resolved")
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(
        resolve,
        token_counter=lambda _messages, _system, _tools: 17,
        progress_timeout_seconds=60.0,
        generation_id=7,
    )
    routed = _routed_request(_target("fallback", "fallback-model"))

    with patch("free_claude_code.application.execution.trace_event") as trace_mock:
        chunks = [
            chunk
            async for chunk in executor.stream_messages(
                routed,
                raw_log_payload={},
                request_id="req_fallback",
            )
        ]

    assert chunks == ["fallback-frame"]
    assert order == ["fallback-resolved"]
    assert primary.stream_close_calls == 1
    assert fallback.stream_close_calls == 1
    fallback_request = fallback.stream_calls[0]["request"]
    assert isinstance(fallback_request, MessagesRequest)
    assert fallback_request.model == "fallback-model"
    assert fallback.stream_calls[0] == {
        "request": fallback_request,
        "input_tokens": 17,
        "request_id": "req_fallback",
        "response_model": "gateway-model",
        "reasoning": ReasoningPolicy.on(),
    }
    events = [call.kwargs for call in trace_mock.call_args_list]
    started = next(
        event
        for event in events
        if event.get("event") == "free_claude_code.model_fallback.started"
    )
    assert started == {
        "stage": "execution",
        "event": "free_claude_code.model_fallback.started",
        "source": "application",
        "request_id": "req_fallback",
        "wire_api": "messages",
        "from_provider_model_ref": "provider/provider-model",
        "to_provider_model_ref": "fallback/fallback-model",
        "candidate_index": 2,
        "candidate_count": 2,
        "failure_kind": "overloaded",
        "status_code": 529,
        "provider_retryable": True,
        "generation_id": 7,
    }
    selected = next(
        event
        for event in events
        if event.get("event") == "free_claude_code.model_fallback.selected"
    )
    assert selected["selected_provider_model_ref"] == "fallback/fallback-model"
    assert selected["candidate_index"] == 2


@pytest.mark.asyncio
async def test_fallback_chain_preserves_exact_last_failure() -> None:
    first = _execution_failure("first")
    second = _execution_failure("second")
    providers = {
        "provider": ControlledProvider([first]),
        "second": ControlledProvider([second]),
    }
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )
    stream = executor.stream_messages(
        _routed_request(_target("second", "model")),
        raw_log_payload={},
        request_id="req_exhausted",
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value is second
    assert all(provider.stream_close_calls == 1 for provider in providers.values())


@pytest.mark.asyncio
async def test_multiple_retryable_failures_walk_fallbacks_in_order() -> None:
    providers = {
        "provider": ControlledProvider([_execution_failure("primary")]),
        "second": ControlledProvider([_execution_failure("second")]),
        "third": ControlledProvider(["third-frame"]),
    }
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return providers[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream_messages(
        _routed_request(
            _target("second", "second-model"),
            _target("third", "third-model"),
        ),
        raw_log_payload={},
        request_id="req_ordered_fallbacks",
    )

    assert [chunk async for chunk in stream] == ["third-frame"]
    assert resolved_ids == ["provider", "second", "third"]
    assert all(provider.stream_close_calls == 1 for provider in providers.values())


@pytest.mark.asyncio
async def test_empty_primary_completion_does_not_select_fallback() -> None:
    primary = ControlledProvider([])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_empty_primary",
    )

    assert [chunk async for chunk in stream] == []
    assert resolved_ids == ["provider"]
    assert primary.stream_close_calls == 1


@pytest.mark.asyncio
async def test_unexpected_primary_failure_does_not_select_fallback() -> None:
    primary = ControlledProvider([RuntimeError("unexpected")])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_unexpected_primary",
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        await anext(stream)

    assert resolved_ids == ["provider"]
    assert primary.stream_close_calls == 1


@pytest.mark.asyncio
async def test_invalid_none_stream_remains_terminal() -> None:
    primary = FakeProvider()
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    with patch.object(primary, "stream_messages", return_value=None):
        stream = executor.stream_messages(
            _routed_request(_target("fallback", "model")),
            raw_log_payload={},
            request_id="req_invalid_stream",
        )
        with pytest.raises(TypeError):
            await anext(stream)

    assert resolved_ids == ["provider"]
    assert fallback.stream_calls == []


@pytest.mark.parametrize("failure_kind", tuple(FailureKind))
@pytest.mark.asyncio
async def test_nonretryable_provider_failure_selects_fallback_before_first_frame(
    failure_kind: FailureKind,
) -> None:
    primary_failure = ExecutionFailure(
        kind=failure_kind,
        status_code=500,
        message="provider rejected candidate",
        retryable=False,
    )
    primary = ControlledProvider([primary_failure])
    fallback = ControlledProvider(["fallback-frame"])
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_rejected",
    )

    with patch("free_claude_code.application.execution.trace_event") as trace_mock:
        chunks = [chunk async for chunk in stream]

    assert chunks == ["fallback-frame"]
    assert resolved_ids == ["provider", "fallback"]
    fallback_started = next(
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "free_claude_code.model_fallback.started"
    )
    assert fallback_started["failure_kind"] == failure_kind.value
    assert fallback_started["provider_retryable"] is False


@pytest.mark.asyncio
async def test_primary_canonical_startup_failure_selects_fallback() -> None:
    primary_failure = ExecutionFailure(
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="primary context exceeded",
        retryable=False,
    )
    primary = ExecutionFailureStreamConstructionProvider(primary_failure)
    fallback = ControlledProvider(["fallback-frame"])
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )

    stream = executor.stream_messages(
        _routed_request(_target("fallback", "fallback-model")),
        raw_log_payload={},
        request_id="req_startup_fallback",
    )

    assert [chunk async for chunk in stream] == ["fallback-frame"]
    assert primary.stream_calls == []
    fallback_request = fallback.stream_calls[0]["request"]
    assert isinstance(fallback_request, MessagesRequest)
    assert fallback_request.model == "fallback-model"


@pytest.mark.asyncio
async def test_nonretryable_stream_construction_failure_selects_fallback() -> None:
    primary_failure = ExecutionFailure(
        kind=FailureKind.PERMISSION,
        status_code=403,
        message="provider denied candidate",
        retryable=False,
    )
    primary = ExecutionFailureStreamConstructionProvider(primary_failure)
    fallback = ControlledProvider(["fallback-frame"])
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_construction_failure",
    )

    assert [chunk async for chunk in stream] == ["fallback-frame"]
    assert fallback.stream_calls[0]["request_id"] == "req_construction_failure"


@pytest.mark.asyncio
async def test_failure_after_first_frame_never_selects_fallback() -> None:
    primary_failure = _execution_failure("after frame")
    primary = ControlledProvider(["first-frame", primary_failure])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_committed",
    )

    assert await anext(stream) == "first-frame"
    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value is primary_failure
    assert resolved_ids == ["provider"]


@pytest.mark.asyncio
async def test_lazy_fallback_startup_application_error_stops_unchanged() -> None:
    primary = ControlledProvider([_execution_failure("primary")])
    startup_error = InvalidRequestError("fallback is misconfigured")
    fallback = ApplicationErrorValidationProvider(startup_error)
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_startup",
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        await anext(stream)

    assert exc_info.value is startup_error
    assert primary.stream_close_calls == 1
    assert fallback.stream_calls == []


@pytest.mark.asyncio
async def test_candidate_requests_are_isolated_from_provider_mutation() -> None:
    class MutatingProvider(ControlledProvider):
        async def stream_messages(
            self,
            request: MessagesRequest,
            input_tokens: int = 0,
            *,
            request_id: str | None = None,
            response_model: str | None = None,
            reasoning: ReasoningPolicy,
            request_headers: Mapping[str, str] | None = None,
            model_info: ProviderModelInfo | None = None,
        ) -> AsyncIterator[str]:
            request.messages[0].content = "mutated"
            async for chunk in super().stream_messages(
                request,
                input_tokens,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
            ):
                yield chunk

    primary = MutatingProvider([_execution_failure("primary")])
    fallback = FakeProvider()
    providers = {"provider": primary, "fallback": fallback}
    routed = _routed_request(_target("fallback", "model"))
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )

    assert [
        chunk
        async for chunk in executor.stream_messages(
            routed,
            raw_log_payload={},
            request_id="req_isolated",
        )
    ]
    fallback_request = fallback.stream_calls[0]["request"]
    assert isinstance(fallback_request, MessagesRequest)
    assert fallback_request.messages[0].content == "hello"
    assert routed.request.messages[0].content == "hello"


@pytest.mark.asyncio
async def test_closing_executor_stream_closes_provider_stream_once() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    stream = executor.stream_messages(
        routed,
        raw_log_payload={},
        request_id="req_early_close",
    )

    assert await anext(stream) == "event: message_stop\ndata: {}\n\n"
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_stream_construction_failure_remains_deferred_to_iteration() -> None:
    provider = FailingStreamConstructionProvider()
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": provider, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_deferred_construction",
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)

    assert resolved_ids == ["provider"]


@pytest.mark.asyncio
async def test_executor_validation_is_deferred_until_iteration() -> None:
    provider = FailingValidationProvider()
    token_counter = MagicMock(return_value=17)
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=token_counter,
    )

    stream = executor.stream_messages(
        _routed_request(),
        raw_log_payload={},
        request_id="req_application",
    )
    token_counter.assert_called_once()
    with pytest.raises(ValueError, match="invalid provider request"):
        await anext(stream)
    assert provider.stream_calls == []


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_executor_rejects_invalid_progress_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ProviderExecutor(
            lambda _provider_id: FakeProvider(),
            progress_timeout_seconds=timeout_seconds,
        )


@pytest.mark.asyncio
async def test_progress_timeout_before_first_chunk_is_canonical_and_correlated() -> (
    None
):
    provider = ControlledProvider([WaitStep()])
    request_id = "req_progress_timeout"

    with patch("free_claude_code.application.execution.trace_event") as trace_mock:
        stream = _executor_stream(
            provider,
            timeout_seconds=0.02,
            request_id=request_id,
        )
        with pytest.raises(ExecutionFailure) as exc_info:
            await anext(stream)

    failure = exc_info.value
    assert failure.kind == FailureKind.TIMEOUT
    assert failure.status_code == 504
    assert failure.retryable is False
    assert failure.message == (
        "Provider execution made no progress for 0.02 seconds.\n\n"
        f"Request ID: {request_id}"
    )
    assert provider.stream_close_calls == 1
    timeout_traces = [
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "free_claude_code.provider.progress_timeout"
    ]
    assert timeout_traces == [
        {
            "stage": "execution",
            "event": "free_claude_code.provider.progress_timeout",
            "source": "application",
            "request_id": request_id,
            "provider_id": "provider",
            "timeout_seconds": 0.02,
        }
    ]


@pytest.mark.asyncio
async def test_application_progress_timeout_never_resolves_fallback() -> None:
    primary = ControlledProvider([WaitStep()])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=0.02)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_terminal_progress_timeout",
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value.kind is FailureKind.TIMEOUT
    assert exc_info.value.status_code == 504
    assert exc_info.value.retryable is False
    assert resolved_ids == ["provider"]
    assert primary.stream_close_calls == 1
    assert fallback.stream_calls == []


@pytest.mark.asyncio
async def test_provider_cleanup_cannot_delay_fallback_past_progress_deadline() -> None:
    primary = CloseControlledProvider(
        _execution_failure("primary overloaded"),
        close_delay_seconds=0.05,
    )
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=0.02)
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_cleanup_timeout",
    )

    with (
        patch("free_claude_code.application.execution.trace_event") as trace_mock,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        await anext(stream)

    assert exc_info.value.kind is FailureKind.TIMEOUT
    assert exc_info.value.status_code == 504
    assert resolved_ids == ["provider"]
    assert primary.stream_close_calls == 1
    assert fallback.stream_calls == []
    assert all(
        call.kwargs.get("event") != "free_claude_code.model_fallback.started"
        for call in trace_mock.call_args_list
    )


@pytest.mark.asyncio
async def test_cleanup_failure_trace_names_preserved_provider_failure() -> None:
    failure = _execution_failure("primary overloaded")
    provider = CloseControlledProvider(
        failure,
        close_error=RuntimeError("close failed"),
    )
    stream = _executor_stream(
        provider,
        timeout_seconds=1.0,
        request_id="req_close_failure_trace",
    )

    with (
        patch("free_claude_code.core.trace.trace_event") as trace_mock,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        await anext(stream)

    assert exc_info.value is failure
    close_trace = next(
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "stream.input.close_failed"
    )
    assert close_trace["close_exc_type"] == "RuntimeError"
    assert close_trace["preserved_exc_type"] == "ExecutionFailure"


@pytest.mark.asyncio
async def test_progress_timeout_renews_after_output_without_crossing_yield() -> None:
    second_wait = WaitStep()
    third_wait = WaitStep()
    provider = ControlledProvider(["first", second_wait, "second", third_wait, "third"])
    stream = _executor_stream(
        provider,
        timeout_seconds=0.02,
        request_id="req_progress_renewal",
    )

    assert await anext(stream) == "first"
    await asyncio.sleep(0.05)
    second_wait.release.set()
    assert await anext(stream) == "second"

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert third_wait.started.is_set()
    assert exc_info.value.kind == FailureKind.TIMEOUT
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_empty_chunks_do_not_renew_provider_progress() -> None:
    provider = ControlledProvider([EMPTY_FOREVER])
    stream = _executor_stream(
        provider,
        timeout_seconds=0.02,
        request_id="req_empty_progress",
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value.kind == FailureKind.TIMEOUT
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_fallback_transition_does_not_reset_shared_progress_deadline() -> None:
    primary_wait = WaitStep()
    primary = ControlledProvider(
        [primary_wait, _execution_failure("primary overloaded")]
    )
    fallback_wait = WaitStep()
    fallback = ControlledProvider([fallback_wait])
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60,
    )
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_shared_deadline",
    )
    deadlines: list[float | None] = []
    timeouts: list[asyncio.Timeout] = []

    def controlled_timeout_at(deadline):
        deadlines.append(deadline)
        timeout = asyncio.timeout(None)
        timeouts.append(timeout)
        return timeout

    with (
        patch("free_claude_code.application.execution.trace_event") as trace_mock,
        patch(
            "free_claude_code.application.execution.asyncio.timeout_at",
            controlled_timeout_at,
        ),
    ):
        reading = asyncio.ensure_future(anext(stream))
        await primary_wait.started.wait()
        primary_wait.release.set()
        await fallback_wait.started.wait()
        # Expire the actual fallback read context after observing its admission;
        # machine load must not determine whether the shared deadline was reused.
        timeouts[-1].reschedule(asyncio.get_running_loop().time())
        with pytest.raises(ExecutionFailure) as exc_info:
            await reading

    read_deadlines = [deadline for deadline in deadlines if deadline is not None]
    assert len(read_deadlines) >= 3
    assert len(set(read_deadlines)) == 1
    assert exc_info.value.kind == FailureKind.TIMEOUT
    assert primary.stream_close_calls == fallback.stream_close_calls == 1
    timeout_trace = next(
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "free_claude_code.provider.progress_timeout"
    )
    assert timeout_trace["provider_id"] == "fallback"


@pytest.mark.asyncio
async def test_provider_owned_timeout_is_not_reclassified() -> None:
    provider = ControlledProvider([TimeoutError("provider-owned timeout")])
    stream = _executor_stream(
        provider,
        timeout_seconds=1.0,
        request_id="req_provider_timeout",
    )

    with pytest.raises(TimeoutError, match="provider-owned timeout"):
        await anext(stream)

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_cancelling_progress_wait_remains_cancellation() -> None:
    wait = WaitStep()
    provider = ControlledProvider([wait])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": provider, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=1.0,
    )
    stream = executor.stream_messages(
        _routed_request(_target("fallback", "model")),
        raw_log_payload={},
        request_id="req_cancelled_progress",
    )
    task = asyncio.ensure_future(anext(stream))
    await wait.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.stream_close_calls == 1
    assert resolved_ids == ["provider"]
