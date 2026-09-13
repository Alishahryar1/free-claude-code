"""Shared OpenAI Responses execution over the official SDK."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from typing import cast

import httpx2
from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseInputParam, ResponseStreamEvent
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import (
    prepare_history,
    preserve_responses_reasoning,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesProviderStream,
    ResponsesStreamFailure,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
    build_native_responses_request,
    build_responses_provider_request,
    responses_stream_failure_from_event,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.endpoint import EndpointContext, RequestEndpoint
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
    is_retryable_stream_error,
    provider_authentication_status,
    reports_context_window_incomplete,
)
from free_claude_code.providers.history_replay import (
    history_retry_body,
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
)

from .presentation import (
    MessagesResponsesPresenter,
    NativeResponsesPresenter,
    ResponsesExecutionOutcome,
    ResponsesPresenterFactory,
)

type ResponsesEventAdapter = Callable[[str, JsonObject], JsonObject]


class _TruncatedResponsesStream(RetryableProviderProtocolError):
    """A Responses stream ended without a terminal lifecycle event."""


class _ClosableResponsesStream(AsyncIterator[ResponseStreamEvent]):
    """Expose the OpenAI SDK stream through the shared ``aclose`` contract."""

    def __init__(self, stream: AsyncStream[ResponseStreamEvent]) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        return await anext(self._stream)

    async def aclose(self) -> None:
        await self._stream.close()


class OpenAIResponsesTransport:
    """Execute public Responses requests with provider-owned retry semantics."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        admission: ProviderAdmissionController,
        provider_name: str,
        read_timeout_s: float,
        log_raw_sse_events: bool,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
        event_adapter_factory: Callable[[], ResponsesEventAdapter] | None = None,
        omitted_request_fields: frozenset[str] = frozenset(),
        tool_policy: ResponsesToolPolicy = ResponsesToolPolicy(),
    ) -> None:
        self._client = client
        self._endpoint_transport = endpoint_transport
        self._event_adapter_factory = event_adapter_factory
        self._omitted_request_fields = omitted_request_fields
        self._tool_policy = tool_policy
        self._admission = admission
        self._provider_name = provider_name
        self._read_timeout_s = read_timeout_s
        self._log_raw_sse_events = log_raw_sse_events

    def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        can_disable_reasoning: bool = True,
    ) -> AsyncIterator[str]:
        prepared, wire_reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=can_disable_reasoning,
            normal_max_tokens=None,
        )
        body = self._build_messages_body(prepared, reasoning=wire_reasoning)
        correction = (
            ReasoningCorrection((("reasoning",),), "max_output_tokens", None)
            if reasoning.control is ReasoningControl.PREFER_OFF
            and wire_reasoning.control is ReasoningControl.OFF
            else None
        )
        tool_names = OpenAIToolNameCodec.from_request(request)
        message_id = f"msg_{uuid.uuid4()}"
        return self._run_stream(
            body,
            reasoning_correction=correction,
            endpoint_context=endpoint_context,
            extra_headers=dict(extra_headers or {}),
            request_id=request_id,
            response_model=response_model,
            presenter_factory=lambda: MessagesResponsesPresenter(
                ResponsesProviderStream(
                    message_id=message_id,
                    model=response_model,
                    input_tokens=input_tokens,
                    tool_names=tool_names,
                    log_raw_events=self._log_raw_sse_events,
                )
            ),
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens
        body, tools = self._build_native_body(request, reasoning=reasoning)
        return self._run_stream(
            body,
            endpoint_context=endpoint_context,
            extra_headers=dict(extra_headers or {}),
            request_id=request_id,
            response_model=response_model,
            presenter_factory=lambda: NativeResponsesPresenter(
                public_model=response_model, tool_events=tools.event_adapter()
            ),
        )

    def _build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None = None,
        can_disable_reasoning: bool = True,
    ) -> JsonObject:
        validate_history(request.model_dump(mode="json"))
        request, reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=can_disable_reasoning,
            normal_max_tokens=None,
        )
        try:
            return self._prepare_body(
                cast(
                    JsonObject,
                    cast(
                        ResponseCreateParamsStreaming,
                        build_responses_provider_request(request, reasoning=reasoning),
                    ),
                )
            )
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def _build_native_body(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> tuple[JsonObject, ResponsesToolAdapter]:
        validate_history(request.model_dump(mode="json"))
        if not request.model.strip():
            raise InvalidRequestError("Responses request model must not be empty.")
        if request.input is None or request.input == "" or request.input == []:
            raise InvalidRequestError("Responses request input must not be empty.")
        try:
            tools = ResponsesToolAdapter(request, self._tool_policy)
        except ResponsesConversionError as error:
            raise InvalidRequestError(str(error)) from error
        body = self._prepare_body(
            build_native_responses_request(
                tools.request,
                model=request.model,
                reasoning=reasoning,
            )
        )
        return body, tools

    def _prepare_body(self, body: JsonObject) -> JsonObject:
        for field in self._omitted_request_fields:
            body.pop(field, None)
        return body

    async def _run_stream(
        self,
        body: JsonObject,
        *,
        request_id: str | None,
        response_model: str,
        presenter_factory: ResponsesPresenterFactory,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        reasoning_correction: ReasoningCorrection | None = None,
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(request_id=request_id)
        outcome = ResponsesExecutionOutcome()
        endpoint = (
            RequestEndpoint(endpoint_context, self._endpoint_transport)
            if endpoint_context is not None
            else None
        )
        provider_stream = self._run_execution(
            body,
            reasoning_correction=reasoning_correction,
            request_id=request_id,
            response_model=response_model,
            presenter_factory=presenter_factory,
            execution=execution,
            outcome=outcome,
            endpoint=endpoint,
            extra_headers=extra_headers,
        )
        try:
            async for event in provider_stream:
                if endpoint is not None:
                    endpoint.commit()
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            if outcome.failure is None:
                execution.succeed()
            else:
                execution.fail(outcome.failure)
        finally:
            try:
                await maybe_await_aclose(provider_stream)
            finally:
                try:
                    if endpoint is not None:
                        await endpoint.aclose()
                finally:
                    execution.abandon()

    async def _run_execution(
        self,
        body: JsonObject,
        *,
        request_id: str | None,
        response_model: str,
        presenter_factory: ResponsesPresenterFactory,
        execution: ProviderExecution,
        outcome: ResponsesExecutionOutcome,
        endpoint: RequestEndpoint | None = None,
        extra_headers: Mapping[str, str] | None = None,
        reasoning_correction: ReasoningCorrection | None = None,
    ) -> AsyncIterator[str]:
        recovery = RecoveryController()
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            request_id=request_id,
            execution_id=execution.execution_id,
            gateway_model=response_model,
            downstream_model=body.get("model"),
            transport="responses",
        )

        while execution.can_attempt:
            presenter = presenter_factory()
            start_events = tuple(presenter.start())
            presenter_started = False
            adapt_event = (
                self._event_adapter_factory()
                if self._event_adapter_factory is not None
                else None
            )

            scope: ProviderAttemptScope | None = None
            stream_opened = False
            try:
                client = (
                    await endpoint.openai_client(self._client)
                    if endpoint is not None
                    else self._client
                )
                origin = replay_origin(
                    self._provider_name,
                    "responses",
                    str(body["model"]),
                    client=client,
                    endpoint=endpoint.snapshot if endpoint is not None else None,
                )
                sent_body = prepare_history(body, origin)
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=self._provider_name,
                    request_id=request_id,
                )
                sdk_stream = await self._create_sdk_stream(
                    sent_body,
                    client=client,
                    endpoint=endpoint,
                    extra_headers=extra_headers,
                )
                stream = scope.retain(_ClosableResponsesStream(sdk_stream))
                stream_opened = True

                async for upstream_event in stream:
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    if not presenter_started:
                        presenter_started = True
                        for event in start_events:
                            for held in recovery.push(event):
                                yield held
                    payload = cast(
                        JsonObject,
                        upstream_event.to_dict(mode="json"),
                    )
                    if upstream_event.type in {
                        "response.failed",
                        "error",
                        "response.error",
                    }:
                        stream_failure = responses_stream_failure_from_event(
                            upstream_event.type,
                            payload,
                        )
                        if adapt_event is not None:
                            try:
                                stream_failure.payload = adapt_event(
                                    upstream_event.type, payload
                                )
                            except RetryableProviderProtocolError:
                                # Preserve the upstream failure classification.
                                # Synthesize the terminal event if partial output
                                # cannot retain a consistent public identity.
                                stream_failure.payload = None
                        raise stream_failure
                    if reports_context_window_incomplete(
                        upstream_event.type,
                        payload,
                    ):
                        raise context_window_exceeded_provider_failure()
                    if adapt_event is not None:
                        payload = adapt_event(upstream_event.type, payload)
                    response = payload.get("response")
                    if (
                        isinstance(response, dict)
                        and isinstance(response.get("model"), str)
                        and response["model"]
                    ):
                        origin = replace(origin, model=response["model"])
                    payload = preserve_responses_reasoning(payload, origin)
                    for event in presenter.feed(upstream_event.type, payload):
                        for held in recovery.push(event):
                            yield held
                    if presenter.completed:
                        break
                if not presenter.completed:
                    raise _TruncatedResponsesStream(
                        "Provider Responses stream ended without a terminal event."
                    )
                for event in recovery.flush():
                    yield event
                trace_event(
                    stage="provider",
                    event="provider.response.completed",
                    source="provider",
                    provider=self._provider_name,
                    request_id=request_id,
                    transport="responses",
                )
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = _effective_error(raw_error)
                if (
                    scope is not None
                    and endpoint is not None
                    and await endpoint.retry_authentication(
                        error, scope.attempt, execution
                    )
                ):
                    recovery.discard()
                    continue
                if scope is not None and not recovery.committed:
                    corrected_history = history_retry_body(
                        raw_error, sent_body, "responses"
                    )
                    if corrected_history is not None:
                        retry = (
                            execution.can_attempt
                            if scope.attempt.accepted
                            else await scope.attempt.correct(error)
                            is ProviderCorrectionAction.RETRY
                        )
                        if retry:
                            body = corrected_history
                            recovery.discard()
                            continue
                if (
                    scope is not None
                    and reasoning_correction is not None
                    and not recovery.committed
                ):
                    corrected_body = reasoning_correction.retry_body(raw_error, body)
                    if corrected_body is not None:
                        retry = (
                            execution.can_attempt
                            if scope.attempt.accepted
                            else await scope.attempt.correct(error)
                            is ProviderCorrectionAction.RETRY
                        )
                        reasoning_correction = None
                        if retry:
                            body = corrected_body
                            recovery.discard()
                            continue
                attempt_failure = None
                if scope is not None and not scope.attempt.accepted:
                    attempt_failure = await scope.attempt.fail(error)
                if attempt_failure is not None and attempt_failure.retry_allowed:
                    recovery.discard()
                    _trace_early_retry(
                        provider_name=self._provider_name,
                        request_id=request_id,
                        execution=execution,
                    )
                    continue

                retryable = (
                    attempt_failure.retryable
                    if attempt_failure is not None
                    else is_retryable_stream_error(error)
                )
                decision = recovery.advance_failure(
                    retryable=retryable,
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                )
                if decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
                    _trace_early_retry(
                        provider_name=self._provider_name,
                        request_id=request_id,
                        execution=execution,
                    )
                    continue

                failure = classify_provider_failure(
                    error,
                    provider_name=self._provider_name,
                    read_timeout_s=self._read_timeout_s,
                    request_id=request_id,
                )
                trace_event(
                    stage="provider",
                    event="provider.response.error",
                    source="provider",
                    provider=self._provider_name,
                    request_id=request_id,
                    transport="responses",
                    exc_type=type(error).__name__,
                    failure_kind=failure.kind.value,
                    status_code=failure.status_code,
                    provider_retryable=failure.retryable,
                )
                if not decision.committed:
                    recovery.discard()
                    raise failure from raw_error
                for event in presenter.terminal_failure(raw_error, failure):
                    yield event
                if presenter.terminal_failure_completes_wire:
                    outcome.failure = failure
                    return
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Responses execution ended without a terminal result.")

    async def _create_sdk_stream(
        self,
        body: JsonObject,
        *,
        client: AsyncOpenAI,
        endpoint: RequestEndpoint | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> AsyncStream[ResponseStreamEvent]:
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise InvalidRequestError("Responses request model must not be empty.")
        input_value = cast(str | ResponseInputParam, body.get("input"))
        extra_body = {
            key: value
            for key, value in body.items()
            if key not in {"model", "input", "stream", "store"}
        }
        return await client.responses.create(
            model=model,
            input=input_value,
            stream=True,
            store=False,
            extra_body=extra_body or None,
            extra_headers={
                **(extra_headers or {}),
                **(endpoint.openai_headers() if endpoint is not None else {}),
            }
            or None,
        )


def _effective_error(error: Exception) -> Exception:
    if not isinstance(error, ResponsesStreamFailure):
        return error
    message = (
        extract_upstream_error_detail(error).exception_text
        or "Provider response failed."
    )
    if is_context_window_error_code(error.code):
        return context_window_exceeded_provider_failure()
    auth_status = provider_authentication_status(error)
    if auth_status is not None:
        return ExecutionFailure(
            FailureKind.AUTHENTICATION
            if auth_status == 401
            else FailureKind.PERMISSION,
            auth_status,
            message,
            False,
        )
    code = (error.code or "").lower()
    if "rate" in code or "429" in code:
        return ExecutionFailure(FailureKind.RATE_LIMIT, 429, message, True)
    if any(marker in code for marker in ("overload", "capacity", "529")):
        return ExecutionFailure(FailureKind.OVERLOADED, 529, message, True)
    retryable = any(
        marker in code for marker in ("server", "internal", "unavailable", "timeout")
    )
    return ExecutionFailure(FailureKind.UPSTREAM, 502, message, retryable)


def _trace_early_retry(
    *,
    provider_name: str,
    request_id: str | None,
    execution: ProviderExecution,
) -> None:
    trace_event(
        stage="provider",
        event="provider.recovery.early_retry",
        source="provider",
        provider=provider_name,
        request_id=request_id,
        transport="responses",
        attempts_started=execution.attempts_started,
        max_attempts=execution.max_attempts,
    )
