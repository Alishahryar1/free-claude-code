"""Shared OpenAI Responses execution over the official SDK."""

import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace
from typing import cast

import httpx2
from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseInputParam, ResponseStreamEvent
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ProviderCandidate
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.failures import (
    ExecutionFailure,
    FailureKind,
    UnsupportedRequestFeature,
)
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    prepare_history,
    preserve_responses_reasoning,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesProviderStream,
    ResponsesSourceState,
    ResponsesStreamFailure,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
    build_native_responses_request,
    build_responses_provider_request,
    responses_stream_failure_from_event,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.core.recovery import CandidateIncompatible, RecoveryCheckpoint
from free_claude_code.core.recovery_request import continue_request
from free_claude_code.core.stream_events import (
    DecodedStreamEvent,
    RequestOutcome,
    StreamEvent,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
    provider_authentication_status,
    reports_context_window_incomplete,
)
from free_claude_code.providers.history_replay import (
    normalize_messages_history,
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.openai_stream import OpenAIStreamAdapter
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
)
from free_claude_code.providers.stream_candidate import (
    StreamCandidate,
    content_progress,
    request_may_run_server_tools,
)

from .presentation import (
    MessagesResponsesPresenter,
    NativeResponsesPresenter,
    ResponsesPresenterFactory,
)

type ResponsesEventAdapter = Callable[[str, JsonObject], JsonObject]


class _TruncatedResponsesStream(RetryableProviderProtocolError):
    """A Responses stream ended without a terminal lifecycle event."""


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
        request_correction: Callable[
            [Exception, JsonObject, JsonObject], JsonObject | None
        ]
        | None = None,
        omitted_request_fields: frozenset[str] = frozenset(),
        tool_policy: ResponsesToolPolicy = ResponsesToolPolicy(),
    ) -> None:
        self._client = client
        self._endpoint_transport = endpoint_transport
        self._event_adapter_factory = event_adapter_factory
        self._request_correction = request_correction
        self._omitted_request_fields = omitted_request_fields
        self._tool_policy = tool_policy
        self._admission = admission
        self._provider_name = provider_name
        self._read_timeout_s = read_timeout_s
        self._log_raw_sse_events = log_raw_sse_events

    def open_messages(
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
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return self._open_candidate(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
            model_info=model_info,
            can_disable_reasoning=can_disable_reasoning,
        )

    def open_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return self._open_candidate(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
        )

    @asynccontextmanager
    async def _open_candidate(
        self,
        request: MessagesRequest | OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None,
        extra_headers: Mapping[str, str] | None,
        model_info: ProviderModelInfo | None = None,
        can_disable_reasoning: bool = True,
    ) -> AsyncIterator[ProviderCandidate]:
        candidate = _ResponsesCandidate(
            self,
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
            model_info=model_info,
            can_disable_reasoning=can_disable_reasoning,
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()

    def _build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None = None,
        can_disable_reasoning: bool = True,
        preserve_features: bool = False,
    ) -> JsonObject:
        request = normalize_messages_history(request)
        request, reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=can_disable_reasoning,
            normal_max_tokens=None,
            preserve_features=preserve_features,
        )
        try:
            return self._prepare_body(
                cast(
                    JsonObject,
                    cast(
                        ResponseCreateParamsStreaming,
                        build_responses_provider_request(
                            request,
                            reasoning=reasoning,
                            preserve_features=preserve_features,
                        ),
                    ),
                ),
                preserve_features=preserve_features,
            )
        except UnsupportedRequestFeature as error:
            raise CandidateIncompatible(str(error)) from error
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def _build_native_body(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
        preserve_features: bool = False,
    ) -> tuple[JsonObject, ResponsesToolAdapter]:
        validate_history(request.model_dump(mode="json"))
        if not request.model.strip():
            raise InvalidRequestError("Responses request model must not be empty.")
        if request.input is None or request.input == "" or request.input == []:
            raise InvalidRequestError("Responses request input must not be empty.")
        try:
            tools = ResponsesToolAdapter(
                request, self._tool_policy, preserve_features=preserve_features
            )
        except UnsupportedRequestFeature as error:
            raise CandidateIncompatible(str(error)) from error
        except ResponsesConversionError as error:
            raise InvalidRequestError(str(error)) from error
        body = self._prepare_body(
            build_native_responses_request(
                tools.request,
                model=request.model,
                reasoning=reasoning,
                preserve_features=preserve_features,
            ),
            preserve_features=preserve_features,
        )
        return body, tools

    def _prepare_body(
        self, body: JsonObject, *, preserve_features: bool = False
    ) -> JsonObject:
        for field in self._omitted_request_fields:
            if preserve_features and field in body:
                raise CandidateIncompatible(
                    f"Responses provider cannot preserve {field!r}."
                )
            body.pop(field, None)
        return body

    async def _create_sdk_stream(
        self,
        body: JsonObject,
        *,
        client: AsyncOpenAI,
        request_client: OpenAIRequestClient,
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
                **request_client.openai_headers(),
            }
            or None,
        )


class _ResponsesCandidate(StreamCandidate):
    def __init__(
        self,
        transport: OpenAIResponsesTransport,
        request: MessagesRequest | OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None,
        extra_headers: Mapping[str, str] | None,
        model_info: ProviderModelInfo | None,
        can_disable_reasoning: bool,
    ) -> None:
        super().__init__(
            admission=transport._admission,
            provider_name=transport._provider_name,
            protocol="responses",
            read_timeout_s=transport._read_timeout_s,
            request_id=request_id,
            endpoint=RequestEndpoint(endpoint_context)
            if endpoint_context is not None
            else None,
        )
        self._transport = transport
        self._request = request.model_copy(deep=True)
        self.original_body = request.model_dump(mode="json", exclude_unset=True)
        self.input_protocol = (
            "messages" if isinstance(request, MessagesRequest) else "responses"
        )
        self._input_tokens = input_tokens
        self._response_model = response_model
        self._reasoning = reasoning
        self._model_info = model_info
        self._can_disable_reasoning = can_disable_reasoning
        self._extra_headers = dict(extra_headers or {})
        self._request_client = OpenAIRequestClient(transport._endpoint_transport)
        self._client = transport._client
        self._presenter_factory: ResponsesPresenterFactory

    def _build_body(self, checkpoint: RecoveryCheckpoint) -> None:
        request = continue_request(self._request, checkpoint)
        correction = None
        if isinstance(request, MessagesRequest):
            prepared, wire_reasoning = prepare_messages_reasoning(
                request,
                self._reasoning,
                model_info=self._model_info,
                can_disable=self._can_disable_reasoning,
                normal_max_tokens=None,
                preserve_features=self.preserve_features,
            )
            self.body = self._transport._build_messages_body(
                prepared,
                reasoning=wire_reasoning,
                preserve_features=self.preserve_features,
            )
            if (
                self._reasoning.control is ReasoningControl.PREFER_OFF
                and wire_reasoning.control is ReasoningControl.OFF
            ):
                correction = ReasoningCorrection(
                    (("reasoning",),), "max_output_tokens", None
                )
            tool_names = OpenAIToolNameCodec.from_request(request)
            self._presenter_factory = lambda: MessagesResponsesPresenter(
                ResponsesProviderStream(
                    message_id=f"msg_{uuid.uuid4()}",
                    model=self._response_model,
                    input_tokens=self._input_tokens,
                    tool_names=tool_names,
                    log_raw_events=self._transport._log_raw_sse_events,
                )
            )
        else:
            if checkpoint.recovering and request.previous_response_id:
                raise CandidateIncompatible(
                    "Responses recovery requires materialized history without a stored response handle."
                )
            self.body, tools = self._transport._build_native_body(
                request,
                reasoning=self._reasoning,
                preserve_features=self.preserve_features,
            )
            self._presenter_factory = lambda: NativeResponsesPresenter(
                public_model=self._response_model,
                tool_events=tools.event_adapter(),
            )
        self.corrections = RequestCorrections("responses", correction)
        self.replay_safe = not request_may_run_server_tools(self.body, "responses")

    async def _prepare_endpoint(self) -> ReplayOrigin:
        self._client = (
            self._request_client.for_endpoint(
                self._transport._client, await self.endpoint.resolve()
            )
            if self.endpoint is not None
            else self._transport._client
        )
        origin = replay_origin(
            self.provider_name,
            "responses",
            str(self.body["model"]),
            client=self._client,
            endpoint=self.endpoint.snapshot if self.endpoint is not None else None,
        )
        self.sent_body = prepare_history(
            self.body, origin, preserve_features=self.preserve_features
        )
        return origin

    async def _read(
        self, scope: ProviderAttemptScope
    ) -> AsyncIterator[DecodedStreamEvent]:
        assert self.origin is not None
        presenter = self._presenter_factory()
        source_state = ResponsesSourceState()
        start_events = tuple(presenter.start())
        adapt = (
            self._transport._event_adapter_factory()
            if self._transport._event_adapter_factory is not None
            else None
        )
        sdk_stream = await self._transport._create_sdk_stream(
            self.sent_body,
            client=self._client,
            request_client=self._request_client,
            extra_headers=self._extra_headers,
        )
        stream = scope.retain(OpenAIStreamAdapter(sdk_stream))
        async for upstream in stream:
            if not scope.attempt.accepted:
                await scope.attempt.accept()
            raw = cast(
                JsonObject,
                upstream.model_dump(mode="json", exclude_unset=True, warnings=False),
            )
            self.record_usage(raw)
            payload = adapt(upstream.type, raw) if adapt is not None else raw
            response = payload.get("response")
            if (
                isinstance(response, dict)
                and isinstance(response.get("model"), str)
                and response["model"]
            ):
                self.origin = replace(self.origin, model=response["model"])
            for observed in source_state.feed(StreamEvent(upstream.type, payload)):
                kind = observed.kind
                if kind in {"response.failed", "error", "response.error"}:
                    raise responses_stream_failure_from_event(kind, observed.payload)
                if reports_context_window_incomplete(kind, observed.payload):
                    raise context_window_exceeded_provider_failure()
                if kind == "response.completed" and source_state.invalid_input:
                    raise RetryableProviderProtocolError(
                        "Provider completed a response with unfinished or invalid tool input."
                    )
                preserved = preserve_responses_reasoning(observed.payload, self.origin)
                output = (*start_events, *presenter.feed(kind, preserved))
                start_events = ()
                details = (
                    response.get("incomplete_details")
                    if isinstance(response, dict)
                    else None
                )
                reason = details.get("reason") if isinstance(details, dict) else None
                yield DecodedStreamEvent(
                    self.origin,
                    observed,
                    tuple(
                        replace(event, item_completion=observed.item_completion)
                        for event in output
                    ),
                    progress=content_progress("responses", kind, observed.payload),
                    outcome=(
                        RequestOutcome.INCOMPLETE
                        if kind == "response.incomplete"
                        else RequestOutcome.SUCCESS
                    )
                    if presenter.completed
                    else None,
                    stop_reason=reason if isinstance(reason, str) else None,
                    native_reasoning_pending=source_state.native_reasoning_pending,
                )
                if presenter.completed:
                    return
        raise _TruncatedResponsesStream(
            "Provider Responses stream ended without a terminal event."
        )

    def _correction(self, error: Exception) -> JsonObject | None:
        provider_correction = self._transport._request_correction
        return self.corrections.next_body(
            error,
            self.body,
            sent_body=self.sent_body,
            reasoning_error=error,
            after_common=(
                lambda _used: provider_correction(error, self.body, self.sent_body)
            )
            if provider_correction is not None
            else None,
        )

    def _effective_error(self, error: Exception) -> Exception:
        return _effective_error(error)

    async def aclose(self) -> None:
        try:
            await self._request_client.aclose()
        finally:
            await super().aclose()


def _effective_error(error: Exception) -> Exception:
    if isinstance(error, ResponsesConversionError):
        return ExecutionFailure(
            FailureKind.UPSTREAM,
            502,
            "Provider output cannot be represented in the requested protocol.",
            False,
        )
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
