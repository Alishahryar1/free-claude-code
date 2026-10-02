"""Native Messages HTTP execution with one admitted recovery budget."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ProviderCandidate
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    PreparedMessagesRequest,
    build_native_messages_request,
)
from free_claude_code.core.anthropic.native_stream import NativeMessagesStreamState
from free_claude_code.core.failures import UnsupportedRequestFeature
from free_claude_code.core.history_replay import ReplayOrigin, prepare_history
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesMessagesRequest,
    build_responses_messages_request,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningControl,
    ReasoningPolicy,
)
from free_claude_code.core.recovery import CandidateIncompatible, RecoveryCheckpoint
from free_claude_code.core.recovery_request import continue_request
from free_claude_code.core.request_preservation import require_preserved_body
from free_claude_code.core.stream_events import RequestOutcome, StreamEvent
from free_claude_code.core.stream_observations import (
    DecodedStreamEvent,
    MessagesObservation,
)
from free_claude_code.core.tool_adaptation import ResponsesToolIdentity
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
)
from free_claude_code.providers.history_replay import (
    normalize_messages_history,
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.stream_candidate import (
    StreamCandidate,
    content_progress,
    request_may_run_server_tools,
)

from .request_policy import (
    DEFAULT_MESSAGES_OUTPUT_TOKENS,
    MessagesModelCapabilities,
    resolve_messages_options,
)
from .wire import check_messages_failure, messages_events, messages_status_error


class AnthropicMessagesTransport:
    """Borrow HTTP, endpoint and admission owners; retain each response until closed."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        admission: ProviderAdmissionController,
        provider_name: str,
        replay_scope: str,
        read_timeout_s: float,
        capabilities: MessagesModelCapabilities = MessagesModelCapabilities(),
    ) -> None:
        self._client = client
        self._admission = admission
        self._provider_name = provider_name
        self._replay_scope = replay_scope
        self._read_timeout_s = read_timeout_s
        self._capabilities = capabilities

    def _effective_capabilities(
        self, model_info: ProviderModelInfo | None
    ) -> MessagesModelCapabilities:
        if model_info is None or model_info.max_output_tokens is None:
            return self._capabilities
        cap = model_info.max_output_tokens
        if self._capabilities.max_output_tokens is not None:
            cap = min(cap, self._capabilities.max_output_tokens)
        return replace(self._capabilities, max_output_tokens=cap)

    def _messages_body(
        self,
        request: MessagesRequest,
        reasoning: ReasoningPolicy,
        capabilities: MessagesModelCapabilities,
        preserve_native_controls: bool,
        *,
        preserve_features: bool = False,
    ) -> PreparedMessagesRequest:
        request = normalize_messages_history(request)
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_tokens,
                reasoning=reasoning,
                capabilities=capabilities,
                preserve_native_controls=preserve_native_controls,
                preserve_features=preserve_features,
                thinking=request.thinking,
                output_effort=request.output_config.get("effort")
                if request.output_config
                else None,
            )
            return build_native_messages_request(
                request, options=options, preserve_features=preserve_features
            )
        except UnsupportedRequestFeature as error:
            raise CandidateIncompatible(str(error)) from error
        except (NativeMessagesError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def _responses_body(
        self,
        request: OpenAIResponsesRequest,
        reasoning: ReasoningPolicy,
        capabilities: MessagesModelCapabilities,
        *,
        preserve_features: bool = False,
    ) -> ResponsesMessagesRequest:
        validate_history(request.model_dump(mode="json"))
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_output_tokens,
                reasoning=reasoning,
                capabilities=capabilities,
                output_effort=request.reasoning.get("effort")
                if request.reasoning
                else None,
                preserve_features=preserve_features,
            )
            return build_responses_messages_request(
                request, options=options, preserve_features=preserve_features
            )
        except UnsupportedRequestFeature as error:
            raise CandidateIncompatible(str(error)) from error
        except (NativeMessagesError, ResponsesConversionError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def open_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        preserve_native_controls: bool = False,
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return self._open_candidate(
            request,
            endpoint_context=endpoint_context,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            model_info=model_info,
            preserve_native_controls=preserve_native_controls,
        )

    def open_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return self._open_candidate(
            request,
            endpoint_context=endpoint_context,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            model_info=model_info,
        )

    @asynccontextmanager
    async def _open_candidate(
        self,
        request: MessagesRequest | OpenAIResponsesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None,
        response_model: str | None,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None,
        preserve_native_controls: bool = False,
    ) -> AsyncIterator[ProviderCandidate]:
        candidate = _MessagesCandidate(
            self,
            request,
            endpoint_context=endpoint_context,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            model_info=model_info,
            preserve_native_controls=preserve_native_controls,
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()


class _MessagesCandidate(StreamCandidate):
    def __init__(
        self,
        transport: AnthropicMessagesTransport,
        request: MessagesRequest | OpenAIResponsesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None,
        preserve_native_controls: bool,
    ) -> None:
        super().__init__(
            admission=transport._admission,
            provider_name=transport._provider_name,
            protocol="messages",
            read_timeout_s=transport._read_timeout_s,
            request_id=request_id,
            endpoint=RequestEndpoint(endpoint_context),
        )
        self._transport = transport
        self._request = request.model_copy(deep=True)
        self.original_body = request.model_dump(mode="json", exclude_unset=True)
        self.input_protocol = (
            "messages" if isinstance(request, MessagesRequest) else "responses"
        )
        self._response_model = response_model
        self._reasoning = reasoning
        self._model_info = model_info
        self._preserve_native_controls = preserve_native_controls
        self._betas: tuple[str, ...] = ()
        self._tool_identities: dict[str, ResponsesToolIdentity] = {}

    def _build_body(self, checkpoint: RecoveryCheckpoint) -> None:
        request = continue_request(self._request, checkpoint)
        capabilities = self._transport._effective_capabilities(self._model_info)
        correction = None
        if isinstance(request, MessagesRequest):
            reasoning = (
                ReasoningPolicy.provider_default()
                if self._preserve_native_controls
                else self._reasoning
            )
            prepared_request, wire_reasoning = prepare_messages_reasoning(
                request,
                reasoning,
                model_info=self._model_info,
                can_disable=True,
                normal_max_tokens=DEFAULT_MESSAGES_OUTPUT_TOKENS,
                preserve_features=self.preserve_features,
            )
            prepared = self._transport._messages_body(
                prepared_request,
                wire_reasoning,
                capabilities,
                self._preserve_native_controls,
                preserve_features=self.preserve_features,
            )
            self.body = prepared.body
            if self.preserve_features:
                require_preserved_body(
                    {
                        key: value
                        for key, value in request.model_dump(
                            mode="json", exclude_none=True, exclude_unset=True
                        ).items()
                        if key in {"thinking", "output_config"}
                    },
                    self.body,
                    "Native Messages controls",
                )
            self._betas = prepared.betas
            if (
                reasoning.control is ReasoningControl.PREFER_OFF
                and wire_reasoning.control is ReasoningControl.OFF
            ):
                correction = ReasoningCorrection(
                    (("thinking",),),
                    "max_tokens",
                    DEFAULT_MESSAGES_OUTPUT_TOKENS,
                    capabilities.max_output_tokens,
                )
        else:
            if checkpoint.recovering and request.previous_response_id:
                raise CandidateIncompatible(
                    "Messages recovery requires explicit history without a stored response handle."
                )
            prepared_responses = self._transport._responses_body(
                request,
                self._reasoning,
                capabilities,
                preserve_features=self.preserve_features,
            )
            self.body = prepared_responses.body
            self._tool_identities = dict(prepared_responses.tool_identities)
        self.corrections.reasoning = correction
        self.replay_safe = not request_may_run_server_tools(self.body, "messages")

    async def _prepare_endpoint(self) -> ReplayOrigin:
        assert self.endpoint is not None
        endpoint = await self.endpoint.resolve()
        origin = replay_origin(
            self._transport._replay_scope,
            "messages",
            str(self.body["model"]),
            endpoint=endpoint,
        )
        self.sent_body = prepare_history(
            self.body, origin, preserve_features=self.preserve_features
        )
        return origin

    async def _read(
        self, scope: ProviderAttemptScope
    ) -> AsyncIterator[DecodedStreamEvent]:
        assert self.endpoint is not None and self.endpoint.snapshot is not None
        assert self.origin is not None
        endpoint = self.endpoint.snapshot
        source = NativeMessagesStreamState()
        headers = httpx.Headers(
            {"anthropic-version": "2023-06-01", "Accept": "text/event-stream"}
        )
        headers.update(endpoint.headers)
        if endpoint.api_key and not any(
            key.lower() in {"authorization", "x-api-key"} for key in headers
        ):
            headers["x-api-key"] = endpoint.api_key
        if self._betas:
            existing = headers.pop("anthropic-beta", "")
            headers["anthropic-beta"] = ",".join(
                dict.fromkeys([*filter(None, existing.split(",")), *self._betas])
            )
        request = self._transport._client.build_request(
            "POST",
            f"{endpoint.base_url.rstrip('/')}/messages",
            json=self.sent_body,
            headers=headers,
        )
        self._dispatch(scope)
        response = scope.retain(
            await self._transport._client.send(request, stream=True)
        )
        if not response.is_success:
            raise await messages_status_error(response)
        if "text/event-stream" not in response.headers.get("content-type", "").lower():
            raise RetryableProviderProtocolError(
                "Messages upstream did not return an SSE stream."
            )
        allow_empty_completion = False
        native_paused = False
        async for kind, payload in messages_events(response):
            self.record_usage(payload)
            check_messages_failure(kind, payload)
            if kind == "message_delta":
                allow_empty_completion = payload.get("delta", {}).get(
                    "stop_reason"
                ) in {"max_tokens", "pause_turn", "refusal", "stop_sequence"}
                native_paused = (
                    payload.get("delta", {}).get("stop_reason") == "pause_turn"
                )
            completed = source.accept(kind, payload)
            if kind == "message_start":
                message = payload["message"]
                assert isinstance(message, dict)
                model = message.get("model")
                if isinstance(model, str) and model:
                    self.origin = replace(self.origin, model=model)
            if (
                source.completed
                and source.invalid_input
                and source.stop_reason != "max_tokens"
            ):
                raise RetryableProviderProtocolError(
                    "Provider completed a response with unfinished or invalid tool input."
                )
            if kind != "ping" and not scope.attempt.accepted:
                await scope.attempt.accept()
            yield DecodedStreamEvent(
                self.origin,
                StreamEvent(kind, payload),
                observation=MessagesObservation(completed, self._tool_identities),
                progress=content_progress("messages", kind, payload),
                outcome=(
                    RequestOutcome.INCOMPLETE
                    if source.stop_reason in {"max_tokens", "pause_turn"}
                    else RequestOutcome.SUCCESS
                )
                if source.completed
                else None,
                stop_reason=source.stop_reason,
                allow_empty_completion=allow_empty_completion,
                replay_safe=not native_paused,
                native_reasoning_pending=source.native_reasoning_pending,
            )
            if source.completed:
                return
        raise RetryableProviderProtocolError(
            "Messages stream ended without message_stop."
        )

    def _effective_error(self, error: Exception) -> Exception:
        return (
            RetryableProviderProtocolError(str(error))
            if isinstance(error, NativeMessagesError)
            else error
        )
