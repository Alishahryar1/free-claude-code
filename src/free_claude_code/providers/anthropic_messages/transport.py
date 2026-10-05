"""Native Messages HTTP execution with one admitted recovery budget."""

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from functools import partial

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    PreparedMessagesRequest,
    build_native_messages_request,
)
from free_claude_code.core.anthropic.native_stream import NativeMessagesRelay
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import ReplayOrigin, prepare_history
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    AnthropicToResponsesStream,
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
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.continuation import (
    ContinuationRequest,
    SourceRecoveryState,
    public_recovery,
    public_stream_failure,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    is_retryable_stream_error,
)
from free_claude_code.providers.history_replay import (
    normalize_messages_history,
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
    RequestRecovery,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
)

from .request_policy import (
    DEFAULT_MESSAGES_OUTPUT_TOKENS,
    MessagesModelCapabilities,
    resolve_messages_options,
)
from .wire import (
    check_messages_failure,
    is_messages_stop,
    messages_events,
    messages_status_error,
)

type _Presenter = NativeMessagesRelay | AnthropicToResponsesStream


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
    ) -> PreparedMessagesRequest:
        request = normalize_messages_history(request)
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_tokens,
                reasoning=reasoning,
                capabilities=capabilities,
                preserve_native_controls=preserve_native_controls,
                thinking=request.thinking,
                output_effort=request.output_config.get("effort")
                if request.output_config
                else None,
            )
            return build_native_messages_request(request, options=options)
        except (NativeMessagesError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def _responses_body(
        self,
        request: OpenAIResponsesRequest,
        reasoning: ReasoningPolicy,
        capabilities: MessagesModelCapabilities,
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
            )
            return build_responses_messages_request(request, options=options)
        except (NativeMessagesError, ResponsesConversionError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def stream_messages(
        self,
        request: MessagesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        preserve_native_controls: bool = False,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        capabilities = self._effective_capabilities(model_info)
        if preserve_native_controls:
            reasoning = ReasoningPolicy.provider_default()
        prepared_request, wire_reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=True,
            normal_max_tokens=DEFAULT_MESSAGES_OUTPUT_TOKENS,
        )
        prepared = self._messages_body(
            prepared_request, wire_reasoning, capabilities, preserve_native_controls
        )
        correction = (
            ReasoningCorrection(
                (("thinking",),),
                "max_tokens",
                DEFAULT_MESSAGES_OUTPUT_TOKENS,
                capabilities.max_output_tokens,
            )
            if reasoning.control is ReasoningControl.PREFER_OFF
            and wire_reasoning.control is ReasoningControl.OFF
            else None
        )
        return self._stream(
            prepared.body,
            reasoning_correction=correction,
            betas=prepared.betas,
            endpoint_context=endpoint_context,
            request_id=request_id,
            presenter_factory=lambda origin: NativeMessagesRelay(
                public_model=response_model or request.model, replay_origin=origin
            ),
            continuation=continuation,
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        prepared = self._responses_body(
            request, reasoning, self._effective_capabilities(model_info)
        )
        return self._stream(
            prepared.body,
            betas=(),
            endpoint_context=endpoint_context,
            request_id=request_id,
            presenter_factory=lambda origin: AnthropicToResponsesStream(
                request,
                public_model=response_model or request.model,
                tool_identities=prepared.tool_identities,
                replay_origin=origin,
            ),
            continuation=continuation,
        )

    async def _stream(
        self,
        body: JsonObject,
        *,
        betas: tuple[str, ...],
        endpoint_context: EndpointContext,
        request_id: str | None,
        presenter_factory: Callable[[ReplayOrigin], _Presenter],
        reasoning_correction: ReasoningCorrection | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(request_id=request_id)
        run = self._run(
            body,
            reasoning_correction=reasoning_correction,
            betas=betas,
            endpoint_context=endpoint_context,
            execution=execution,
            presenter_factory=presenter_factory,
            continuation=continuation,
        )
        try:
            async for event in run:
                yield event
        except asyncio.CancelledError, GeneratorExit:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            execution.succeed()
        finally:
            await maybe_await_aclose(run)
            execution.abandon()

    async def _run(
        self,
        body: JsonObject,
        *,
        betas: tuple[str, ...],
        endpoint_context: EndpointContext,
        execution: ProviderExecution,
        presenter_factory: Callable[[ReplayOrigin], _Presenter],
        reasoning_correction: ReasoningCorrection | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        recovery = RecoveryController(execution.delivery)
        request_endpoint = RequestEndpoint(endpoint_context)
        request_recovery = RequestRecovery(
            execution, endpoint=request_endpoint, stream=recovery
        )
        corrections = RequestCorrections("messages", reasoning_correction)
        continuation_request = ContinuationRequest("messages")
        operation_kind = ProviderOperationKind.GENERATION
        if continuation is not None:
            body = continuation_request.build(
                body, continuation.text, continuation.thinking
            )
            operation_kind = ProviderOperationKind.CONTINUATION
        source: SourceRecoveryState | None = None
        while execution.can_attempt:
            normal_stop_seen = False
            source = SourceRecoveryState(execution, previous=source)
            scope: ProviderAttemptScope | None = None
            stream_opened = False
            sent_body = body
            presenter: _Presenter | None = None
            try:
                attempt = await execution.open_attempt(operation_kind)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=self._provider_name,
                    request_id=execution.request_id,
                )
                endpoint = await request_endpoint.resolve()
                origin = replay_origin(
                    self._replay_scope,
                    "messages",
                    str(body["model"]),
                    endpoint=endpoint,
                )
                sent_body = prepare_history(body, origin)
                presenter = presenter_factory(origin)
                headers = httpx.Headers(
                    {
                        "anthropic-version": "2023-06-01",
                        "Accept": "text/event-stream",
                    }
                )
                headers.update(endpoint.headers)
                if endpoint.api_key and not any(
                    key.lower() in {"authorization", "x-api-key"} for key in headers
                ):
                    headers["x-api-key"] = endpoint.api_key
                if betas:
                    existing = headers.pop("anthropic-beta", "")
                    headers["anthropic-beta"] = ",".join(
                        dict.fromkeys([*filter(None, existing.split(",")), *betas])
                    )
                base_url = endpoint.base_url.rstrip("/")
                path = "/messages"
                response = scope.retain(
                    await self._client.send(
                        self._client.build_request(
                            "POST",
                            f"{base_url}{path}",
                            json=sent_body,
                            headers=headers,
                        ),
                        stream=True,
                    )
                )
                if not response.is_success:
                    raise await messages_status_error(response)
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type.lower():
                    raise RetryableProviderProtocolError(
                        "Messages upstream did not return an SSE stream."
                    )
                stream_opened = True
                async for event_type, payload in messages_events(response):
                    normal_stop_seen |= is_messages_stop(event_type, payload)
                    check_messages_failure(event_type, payload)
                    source.observe(
                        "messages",
                        payload,
                        continuing=operation_kind is ProviderOperationKind.CONTINUATION,
                    )
                    output = presenter.feed(event_type, payload)
                    if event_type != "ping" and not attempt.accepted:
                        await attempt.accept()
                    for event in (output,) if isinstance(output, str) else output:
                        for held in recovery.push(event):
                            yield held
                    if presenter.completed:
                        break
                if not presenter.completed:
                    raise RetryableProviderProtocolError(
                        "Messages stream ended without message_stop."
                    )
                for event in recovery.flush():
                    yield event
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = (
                    RetryableProviderProtocolError(str(raw_error))
                    if isinstance(raw_error, NativeMessagesError)
                    else raw_error
                )
                status = (
                    error.response.status_code
                    if isinstance(error, httpx.HTTPStatusError)
                    else error.status_code
                    if isinstance(error, ExecutionFailure)
                    else None
                )
                attempt_failure = None
                if scope is not None:
                    corrected_body = await request_recovery.retry_request(
                        error,
                        status,
                        scope.attempt,
                        body,
                        operation_kind=operation_kind,
                        normal_stop_seen=normal_stop_seen,
                        propose_correction=partial(
                            corrections.next_body,
                            raw_error,
                            body,
                            sent_body=sent_body,
                            reasoning_error=error,
                        ),
                    )
                    if corrected_body is not None:
                        body = corrected_body
                        recovery.discard()
                        continue
                if scope is not None and not scope.attempt.accepted:
                    if (
                        normal_stop_seen
                        and execution.delivery is not None
                        and not execution.delivery.content_released
                    ):
                        await scope.attempt.accept()
                    else:
                        attempt_failure = await scope.attempt.fail(error)
                if attempt_failure is not None and attempt_failure.retry_allowed:
                    recovery.discard()
                    continue
                decision = recovery.advance_failure(
                    retryable=attempt_failure.retryable
                    if attempt_failure is not None
                    else is_retryable_stream_error(error),
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                    normal_stop_seen=normal_stop_seen,
                )
                if decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
                    continue
                if scope is not None:
                    await scope.aclose(active_error=error)
                recovered = public_recovery(
                    execution,
                    body=body,
                    request=continuation_request,
                    retryable=decision.retryable,
                    normal_stop_seen=normal_stop_seen,
                    source=source,
                )
                if recovered is not None:
                    recovery.discard()
                    if recovered.body is None:
                        return
                    body = recovered.body
                    operation_kind = ProviderOperationKind.CONTINUATION
                    corrections = RequestCorrections("messages", reasoning_correction)
                    continue
                failure = classify_provider_failure(
                    error,
                    provider_name=self._provider_name,
                    read_timeout_s=self._read_timeout_s,
                    request_id=execution.request_id,
                )
                trace_event(
                    stage="provider",
                    event="provider.response.error",
                    source="provider",
                    provider=self._provider_name,
                    request_id=execution.request_id,
                    transport="messages",
                    failure_kind=failure.kind.value,
                )
                if execution.delivery is not None:
                    recovery.discard()
                    raise public_stream_failure(
                        failure,
                        source=source,
                        body=body,
                        protocol="messages",
                        normal_stop_seen=normal_stop_seen,
                        responses_failure_payload=(
                            presenter.failure_payload(failure)
                            if isinstance(presenter, AnthropicToResponsesStream)
                            else None
                        ),
                    ) from raw_error
                if decision.committed and isinstance(
                    presenter, AnthropicToResponsesStream
                ):
                    execution.fail(failure)
                    for event in presenter.terminal_failure(failure):
                        yield event
                    return
                recovery.discard()
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())
        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Messages execution ended without a terminal result.")
