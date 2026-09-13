"""Native Messages HTTP execution with one admitted recovery budget."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from typing import cast

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.errors import anthropic_status_for_error_type
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    PreparedMessagesRequest,
    build_native_messages_request,
)
from free_claude_code.core.anthropic.native_stream import NativeMessagesRelay
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
    redact_sensitive_error_text,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
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
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.endpoint import EndpointContext
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
    is_context_window_finish_reason,
    is_retryable_stream_error,
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

from .request_policy import (
    DEFAULT_MESSAGES_OUTPUT_TOKENS,
    MessagesModelCapabilities,
    resolve_messages_options,
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

    def _messages_body(
        self,
        request: MessagesRequest,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None = None,
    ) -> PreparedMessagesRequest:
        validate_history(request.model_dump(mode="json"))
        request, reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=True,
            normal_max_tokens=DEFAULT_MESSAGES_OUTPUT_TOKENS,
        )
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_tokens,
                reasoning=reasoning,
                capabilities=self._capabilities,
                thinking=request.thinking,
                output_effort=request.output_config.get("effort")
                if request.output_config
                else None,
            )
            return build_native_messages_request(request, options=options)
        except (NativeMessagesError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def _responses_body(
        self, request: OpenAIResponsesRequest, reasoning: ReasoningPolicy
    ) -> ResponsesMessagesRequest:
        validate_history(request.model_dump(mode="json"))
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_output_tokens,
                reasoning=reasoning,
                capabilities=self._capabilities,
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
    ) -> AsyncIterator[str]:
        prepared_request, wire_reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=True,
            normal_max_tokens=DEFAULT_MESSAGES_OUTPUT_TOKENS,
        )
        prepared = self._messages_body(prepared_request, wire_reasoning)
        correction = (
            ReasoningCorrection(
                (("thinking",),),
                "max_tokens",
                DEFAULT_MESSAGES_OUTPUT_TOKENS,
                self._capabilities.max_output_tokens,
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
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        prepared = self._responses_body(request, reasoning)
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
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(request_id=request_id)
        run = self._run(
            body,
            reasoning_correction=reasoning_correction,
            betas=betas,
            endpoint_context=endpoint_context,
            execution=execution,
            presenter_factory=presenter_factory,
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
    ) -> AsyncIterator[str]:
        recovery = RecoveryController()
        refreshed = False
        force_refresh = False
        while execution.can_attempt:
            scope: ProviderAttemptScope | None = None
            stream_opened = False
            sent_body = body
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=self._provider_name,
                    request_id=execution.request_id,
                )
                endpoint = await endpoint_context.endpoint(force_refresh=force_refresh)
                force_refresh = False
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
                path = "/messages" if base_url.endswith("/v1") else "/v1/messages"
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
                    raise await _status_error(response)
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type.lower():
                    raise RetryableProviderProtocolError(
                        "Messages upstream did not return an SSE stream."
                    )
                stream_opened = True
                async for event_type, payload in _events(response):
                    _check_failure(event_type, payload)
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
                if (
                    scope is not None
                    and status in {401, 403}
                    and not refreshed
                    and not recovery.committed
                ):
                    retry = (
                        execution.can_attempt
                        if scope.attempt.accepted
                        else await scope.attempt.correct(error)
                        is ProviderCorrectionAction.RETRY
                    )
                    if retry:
                        refreshed = force_refresh = True
                        recovery.discard()
                        continue
                attempt_failure = None
                if scope is not None and not recovery.committed:
                    corrected_history = history_retry_body(
                        raw_error, sent_body, "messages"
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
                    corrected_body = reasoning_correction.retry_body(error, body)
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
                if scope is not None and not scope.attempt.accepted:
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
                )
                if decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
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


async def _events(response: httpx.Response) -> AsyncIterator[tuple[str, JsonObject]]:
    decoder = AnthropicSSEDecoder()
    async for chunk in response.aiter_text():
        for event in decoder.feed(chunk):
            payload = cast(JsonObject, event.data)
            kind = event.event or payload.get("type")
            if not isinstance(kind, str) or not kind:
                raise RetryableProviderProtocolError(
                    "Messages stream has an invalid event type."
                )
            yield kind, payload
    for event in decoder.finish():
        payload = cast(JsonObject, event.data)
        kind = event.event or payload.get("type")
        if not isinstance(kind, str) or not kind:
            raise RetryableProviderProtocolError(
                "Messages stream has an invalid final event."
            )
        yield kind, payload


def _check_failure(event_type: str, payload: JsonObject) -> None:
    delta = payload.get("delta")
    if (
        event_type == "message_delta"
        and isinstance(delta, Mapping)
        and is_context_window_finish_reason(delta.get("stop_reason"))
    ):
        raise context_window_exceeded_provider_failure()
    if event_type != "error":
        return
    error = payload.get("error")
    kind = error.get("type") if isinstance(error, Mapping) else None
    if isinstance(error, Mapping) and any(
        is_context_window_error_code(error.get(key)) for key in ("type", "code")
    ):
        raise context_window_exceeded_provider_failure()
    status = anthropic_status_for_error_type(kind if isinstance(kind, str) else "")
    failure_kind = {
        400: FailureKind.INVALID_REQUEST,
        401: FailureKind.AUTHENTICATION,
        402: FailureKind.PERMISSION,
        403: FailureKind.PERMISSION,
        404: FailureKind.INVALID_REQUEST,
        413: FailureKind.INVALID_REQUEST,
        429: FailureKind.RATE_LIMIT,
        504: FailureKind.TIMEOUT,
        529: FailureKind.OVERLOADED,
    }.get(status, FailureKind.UPSTREAM)
    message = error.get("message") if isinstance(error, Mapping) else None
    failure = ExecutionFailure(
        failure_kind,
        status,
        redact_sensitive_error_text(message[:ERROR_DETAIL_DISPLAY_CAP_BYTES])
        if isinstance(message, str) and message
        else "Messages upstream returned an error.",
        status == 429 or status >= 500,
    )
    attach_upstream_error_body(failure, json.dumps(payload))
    raise failure


async def _status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    limit = ERROR_DETAIL_DISPLAY_CAP_BYTES
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk[: limit + 1 - len(body)])
        if len(body) > limit:
            break
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        attach_upstream_error_body(
            error, bytes(body[:limit]), truncated=len(body) > limit
        )
        return error
    raise AssertionError("Expected an unsuccessful Messages response.")
