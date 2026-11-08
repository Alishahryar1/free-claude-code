"""Native GenerateContent attempts using the shared execution/recovery owners."""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from functools import partial
from typing import Any
from urllib.parse import quote

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.stream_contracts import SSEEvent
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.history_replay import HistoryReplayError
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.continuation import (
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
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.history_replay import replay_origin
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.openai_chat import ChatStreamOutput
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
    RequestRecovery,
)
from free_claude_code.providers.stream_read import read_stream_event
from free_claude_code.providers.stream_recovery import (
    HoldbackSignal,
    RecoveryController,
    RecoveryFailureAction,
)

from .stream import GeminiStream


async def _events(
    response: httpx.Response, *, read_timeout: float
) -> AsyncIterator[dict[str, Any]]:
    async with aclosing(_decoded_events(response)) as source:
        while True:
            try:
                event = await read_stream_event(
                    source,
                    timeout=read_timeout,
                    is_keepalive=lambda event: (
                        event.event == "ping" or event.data.get("type") == "ping"
                    ),
                )
            except StopAsyncIteration:
                return
            yield event.data


async def _decoded_events(response: httpx.Response) -> AsyncGenerator[SSEEvent]:
    decoder = AnthropicSSEDecoder()
    async for chunk in response.aiter_text():
        for event in decoder.feed(chunk):
            yield event
    for event in decoder.finish():
        yield event


class GeminiTransport:
    def __init__(
        self,
        client: httpx.AsyncClient,
        admission: ProviderAdmissionController,
        history_cache: HistoryModeCache,
        *,
        read_timeout: float,
    ) -> None:
        self._client = client
        self._admission = admission
        self._history_cache = history_cache
        self._read_timeout = read_timeout

    async def stream(
        self,
        body: JsonObject,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None,
        response_model: str,
        output_factory: Callable[[], ChatStreamOutput | None],
        include_thoughts: bool = True,
        reasoning_correction: ReasoningCorrection | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(
            request_id=request_id, model=str(body["model"])
        )
        run = self._run(
            body,
            execution=execution,
            endpoint_context=endpoint_context,
            response_model=response_model,
            output_factory=output_factory,
            include_thoughts=include_thoughts,
            reasoning_correction=reasoning_correction,
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
        execution: ProviderExecution,
        endpoint_context: EndpointContext,
        response_model: str,
        output_factory: Callable[[], ChatStreamOutput | None],
        include_thoughts: bool,
        reasoning_correction: ReasoningCorrection | None,
        continuation: ContinuationSeed | None,
    ) -> AsyncIterator[str]:
        recovery = RecoveryController(execution.delivery)
        endpoint = RequestEndpoint(endpoint_context)
        request_recovery = RequestRecovery(
            execution, endpoint=endpoint, stream=recovery
        )
        corrections = RequestCorrections("gemini", reasoning=reasoning_correction)
        continuation_request = ContinuationRequest("gemini")
        operation = ProviderOperationKind.GENERATION
        if continuation is not None:
            body = continuation_request.build(body, continuation)
            operation = ProviderOperationKind.CONTINUATION
        source: SourceRecoveryState | None = None
        while execution.can_attempt:
            source = SourceRecoveryState(execution, previous=source)
            scope: ProviderAttemptScope | None = None
            presenter: GeminiStream | None = None
            sent_body = body
            opened = False
            try:
                attempt = await execution.open_attempt(operation)
                scope = ProviderAttemptScope(
                    attempt, provider_name="GEMINI", request_id=execution.request_id
                )
                snapshot = await endpoint.resolve()
                origin = replay_origin(
                    "GEMINI", "gemini", str(body["model"]), endpoint=snapshot
                )
                try:
                    sent_body = corrections.prepare_history(
                        body,
                        origin,
                        self._history_cache,
                        continuation=operation is ProviderOperationKind.CONTINUATION,
                    )
                except HistoryReplayError as error:
                    raise InvalidRequestError(str(error)) from error
                assert corrections.history is not None
                model = str(sent_body["model"]).removeprefix("models/")
                headers = {"Accept": "text/event-stream", **snapshot.headers}
                if snapshot.api_key:
                    headers["x-goog-api-key"] = snapshot.api_key
                response = scope.retain(
                    await self._client.send(
                        self._client.build_request(
                            "POST",
                            f"{snapshot.base_url.rstrip('/')}/models/{quote(model, safe='')}:streamGenerateContent",
                            params={"alt": "sse"},
                            headers=headers,
                            json={
                                key: value
                                for key, value in sent_body.items()
                                if key != "model"
                            },
                        ),
                        stream=True,
                    )
                )
                if not response.is_success:
                    await response.aread()
                    response.raise_for_status()
                if (
                    "text/event-stream"
                    not in response.headers.get("content-type", "").lower()
                ):
                    raise RetryableProviderProtocolError(
                        "Gemini upstream did not return an SSE stream"
                    )
                opened = True
                declarations = [
                    function
                    for tool in body.get("tools", [])
                    if isinstance(tool, dict)
                    for function in tool.get("functionDeclarations", [])
                    if isinstance(function, dict)
                ]
                presenter = GeminiStream(
                    model=response_model,
                    origin=origin,
                    output=output_factory(),
                    tool_names=frozenset(str(tool["name"]) for tool in declarations),
                    tool_defaults={
                        str(tool["name"]): {
                            name: schema["default"]
                            for name, schema in (
                                tool.get("parametersJsonSchema")
                                or tool.get("parameters")
                                or {}
                            )
                            .get("properties", {})
                            .items()
                            if isinstance(schema, dict) and "default" in schema
                        }
                        for tool in declarations
                    },
                    include_thoughts=include_thoughts,
                    continuation_context=continuation_request.sent_context(
                        sent_body,
                        source=corrections.history.source,
                        source_paths=corrections.history.paths,
                    ),
                )
                upstream = _events(response, read_timeout=self._read_timeout)
                try:
                    async with recovery.read_stream(upstream) as events:
                        async for payload in events:
                            if payload is HoldbackSignal.EXPIRED:
                                for event in recovery.flush():
                                    yield event
                                continue
                            source.observe(
                                "gemini",
                                payload,
                                continuing=operation
                                is ProviderOperationKind.CONTINUATION,
                            )
                            output = presenter.feed(payload)
                            corrections.accepted_history()
                            if not attempt.accepted:
                                await attempt.accept()
                            for event in output:
                                for held in recovery.push(event):
                                    yield held
                finally:
                    await maybe_await_aclose(upstream)
                if not presenter.normal_stop_seen:
                    raise RetryableProviderProtocolError(
                        "Gemini stream ended without finishReason"
                    )
                for event in presenter.finish():
                    for held in recovery.push(event):
                        yield held
                for event in recovery.flush():
                    yield event
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as error:
                normal_stop = presenter.normal_stop_seen if presenter else False
                attempt_failure = None
                if scope is not None:
                    corrected = await request_recovery.retry_request(
                        error,
                        error.response.status_code
                        if isinstance(error, httpx.HTTPStatusError)
                        else None,
                        scope.attempt,
                        body,
                        operation_kind=operation,
                        normal_stop_seen=normal_stop,
                        propose_correction=partial(
                            corrections.next_body,
                            error,
                            body,
                            sent_body=sent_body,
                            reasoning_error=error,
                            reasoning_sent_body=sent_body,
                        ),
                    )
                    if corrected is not None:
                        body = corrected
                        recovery.discard()
                        continue
                    if not scope.attempt.accepted:
                        attempt_failure = await scope.attempt.fail(error)
                if attempt_failure is not None and attempt_failure.retry_allowed:
                    recovery.discard()
                    continue
                decision = recovery.advance_failure(
                    retryable=attempt_failure.retryable
                    if attempt_failure
                    else is_retryable_stream_error(error),
                    stream_opened=opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                    normal_stop_seen=normal_stop,
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
                    normal_stop_seen=normal_stop,
                    source=source,
                )
                if recovered is not None:
                    recovery.discard()
                    if recovered.body is None:
                        return
                    body = recovered.body
                    operation = ProviderOperationKind.CONTINUATION
                    corrections = RequestCorrections("gemini")
                    continue
                if isinstance(error, InvalidRequestError):
                    raise
                failure = classify_provider_failure(
                    error,
                    provider_name="GEMINI",
                    read_timeout_s=self._read_timeout,
                    request_id=execution.request_id,
                )
                recovery.discard()
                if execution.delivery is not None:
                    raise public_stream_failure(
                        failure,
                        source=source,
                        body=body,
                        protocol="gemini",
                        normal_stop_seen=normal_stop,
                    ) from error
                raise failure from error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())
        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Gemini execution ended without a terminal result")
