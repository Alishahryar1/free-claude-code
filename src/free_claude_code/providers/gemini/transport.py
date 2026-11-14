"""Native GenerateContent attempts using the shared execution/recovery owners."""

import asyncio
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
from free_claude_code.core.stream_output import StreamOutput, completed_output
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.attempt_loop import (
    AttemptContext,
    AttemptPlan,
    TransportAttempt,
    run_attempts,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.history_replay import ReplayOrigin, replay_origin
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.openai_chat import ChatStreamOutput
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection
from free_claude_code.providers.recovery_errors import RetryableProviderProtocolError
from free_claude_code.providers.request_correction import (
    RequestCorrector,
)
from free_claude_code.providers.request_recovery import (
    RequestRecovery,
)
from free_claude_code.providers.stream_failure_resolution import (
    StreamFailurePolicy,
    StreamFailureResolver,
)
from free_claude_code.providers.stream_read import read_stream_event

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
    ) -> AsyncIterator[StreamOutput]:
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

    def _run(
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
    ) -> AsyncIterator[StreamOutput]:
        endpoint = RequestEndpoint(endpoint_context)
        continuation_request = ContinuationRequest("gemini")
        operation = ProviderOperationKind.GENERATION
        if continuation is not None:
            body = continuation_request.build(body, continuation)
            operation = ProviderOperationKind.CONTINUATION
        resolver = StreamFailureResolver(
            StreamFailurePolicy(
                provider_name="GEMINI",
                read_timeout_s=self._read_timeout,
                request_id=execution.request_id,
                # History the caller sent cannot be replayed; that is their error.
                raise_unfinalized=(InvalidRequestError,),
            ),
            recovery=RequestRecovery(execution, endpoint=endpoint),
            continuation=continuation_request,
        )
        return run_attempts(
            AttemptPlan(
                body=body,
                operation_kind=operation,
                execution=execution,
                resolver=resolver,
                corrector=partial(
                    RequestCorrector, "gemini", reasoning=reasoning_correction
                ),
                start=partial(
                    _GeminiAttempt,
                    transport=self,
                    endpoint=endpoint,
                    continuation=continuation_request,
                    response_model=response_model,
                    output_factory=output_factory,
                    include_thoughts=include_thoughts,
                ),
                provider_name="GEMINI",
                transport="gemini",
                gateway_model=response_model,
            )
        )


class _GeminiAttempt(TransportAttempt):
    """One GenerateContent attempt."""

    def __init__(
        self,
        context: AttemptContext,
        *,
        transport: GeminiTransport,
        endpoint: RequestEndpoint,
        continuation: ContinuationRequest,
        response_model: str,
        output_factory: Callable[[], ChatStreamOutput | None],
        include_thoughts: bool,
    ) -> None:
        super().__init__(context)
        self._transport = transport
        self._endpoint = endpoint
        self._continuation = continuation
        self._response_model = response_model
        self._output_factory = output_factory
        self._include_thoughts = include_thoughts
        self._presenter: GeminiStream | None = None
        self._request: httpx.Request | None = None
        self._origin: ReplayOrigin | None = None

    @property
    def normal_stop_seen(self) -> bool:
        return self._presenter is not None and self._presenter.normal_stop_seen

    async def prepare(self) -> None:
        context = self.context
        snapshot = await self._endpoint.resolve()
        self._origin = replay_origin(
            "GEMINI", "gemini", str(self.body["model"]), endpoint=snapshot
        )
        try:
            self.sent_body = context.corrections.prepare_history(
                self.body,
                self._origin,
                self._transport._history_cache,
                continuation=context.continuing,
            )
        except HistoryReplayError as error:
            raise InvalidRequestError(str(error)) from error
        model = str(self.sent_body["model"]).removeprefix("models/")
        headers = {"Accept": "text/event-stream", **snapshot.headers}
        if snapshot.api_key:
            headers["x-goog-api-key"] = snapshot.api_key
        self._request = self._transport._client.build_request(
            "POST",
            f"{snapshot.base_url.rstrip('/')}/models/{quote(model, safe='')}:streamGenerateContent",
            params={"alt": "sse"},
            headers=headers,
            json={
                key: value for key, value in self.sent_body.items() if key != "model"
            },
        )

    async def stream(self, scope: ProviderAttemptScope) -> AsyncIterator[StreamOutput]:
        assert self._request is not None and self._origin is not None
        corrections = self.context.corrections
        assert corrections.history is not None
        client = self._transport._client
        response = scope.retain(await client.send(self._request, stream=True))
        if not response.is_success:
            await response.aread()
            response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", "").lower():
            raise RetryableProviderProtocolError(
                "Gemini upstream did not return an SSE stream"
            )
        declarations = [
            function
            for tool in self.body.get("tools", [])
            if isinstance(tool, dict)
            for function in tool.get("functionDeclarations", [])
            if isinstance(function, dict)
        ]
        presenter = self._presenter = GeminiStream(
            model=self._response_model,
            origin=self._origin,
            output=self._output_factory(),
            tool_names=frozenset(str(tool["name"]) for tool in declarations),
            tool_defaults={
                str(tool["name"]): {
                    name: schema["default"]
                    for name, schema in (
                        tool.get("parametersJsonSchema") or tool.get("parameters") or {}
                    )
                    .get("properties", {})
                    .items()
                    if isinstance(schema, dict) and "default" in schema
                }
                for tool in declarations
            },
            include_thoughts=self._include_thoughts,
            continuation_context=self._continuation.sent_context(
                self.sent_body,
                source=corrections.history.source,
                source_paths=corrections.history.paths,
            ),
        )
        upstream = _events(response, read_timeout=self._transport._read_timeout)
        try:
            async for payload in upstream:
                self.context.source.observe(
                    "gemini", payload, continuing=self.context.continuing
                )
                output = presenter.feed(payload)
                corrections.accepted_history()
                if not scope.attempt.accepted:
                    await scope.attempt.accept()
                if output:
                    yield completed_output(output)
        finally:
            await maybe_await_aclose(upstream)
        if not presenter.normal_stop_seen:
            raise RetryableProviderProtocolError(
                "Gemini stream ended without finishReason"
            )
        for event in presenter.finish():
            yield StreamOutput((event,))
