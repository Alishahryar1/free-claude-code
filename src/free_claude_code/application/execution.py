"""Provider execution shared by inbound API adapters."""

import asyncio
import math
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from types import MappingProxyType
from typing import Literal

from loguru import logger

from free_claude_code.core.anthropic import (
    Message,
    SystemContent,
    Tool,
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.trace import (
    close_stream_input,
    trace_event,
    traced_async_stream,
)

from .model_metadata import ProviderModelInfo
from .ports import ProviderResolver
from .routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
    RoutedResponsesRequest,
)

TokenCounter = Callable[
    [list[Message], str | list[SystemContent] | None, list[Tool] | None],
    int,
]
ResponsesTokenCounter = Callable[[OpenAIResponsesRequest], int]
WireApi = Literal["messages", "responses"]
CandidateStreamOpener = Callable[[int, ProviderModelTarget], AsyncIterator[str]]


class ProviderExecutor:
    """Resolve a provider and execute one routed Anthropic Messages stream."""

    def __init__(
        self,
        provider_resolver: ProviderResolver,
        *,
        progress_timeout_seconds: float,
        token_counter: TokenCounter = get_token_count,
        responses_token_counter: ResponsesTokenCounter = estimate_responses_input_tokens,
        generation_id: int | None = None,
        log_raw_payloads: bool = False,
        request_headers: Mapping[str, str] | None = None,
        model_infos: tuple[ProviderModelInfo, ...] = (),
    ) -> None:
        if not math.isfinite(progress_timeout_seconds) or progress_timeout_seconds <= 0:
            raise ValueError("progress_timeout_seconds must be finite and positive")
        self._provider_resolver = provider_resolver
        self._model_infos = {info.model_id: info for info in model_infos}
        self._token_counter = token_counter
        self._responses_token_counter = responses_token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads
        self._request_headers = MappingProxyType(dict(request_headers or {}))
        self._progress_timeout_seconds = float(progress_timeout_seconds)

    def _progress_timeout_failure(
        self,
        *,
        request_id: str,
        provider_id: str,
    ) -> ExecutionFailure:
        trace_event(
            stage="execution",
            event="free_claude_code.provider.progress_timeout",
            source="application",
            request_id=request_id,
            provider_id=provider_id,
            timeout_seconds=self._progress_timeout_seconds,
        )
        timeout_text = f"{self._progress_timeout_seconds:g}"
        return ExecutionFailure(
            kind=FailureKind.TIMEOUT,
            status_code=504,
            message=(
                f"Provider execution made no progress for {timeout_text} seconds.\n\n"
                f"Request ID: {request_id}"
            ),
            retryable=False,
        )

    def _trace_fallback_started(
        self,
        *,
        request_id: str,
        wire_api: WireApi,
        failed: ProviderModelTarget,
        selected: ProviderModelTarget,
        failure: ExecutionFailure,
        candidate_index: int,
        candidate_count: int,
    ) -> None:
        fields: dict[str, object] = {
            "stage": "execution",
            "event": "free_claude_code.model_fallback.started",
            "source": "application",
            "request_id": request_id,
            "wire_api": wire_api,
            "from_provider_model_ref": failed.provider_model_ref,
            "to_provider_model_ref": selected.provider_model_ref,
            "candidate_index": candidate_index,
            "candidate_count": candidate_count,
            "failure_kind": failure.kind.value,
            "status_code": failure.status_code,
            "provider_retryable": failure.retryable,
        }
        if self._generation_id is not None:
            fields["generation_id"] = self._generation_id
        trace_event(**fields)
        logger.info(
            "Model fallback: request_id={} from={} to={} candidate={}/{} "
            "failure_kind={} status_code={}",
            request_id,
            failed.provider_model_ref,
            selected.provider_model_ref,
            candidate_index,
            candidate_count,
            failure.kind.value,
            failure.status_code,
        )

    def _trace_fallback_selected(
        self,
        *,
        request_id: str,
        wire_api: WireApi,
        selected: ProviderModelTarget,
        candidate_index: int,
        candidate_count: int,
    ) -> None:
        fields: dict[str, object] = {
            "stage": "execution",
            "event": "free_claude_code.model_fallback.selected",
            "source": "application",
            "request_id": request_id,
            "wire_api": wire_api,
            "selected_provider_model_ref": selected.provider_model_ref,
            "candidate_index": candidate_index,
            "candidate_count": candidate_count,
        }
        if self._generation_id is not None:
            fields["generation_id"] = self._generation_id
        trace_event(**fields)

    def stream_messages(
        self,
        routed: RoutedMessagesRequest,
        *,
        raw_log_payload: object,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Execute one Anthropic Messages request."""

        primary = routed.resolved.primary
        primary_provider = self._provider_resolver(primary.provider_id)
        primary_request = routed.request.model_copy(deep=True)
        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        def open_candidate(
            index: int,
            target: ProviderModelTarget,
        ) -> AsyncIterator[str]:
            provider = (
                primary_provider
                if index == 0
                else self._provider_resolver(target.provider_id)
            )
            request = (
                primary_request
                if index == 0
                else routed.request.model_copy(
                    update={"model": target.provider_model},
                    deep=True,
                )
            )
            return provider.stream_messages(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=routed.reasoning,
                model_info=self._model_infos.get(target.provider_model_ref),
                request_headers=self._request_headers,
            )

        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=anthropic_request_snapshot(routed.request),
            ingress_count_name="message_count",
            ingress_count=len(routed.request.messages),
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def stream_responses(
        self,
        routed: RoutedResponsesRequest,
        *,
        raw_log_payload: object,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Execute one native OpenAI Responses request."""

        primary = routed.resolved.primary
        primary_provider = self._provider_resolver(primary.provider_id)
        primary_request = routed.request.model_copy(deep=True)
        input_tokens = self._responses_token_counter(routed.request)

        def open_candidate(
            index: int,
            target: ProviderModelTarget,
        ) -> AsyncIterator[str]:
            provider = (
                primary_provider
                if index == 0
                else self._provider_resolver(target.provider_id)
            )
            request = (
                primary_request
                if index == 0
                else routed.request.model_copy(
                    update={"model": target.provider_model},
                    deep=True,
                )
            )
            return provider.stream_responses(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=routed.reasoning,
                request_headers=self._request_headers,
            )

        raw_input = routed.request.input
        input_item_count = (
            len(raw_input)
            if isinstance(raw_input, list)
            else int(raw_input is not None)
        )
        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="responses",
            raw_log_label="FULL_RESPONSES_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot={
                "model": routed.request.model,
                "input_item_count": input_item_count,
                "tool_count": len(routed.request.tools or ()),
            },
            ingress_count_name="input_item_count",
            ingress_count=input_item_count,
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def _stream_candidates(
        self,
        *,
        resolved: ResolvedModelRoute,
        reasoning: ReasoningPolicy,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: object,
        request_snapshot: dict[str, object],
        ingress_count_name: str,
        ingress_count: int,
        request_id: str,
        open_candidate: CandidateStreamOpener,
    ) -> AsyncIterator[str]:
        """Start and consume candidates through one protocol-blind lifecycle."""

        primary = resolved.primary
        candidates = (primary, *resolved.fallbacks)
        gateway_model = resolved.original_model
        route_trace: dict[str, object] = {
            "stage": "routing",
            "event": "free_claude_code.api.route.resolved",
            "source": "api",
            "request_id": request_id,
            "provider_id": primary.provider_id,
            "provider_model": primary.provider_model,
            "provider_model_ref": primary.provider_model_ref,
            "fallback_count": len(resolved.fallbacks),
            "gateway_model": gateway_model,
            "reasoning_control": reasoning.control.value,
            "reasoning_effort": (
                reasoning.effort.value if reasoning.effort is not None else None
            ),
            "reasoning_budget_tokens": reasoning.budget_tokens,
        }
        if wire_api == "responses":
            route_trace["wire_api"] = "responses"
        if self._generation_id is not None:
            route_trace["generation_id"] = self._generation_id
        trace_event(**route_trace)

        request_snapshot["model"] = gateway_model
        ingress_trace: dict[str, object] = {
            "stage": "ingress",
            "event": (
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            "source": "api",
            "snapshot": request_snapshot,
            "request_id": request_id,
            ingress_count_name: ingress_count,
        }
        trace_event(
            **ingress_trace,
        )

        if self._log_raw_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        async def provider_body() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            progress_deadline = loop.time() + self._progress_timeout_seconds
            for index, target in enumerate(candidates):
                provider_stream: AsyncIterator[str] | None = None
                candidate_committed = False
                candidate_failure: ExecutionFailure | None = None
                try:
                    try:
                        provider_stream = open_candidate(index, target)
                    except ExecutionFailure as failure:
                        candidate_failure = failure

                    if provider_stream is None and candidate_failure is None:
                        raise TypeError(
                            "provider stream method must return an async iterator"
                        )
                    while provider_stream is not None:
                        if loop.time() >= progress_deadline:
                            raise self._progress_timeout_failure(
                                request_id=request_id,
                                provider_id=target.provider_id,
                            )
                        progress_timeout = asyncio.timeout_at(progress_deadline)
                        read_failure: ExecutionFailure | None = None
                        try:
                            async with progress_timeout:
                                try:
                                    chunk = await anext(provider_stream)
                                except ExecutionFailure as failure:
                                    read_failure = failure
                        except StopAsyncIteration:
                            break
                        except TimeoutError as exc:
                            if not progress_timeout.expired():
                                raise
                            raise self._progress_timeout_failure(
                                request_id=request_id,
                                provider_id=target.provider_id,
                            ) from exc
                        if progress_timeout.expired():
                            raise self._progress_timeout_failure(
                                request_id=request_id,
                                provider_id=target.provider_id,
                            )
                        if read_failure is not None:
                            candidate_failure = read_failure
                            break
                        if not chunk:
                            await asyncio.sleep(0)
                            continue
                        if not candidate_committed:
                            candidate_committed = True
                            if index > 0:
                                self._trace_fallback_selected(
                                    request_id=request_id,
                                    wire_api=wire_api,
                                    selected=target,
                                    candidate_index=index + 1,
                                    candidate_count=len(candidates),
                                )
                        yield chunk
                        progress_deadline = loop.time() + self._progress_timeout_seconds
                finally:
                    if provider_stream is not None:
                        active_error = sys.exception()
                        preserved_error = active_error or candidate_failure
                        cleanup_timeout = asyncio.timeout_at(
                            progress_deadline if active_error is None else None
                        )
                        try:
                            async with cleanup_timeout:
                                await close_stream_input(
                                    provider_stream,
                                    owner="provider_executor",
                                    source="api",
                                    preserved_error=preserved_error,
                                )
                        except TimeoutError as exc:
                            if not cleanup_timeout.expired():
                                raise
                            raise self._progress_timeout_failure(
                                request_id=request_id,
                                provider_id=target.provider_id,
                            ) from exc

                if candidate_failure is None:
                    return
                if candidate_committed or index + 1 >= len(candidates):
                    raise candidate_failure
                next_target = candidates[index + 1]
                self._trace_fallback_started(
                    request_id=request_id,
                    wire_api=wire_api,
                    failed=target,
                    selected=next_target,
                    failure=candidate_failure,
                    candidate_index=index + 2,
                    candidate_count=len(candidates),
                )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "initial_provider_id": primary.provider_id,
            "gateway_model": gateway_model,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            provider_body(),
            stage="egress",
            source="api",
            complete_event=(
                "free_claude_code.api.responses.stream_completed"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_completed"
            ),
            interrupted_event=(
                "free_claude_code.api.responses.stream_interrupted"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_interrupted"
            ),
            chunk_event=None,
            extra=stream_trace,
        )
