"""Provider execution shared by inbound API adapters."""

import json
import math
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from types import MappingProxyType
from typing import Literal

from loguru import logger

from free_claude_code.core.anthropic import (
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.anthropic.recovery_stream import (
    MessagesRecoveryWriter,
    NativeMessagesCompletionWriter,
)
from free_claude_code.core.anthropic.tokens import TokenCounter
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesRecoveryWriter,
    estimate_responses_input_tokens,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.token_estimation import estimate_text_tokens
from free_claude_code.core.trace import (
    trace_event,
    traced_async_stream,
)

from .ports import ModelInfoLookup, ProviderCandidate, ProviderResolver
from .recovery import CandidateOpener, RecoveryCoordinator, RecoveryWriter
from .routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
    RoutedNativeMessagesRequest,
    RoutedResponsesRequest,
)

ResponsesTokenCounter = Callable[[OpenAIResponsesRequest], int]
WireApi = Literal["messages", "responses"]


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
        model_info_lookup: ModelInfoLookup | None = None,
    ) -> None:
        if not math.isfinite(progress_timeout_seconds) or progress_timeout_seconds <= 0:
            raise ValueError("progress_timeout_seconds must be finite and positive")
        self._provider_resolver = provider_resolver
        self._model_info_lookup = model_info_lookup or (lambda _provider, _model: None)
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

    def stream_native_messages(
        self,
        routed: RoutedNativeMessagesRequest,
        *,
        request_id: str,
    ) -> AsyncIterator[str]:
        async def open_candidate(
            index: int, target: ProviderModelTarget
        ) -> AbstractAsyncContextManager[ProviderCandidate]:
            provider = await self._provider_resolver(target.provider_id)
            return provider.open_native_messages(
                routed.request.with_model(target.provider_model),
                request_id=request_id,
                response_model=routed.resolved.original_model,
                request_headers=self._request_headers,
            )

        messages = routed.request.body["messages"]
        assert isinstance(messages, list)
        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=ReasoningPolicy.provider_default(),
            wire_api="messages",
            raw_log_label="FULL_NATIVE_MESSAGES_PAYLOAD",
            raw_log_payload=lambda: routed.request.body,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "message_count": len(messages),
                "contract": "native",
            },
            ingress_count_name="message_count",
            ingress_count=len(messages),
            request_id=request_id,
            open_candidate=open_candidate,
            writer=(
                MessagesRecoveryWriter(
                    model=routed.resolved.original_model,
                    input_tokens=estimate_text_tokens(
                        json.dumps(routed.request.body, ensure_ascii=False)
                    ),
                    native=True,
                )
                if routed.request.stream
                else NativeMessagesCompletionWriter(
                    model=routed.resolved.original_model
                )
            ),
        )

    def stream_messages(
        self,
        routed: RoutedMessagesRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[str]:
        """Execute one Anthropic Messages request."""

        primary_request = routed.request.model_copy(deep=True)
        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        async def open_candidate(
            index: int,
            target: ProviderModelTarget,
        ) -> AbstractAsyncContextManager[ProviderCandidate]:
            provider = await self._provider_resolver(target.provider_id)
            request = (
                primary_request
                if index == 0
                else routed.request.model_copy(
                    update={"model": target.provider_model},
                    deep=True,
                )
            )
            return provider.open_messages(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=routed.reasoning,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
                request_headers=self._request_headers,
            )

        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: anthropic_request_snapshot(routed.request),
            ingress_count_name="message_count",
            ingress_count=len(routed.request.messages),
            request_id=request_id,
            open_candidate=open_candidate,
            writer=MessagesRecoveryWriter(
                model=routed.resolved.original_model, input_tokens=input_tokens
            ),
        )

    def stream_responses(
        self,
        routed: RoutedResponsesRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[str]:
        """Execute one native OpenAI Responses request."""

        primary_request = routed.request.model_copy(deep=True)
        input_tokens = self._responses_token_counter(routed.request)

        async def open_candidate(
            index: int,
            target: ProviderModelTarget,
        ) -> AbstractAsyncContextManager[ProviderCandidate]:
            provider = await self._provider_resolver(target.provider_id)
            request = (
                primary_request
                if index == 0
                else routed.request.model_copy(
                    update={"model": target.provider_model},
                    deep=True,
                )
            )
            return provider.open_responses(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=routed.reasoning,
                request_headers=self._request_headers,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
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
            request_snapshot=lambda: {
                "model": routed.request.model,
                "input_item_count": input_item_count,
                "tool_count": len(routed.request.tools or ()),
            },
            ingress_count_name="input_item_count",
            ingress_count=input_item_count,
            request_id=request_id,
            open_candidate=open_candidate,
            writer=ResponsesRecoveryWriter(
                model=routed.resolved.original_model, input_tokens=input_tokens
            ),
        )

    def _stream_candidates(
        self,
        *,
        resolved: ResolvedModelRoute,
        reasoning: ReasoningPolicy,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: Callable[[], object],
        request_snapshot: Callable[[], dict[str, object]],
        ingress_count_name: str,
        ingress_count: int,
        request_id: str,
        open_candidate: CandidateOpener,
        writer: RecoveryWriter,
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

        ingress_trace: dict[str, object] = {
            "stage": "ingress",
            "event": (
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            "source": "api",
            "request_id": request_id,
            ingress_count_name: ingress_count,
        }
        trace_event(
            lambda: {"snapshot": {**request_snapshot(), "model": gateway_model}},
            **ingress_trace,
        )

        if self._log_raw_payloads:
            logger.opt(lazy=True).debug(
                f"{raw_log_label} [{{}}]: {{}}", lambda: request_id, raw_log_payload
            )

        coordinator = RecoveryCoordinator(
            candidates=candidates,
            opener=open_candidate,
            writer=writer,
            progress_timeout_seconds=self._progress_timeout_seconds,
            timeout_failure=lambda provider_id: self._progress_timeout_failure(
                request_id=request_id,
                provider_id=provider_id,
            ),
            request_id=request_id,
            on_fallback=lambda failed, selected, failure, index: (
                self._trace_fallback_started(
                    request_id=request_id,
                    wire_api=wire_api,
                    failed=failed,
                    selected=selected,
                    failure=failure,
                    candidate_index=index + 1,
                    candidate_count=len(candidates),
                )
            ),
            on_selected=lambda selected, index: self._trace_fallback_selected(
                request_id=request_id,
                wire_api=wire_api,
                selected=selected,
                candidate_index=index + 1,
                candidate_count=len(candidates),
            ),
        )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "initial_provider_id": primary.provider_id,
            "gateway_model": gateway_model,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            coordinator.stream(),
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
