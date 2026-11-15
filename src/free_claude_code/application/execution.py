"""Provider execution shared by inbound API adapters."""

import json
from collections.abc import AsyncIterator, Callable, Mapping
from types import MappingProxyType

from loguru import logger

from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic import (
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.anthropic.tokens import TokenCounter
from free_claude_code.core.gemini.tokens import estimate_gemini_input_tokens
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.core.token_estimation import estimate_text_tokens
from free_claude_code.core.trace import (
    trace_event,
    traced_async_stream,
)

from .model_fallback import CandidateStreamOpener, WireApi, stream_model_candidates
from .ports import ModelInfoLookup, ProviderResolver
from .reasoning import (
    client_chat_reasoning_policy,
    client_reasoning_policy,
    client_responses_reasoning_policy,
    prepare_fallback_request,
    resolve_fallback_reasoning,
)
from .routing import (
    ProviderModelCandidate,
    ResolvedModelRoute,
    RoutedChatRequest,
    RoutedGeminiRequest,
    RoutedMessagesRequest,
    RoutedNativeMessagesRequest,
    RoutedResponsesRequest,
)

ResponsesTokenCounter = Callable[[OpenAIResponsesRequest], int]


class ProviderExecutor:
    """Prepare routed requests and compose their model execution streams."""

    def __init__(
        self,
        provider_resolver: ProviderResolver,
        *,
        token_counter: TokenCounter = get_token_count,
        responses_token_counter: ResponsesTokenCounter = estimate_responses_input_tokens,
        generation_id: int | None = None,
        log_raw_payloads: bool = False,
        request_headers: Mapping[str, str] | None = None,
        model_info_lookup: ModelInfoLookup | None = None,
    ) -> None:
        self._provider_resolver = provider_resolver
        self._model_info_lookup = model_info_lookup or (lambda _provider, _model: None)
        self._token_counter = token_counter
        self._responses_token_counter = responses_token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads
        self._request_headers = MappingProxyType(dict(request_headers or {}))

    def stream_native_messages(
        self,
        routed: RoutedNativeMessagesRequest,
        *,
        request_id: str,
    ) -> AsyncIterator[StreamOutput]:
        native_reasoning = ReasoningPolicy.provider_default()

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[StreamOutput]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            return provider.stream_native_messages(
                routed.request.with_model(target.provider_model),
                request_id=request_id,
                response_model=routed.resolved.original_model,
                request_headers=self._request_headers,
                reasoning_override=(
                    None
                    if candidate.reasoning_preference
                    in {ReasoningPreference.INHERIT, ReasoningPreference.CLIENT}
                    else resolve_fallback_reasoning(
                        native_reasoning,
                        candidate.reasoning_preference,
                        client=native_reasoning,
                    )
                ),
                continuation=continuation,
            )

        messages = routed.request.body["messages"]
        assert isinstance(messages, list)
        return self._response_stream(
            resolved=routed.resolved,
            reasoning=native_reasoning,
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
        )

    def stream_messages(
        self,
        routed: RoutedMessagesRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[StreamOutput]:
        """Execute one Anthropic Messages request."""

        primary_request = routed.request.model_copy(deep=True)
        client_reasoning = client_reasoning_policy(routed.request)
        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[StreamOutput]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = (
                primary_request
                if index == 0
                else prepare_fallback_request(
                    routed.request,
                    model=target.provider_model,
                    preference=candidate.reasoning_preference,
                )
            )
            return provider.stream_messages(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
                request_headers=self._request_headers,
                continuation=continuation,
            )

        return self._response_stream(
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
        )

    def stream_responses(
        self,
        routed: RoutedResponsesRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[StreamOutput]:
        """Execute one native OpenAI Responses request."""

        primary_request = routed.request.model_copy(deep=True)
        client_reasoning = client_responses_reasoning_policy(routed.request)
        input_tokens = self._responses_token_counter(routed.request)

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[StreamOutput]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = (
                primary_request
                if index == 0
                else prepare_fallback_request(
                    routed.request,
                    model=target.provider_model,
                    preference=candidate.reasoning_preference,
                )
            )
            return provider.stream_responses(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                request_headers=self._request_headers,
                continuation=continuation,
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
        return self._response_stream(
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
        )

    def stream_gemini(
        self,
        routed: RoutedGeminiRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[StreamOutput]:
        client_reasoning = routed.request.reasoning_policy()
        input_tokens = estimate_gemini_input_tokens(routed.request)

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[StreamOutput]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = prepare_fallback_request(
                routed.request,
                model=target.provider_model,
                preference=(
                    routed.resolved.reasoning_preference
                    if candidate.reasoning_preference is ReasoningPreference.INHERIT
                    else candidate.reasoning_preference
                ),
            )
            return provider.stream_gemini(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                request_headers=self._request_headers,
                continuation=continuation,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
            )

        return self._response_stream(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="gemini",
            raw_log_label="FULL_GEMINI_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "content_count": len(routed.request.contents),
            },
            ingress_count_name="content_count",
            ingress_count=len(routed.request.contents),
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def stream_chat(
        self,
        routed: RoutedChatRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[StreamOutput]:
        client_reasoning = client_chat_reasoning_policy(routed.request)
        input_tokens = sum(
            estimate_text_tokens(json.dumps(message)[:1_000_000])
            for message in routed.request.messages
        )

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[StreamOutput]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = prepare_fallback_request(
                routed.request,
                model=target.provider_model,
                preference=candidate.reasoning_preference,
            )
            return provider.stream_chat(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                request_headers=self._request_headers,
                continuation=continuation,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
            )

        return self._response_stream(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="chat",
            raw_log_label="FULL_CHAT_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "message_count": len(routed.request.messages),
            },
            ingress_count_name="message_count",
            ingress_count=len(routed.request.messages),
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def _response_stream(
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
        open_candidate: CandidateStreamOpener,
    ) -> AsyncIterator[StreamOutput]:
        """Compose request observations and the candidate response stream."""

        primary = resolved.primary
        candidates = (ProviderModelCandidate(primary), *resolved.fallbacks)
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
        if wire_api != "messages":
            route_trace["wire_api"] = wire_api
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

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "initial_provider_id": primary.provider_id,
            "gateway_model": gateway_model,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            stream_model_candidates(
                candidates,
                open_candidate,
                get_delivery=current_stream_delivery,
                request_id=request_id,
                wire_api=wire_api,
                generation_id=self._generation_id,
            ),
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
