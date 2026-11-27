"""Provider execution shared by inbound API adapters."""

import json
import math
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from types import MappingProxyType

from loguru import logger

from free_claude_code.application.key_rotation import (
    RotationScope,
    configured_keys,
    error_chain,
    key_rotation_registry,
    rotation_scope,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.anthropic.tokens import TokenCounter
from free_claude_code.core.failure_evidence import tpm_limit_and_requested
from free_claude_code.core.failures import ExecutionFailure, FailureKind
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


def _estimate_request_tokens(payload: object) -> int:
    """Conservative size estimate (~4 characters per token) of an ingress payload."""
    try:
        return len(json.dumps(payload, default=str)) // 4
    except TypeError, ValueError:
        return 0


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
        settings: Settings | None = None,
    ) -> None:
        self._provider_resolver = provider_resolver
        self._model_info_lookup = model_info_lookup or (lambda _provider, _model: None)
        self._token_counter = token_counter
        self._responses_token_counter = responses_token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads
        self._request_headers = MappingProxyType(dict(request_headers or {}))
        self._settings = settings

    def _rotation_candidates(
        self,
        resolved: ResolvedModelRoute,
        request_tokens: Callable[[], int] | None = None,
    ) -> tuple[ProviderModelCandidate, ...]:
        original = (ProviderModelCandidate(resolved.primary), *resolved.fallbacks)
        if self._settings is None:
            return original
        expanded = self._expand_candidates(resolved, original)
        if not expanded and self._cooling_is_advertised(resolved, original):
            # Providers advertised these waits, and FCC's provider admission
            # tracks them and answers with the preserved upstream detail.
            return original
        if request_tokens is None or not any(
            key_rotation_registry.tpm_limit(item.target.provider_id) is not None
            for item in expanded
        ):
            return expanded
        size = request_tokens()
        # Skip providers whose observed token-rate cap is below this request's
        # size (it can never be accepted there), but never drop every route.
        fitting = tuple(
            item
            for item in expanded
            if (limit := key_rotation_registry.tpm_limit(item.target.provider_id))
            is None
            or size <= limit
        )
        return fitting or expanded

    def _expand_candidates(
        self,
        resolved: ResolvedModelRoute,
        original: tuple[ProviderModelCandidate, ...],
    ) -> tuple[ProviderModelCandidate, ...]:
        assert self._settings is not None
        key_map: dict[str, list[tuple[int, str]]] = {}
        for provider_id, number, key in configured_keys(self._settings):
            key_map.setdefault(provider_id, []).append((number, key))
        expanded: list[ProviderModelCandidate] = []
        seen: set[tuple[str, str, int]] = set()
        for candidate in original:
            target = candidate.target
            keys = key_map.get(target.provider_id, [])
            same_provider = [
                item
                for item in keys
                if item[0] != target.api_key_number
                and key_rotation_registry.available(
                    target.provider_id, item[0], item[1]
                )
            ]
            primary_key = next(
                (key for number, key in keys if number == target.api_key_number), None
            )
            primary_available = primary_key is None or key_rotation_registry.available(
                target.provider_id, target.api_key_number, primary_key
            )
            if primary_available:
                identity = (
                    target.provider_id,
                    target.provider_model,
                    target.api_key_number,
                )
                if identity in seen:
                    existing = next(
                        position
                        for position, item in enumerate(expanded)
                        if (
                            item.target.provider_id,
                            item.target.provider_model,
                            item.target.api_key_number,
                        )
                        == identity
                    )
                    if expanded[existing].is_key_rotation:
                        expanded[existing] = candidate
                else:
                    expanded.append(candidate)
                    seen.add(identity)
            for key_index, (number, _key) in enumerate(same_provider):
                identity = (target.provider_id, target.provider_model, number)
                replacement = ProviderModelCandidate(
                    replace(target, api_key_number=number),
                    candidate.reasoning_preference,
                    primary_available or key_index > 0,
                )
                if identity not in seen:
                    expanded.append(replacement)
                    seen.add(identity)
                elif not replacement.is_key_rotation:
                    for position, item in enumerate(expanded):
                        if (
                            item.target.provider_id,
                            item.target.provider_model,
                            item.target.api_key_number,
                        ) == identity and item.is_key_rotation:
                            expanded[position] = replacement
                            break
        logger.info(
            "API key rotation candidates selected: provider={} candidates={}",
            resolved.primary.provider_id,
            len(expanded),
        )
        return tuple(expanded)

    def _cooling_is_advertised(
        self,
        resolved: ResolvedModelRoute,
        original: tuple[ProviderModelCandidate, ...],
    ) -> bool:
        assert self._settings is not None
        providers = dict.fromkeys(item.target.provider_id for item in original)
        return all(
            key_rotation_registry.provider_advertised(
                provider,
                [
                    (number, key)
                    for owner, number, key in configured_keys(self._settings)
                    if owner == provider
                ],
            )
            for provider in providers
        )

    def _all_cooling_failure(self, resolved: ResolvedModelRoute) -> ExecutionFailure:
        """429 for a request whose every route has only cooling credentials."""
        assert self._settings is not None
        providers = [
            resolved.primary.provider_id,
            *(item.target.provider_id for item in resolved.fallbacks),
        ]
        waits = {
            provider: key_rotation_registry.remaining(
                provider,
                [
                    (number, key)
                    for owner, number, key in configured_keys(self._settings)
                    if owner == provider
                ],
            )
            for provider in dict.fromkeys(providers)
        }
        active = {provider: wait for provider, wait in waits.items() if wait > 0}
        provider = (
            min(active, key=lambda item: active[item]) if active else providers[0]
        )
        seconds = math.ceil(active[provider]) if active else 1
        return ExecutionFailure(
            FailureKind.RATE_LIMIT,
            429,
            key_rotation_registry.cooldown_message(provider, seconds),
            False,
        )

    def _record_key_failure(
        self, candidate: ProviderModelCandidate, error: Exception
    ) -> None:
        if self._settings is None:
            return
        target = candidate.target
        key = next(
            (
                key
                for provider, number, key in configured_keys(self._settings)
                if provider == target.provider_id and number == target.api_key_number
            ),
            None,
        )
        scope = rotation_scope(error)
        if key is None or scope is None:
            return
        if isinstance(error, ExecutionFailure):
            key_rotation_registry.remember_detail(target.provider_id, error.message)
        fallback = self._settings.api_key_cooldown_fallback_seconds
        if scope is RotationScope.PROVIDER:
            keys = [
                (number, value)
                for provider, number, value in configured_keys(self._settings)
                if provider == target.provider_id
            ]
            key_rotation_registry.mark_provider(
                target.provider_id, keys, error, fallback_seconds=fallback
            )
            parsed = next(
                (
                    found
                    for item in error_chain(error)
                    if isinstance(item, Exception)
                    and (found := tpm_limit_and_requested(item)) is not None
                ),
                None,
            )
            if parsed is not None and parsed[1] > parsed[0]:
                key_rotation_registry.learn_tpm_limit(target.provider_id, parsed[0])
        else:
            key_rotation_registry.mark(
                target.provider_id,
                target.api_key_number,
                key,
                error,
                fallback_seconds=fallback,
            )
        logger.warning(
            "API key cooldown applied: provider={} key_number={} scope={}",
            target.provider_id,
            target.api_key_number,
            scope.value,
        )

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
                if target == routed.resolved.primary
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
                if target == routed.resolved.primary
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
        candidates = self._rotation_candidates(
            resolved, lambda: _estimate_request_tokens(raw_log_payload())
        )
        gateway_model = resolved.original_model
        route_trace: dict[str, object] = {
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
        trace_event(
            stage="routing",
            event="free_claude_code.api.route.resolved",
            source="api",
            **route_trace,
        )

        trace_event(
            lambda: {"snapshot": {**request_snapshot(), "model": gateway_model}},
            stage="ingress",
            event=(
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            source="api",
            request_id=request_id,
            **{ingress_count_name: ingress_count},
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
                on_failure=self._record_key_failure,
                unavailable=lambda: self._all_cooling_failure(resolved),
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
