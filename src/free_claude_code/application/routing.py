"""Model routing for Claude-compatible requests."""

from dataclasses import dataclass, replace

from loguru import logger
from pydantic import ValidationError

from free_claude_code.application.errors import (
    InvalidRequestError,
    UnknownProviderError,
)
from free_claude_code.config.model_refs import (
    is_retired_model_ref,
    parse_model_name,
    parse_provider_type,
    replace_retired_model_ref,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.routing_policy import parse_candidates
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    MessagesRequest,
    NativeTokenCountRequest,
    TokenCountRequest,
    get_token_count,
)
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.gateway_model_ids import (
    DESKTOP_MODEL_PREFIX,
    DESKTOP_NO_THINKING_PREFIX,
    decode_gateway_model_id,
)
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.trace import trace_event

from .policy_selection import candidate_destination_allowed
from .reasoning import (
    resolve_gemini_reasoning_policy,
    resolve_reasoning_policy,
    resolve_responses_reasoning_policy,
)

_ROUTE_SETTINGS = (
    ("fable", "model_fable", "reasoning_fable"),
    ("opus", "model_opus", "reasoning_opus"),
    ("haiku", "model_haiku", "reasoning_haiku"),
    ("sonnet", "model_sonnet", "reasoning_sonnet"),
)


@dataclass(frozen=True, slots=True)
class ProviderModelTarget:
    """One canonical provider/model execution target."""

    provider_id: str
    provider_model: str
    provider_model_ref: str
    api_key_number: int = 1


@dataclass(frozen=True, slots=True)
class ProviderModelCandidate:
    """One execution target with a candidate-local reasoning preference."""

    target: ProviderModelTarget
    reasoning_preference: ReasoningPreference = ReasoningPreference.INHERIT


@dataclass(frozen=True, slots=True)
class ResolvedModelRoute:
    """One public model resolved to a primary and ordered fallbacks."""

    original_model: str
    primary: ProviderModelTarget
    fallbacks: tuple[ProviderModelCandidate, ...]
    reasoning_preference: ReasoningPreference
    policy_mode: str = "off"
    policy_reason: str = "legacy_mapping"
    proposed_model: str | None = None
    estimated_cost: float | None = None
    policy_deadline_seconds: float | None = None
    policy_first_event_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RoutedMessagesRequest:
    request: MessagesRequest
    resolved: ResolvedModelRoute
    reasoning: ReasoningPolicy


@dataclass(frozen=True, slots=True)
class RoutedNativeMessagesRequest:
    request: NativeMessagesRequest
    resolved: ResolvedModelRoute


def supports_native_messages(provider_id: str) -> bool:
    descriptor = PROVIDER_CATALOG.get(provider_id)
    return descriptor is not None and descriptor.native_messages_passthrough


@dataclass(frozen=True, slots=True)
class RoutedResponsesRequest:
    request: OpenAIResponsesRequest
    resolved: ResolvedModelRoute
    reasoning: ReasoningPolicy


@dataclass(frozen=True, slots=True)
class RoutedGeminiRequest:
    request: GeminiRequest
    resolved: ResolvedModelRoute
    reasoning: ReasoningPolicy


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest | NativeTokenCountRequest
    resolved: ResolvedModelRoute


class ModelRouter:
    """Resolve incoming Claude model names to configured provider/model pairs."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._policy_candidates = parse_candidates(settings.routing_candidates)

    def resolve(self, claude_model_name: str) -> ResolvedModelRoute:
        (
            direct_provider_id,
            direct_provider_model,
            force_reasoning_off,
        ) = self._direct_provider_model(claude_model_name)
        if direct_provider_id is not None and direct_provider_model is not None:
            reasoning_preference = (
                ReasoningPreference.OFF
                if force_reasoning_off
                else self._settings.reasoning_policy
            )
            logger.debug(
                "MODEL DIRECT: '{}' -> provider='{}' model='{}' reasoning={}",
                claude_model_name,
                direct_provider_id,
                direct_provider_model,
                reasoning_preference.value,
            )
            primary = self._target(direct_provider_id, direct_provider_model)
            return ResolvedModelRoute(
                original_model=claude_model_name,
                primary=primary,
                fallbacks=(
                    self._fallback_targets(primary)
                    if self._settings.routing_mode == "off"
                    else ()
                ),
                reasoning_preference=reasoning_preference,
                policy_mode=self._settings.routing_mode,
                policy_reason="explicit_model"
                if self._settings.routing_mode != "off"
                else "legacy_mapping",
            )

        provider_model_ref = self._resolve_model_ref(claude_model_name)
        reasoning_preference = self._resolve_reasoning_preference(claude_model_name)
        primary = self._target_from_ref(provider_model_ref)
        if primary.provider_model != claude_model_name:
            logger.debug(
                "MODEL MAPPING: '{}' -> '{}'",
                claude_model_name,
                primary.provider_model,
            )
        return ResolvedModelRoute(
            original_model=claude_model_name,
            primary=primary,
            fallbacks=self._fallback_targets(primary),
            reasoning_preference=reasoning_preference,
            policy_mode=self._settings.routing_mode,
        )

    def _target_from_ref(
        self, provider_model_ref: str, api_key_number: int = 1
    ) -> ProviderModelTarget:
        return self._target(
            parse_provider_type(provider_model_ref),
            parse_model_name(provider_model_ref),
            api_key_number,
        )

    def _target(
        self, provider_id: str, provider_model: str, api_key_number: int = 1
    ) -> ProviderModelTarget:
        self._validate_provider_id(provider_id)
        return ProviderModelTarget(
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=f"{provider_id}/{provider_model}",
            api_key_number=api_key_number,
        )

    def _fallback_targets(
        self, primary: ProviderModelTarget
    ) -> tuple[ProviderModelCandidate, ...]:
        configured = tuple(
            ProviderModelCandidate(
                self._target_from_ref(entry.model, entry.api_key_number),
                entry.reasoning,
            )
            for entry in self._settings.model_fallbacks or ()
        )
        return tuple(
            candidate for candidate in configured if candidate.target != primary
        )

    def _validate_provider_id(self, provider_id: str) -> None:
        if provider_id not in self._settings.provider_ids:
            raise UnknownProviderError.for_provider(
                provider_id, self._settings.provider_ids
            )

    def _direct_provider_model(
        self, model_name: str
    ) -> tuple[str | None, str | None, bool]:
        try:
            decoded = decode_gateway_model_id(model_name)
        except ValueError:
            raise InvalidRequestError("Invalid Claude Desktop model ID") from None
        candidate = (
            f"{decoded.provider_id}/{decoded.provider_model}"
            if decoded is not None
            else model_name
        )
        if is_retired_model_ref(candidate):
            return (
                parse_provider_type(self._settings.model),
                parse_model_name(self._settings.model),
                decoded.force_reasoning_off if decoded is not None else False,
            )
        replacement = replace_retired_model_ref(candidate)
        if replacement != candidate:
            return (
                parse_provider_type(replacement),
                parse_model_name(replacement),
                decoded.force_reasoning_off if decoded is not None else False,
            )
        if decoded is not None:
            if decoded.provider_id not in self._settings.provider_ids:
                if decoded.provider_id.startswith("custom_") or model_name.partition(
                    "/"
                )[0] in {
                    DESKTOP_MODEL_PREFIX,
                    DESKTOP_NO_THINKING_PREFIX,
                }:
                    raise UnknownProviderError.for_provider(
                        decoded.provider_id, self._settings.provider_ids
                    )
                return None, None, False
            return (
                decoded.provider_id,
                decoded.provider_model,
                decoded.force_reasoning_off,
            )

        provider_id, separator, provider_model = model_name.partition("/")
        if not separator:
            return None, None, False
        if provider_id not in self._settings.provider_ids:
            if provider_id.startswith("custom_"):
                raise UnknownProviderError.for_provider(
                    provider_id, self._settings.provider_ids
                )
            return None, None, False
        if not provider_model:
            return None, None, False
        return provider_id, provider_model, False

    def _resolve_model_ref(self, claude_model_name: str) -> str:
        """Resolve a Claude model name to the configured provider/model ref."""

        route = self._matched_route(claude_model_name)
        if route is not None:
            model = getattr(self._settings, route[1])
            if isinstance(model, str):
                return model
        return self._settings.model

    def _resolve_reasoning_preference(
        self, claude_model_name: str
    ) -> ReasoningPreference:
        """Resolve a route override without inspecting the provider model."""

        route = self._matched_route(claude_model_name)
        if route is not None:
            preference = getattr(self._settings, route[2])
            if preference is not ReasoningPreference.INHERIT:
                return preference
        return self._settings.reasoning_policy

    @staticmethod
    def _matched_route(model_name: str) -> tuple[str, str, str] | None:
        normalized = model_name.lower()
        return next(
            (route for route in _ROUTE_SETTINGS if route[0] in normalized),
            None,
        )

    def requires_policy_inspection(self, model: str) -> bool:
        """Automatic aliases need inspection before selecting native passthrough."""
        return (
            self._settings.routing_mode == "auto"
            and self._direct_provider_model(model)[0] is None
        )

    def _apply_policy(
        self,
        resolved: ResolvedModelRoute,
        *,
        payload: dict[str, object],
        input_tokens: int,
        output_tokens: int | None,
        reasoning: ReasoningPolicy,
    ) -> ResolvedModelRoute:
        if (
            self._settings.routing_mode == "off"
            or resolved.policy_reason == "explicit_model"
        ):
            return resolved
        if payload.get("extra_body"):
            if self._settings.routing_mode == "auto":
                raise InvalidRequestError(
                    "Automatic routing cannot validate extra_body overrides."
                )
            return replace(resolved, policy_reason="unsupported_extra_body")
        block_types: set[str] = set()
        pending: list[object] = [payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                kind = value.get("type")
                if isinstance(kind, str):
                    block_types.add(kind)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        needs_tools = bool(payload.get("tools")) or bool(
            block_types
            & {"tool_use", "tool_result", "function_call", "function_call_output"}
        )
        required_context = int(input_tokens * 1.1) + 256 + (output_tokens or 0)
        eligible = [
            candidate
            for candidate in self._policy_candidates
            if output_tokens is not None
            and output_tokens > 0
            and candidate.context_tokens >= required_context
            and candidate.supports_tools is not None
            and candidate.supports_thinking is not None
            and (not needs_tools or candidate.supports_tools)
            and (not reasoning.requests_reasoning or candidate.supports_thinking)
            and (
                not block_types.intersection({"image", "input_image"})
                or candidate.supports_images is True
            )
            and (
                not block_types.intersection({"document", "input_file"})
                or candidate.supports_documents is True
            )
            and not payload.get("mcp_servers")
            and candidate_destination_allowed(candidate, self._settings)
        ]
        if not eligible:
            if self._settings.routing_mode == "auto":
                raise InvalidRequestError(
                    "No eligible routing candidate for this request"
                )
            return replace(resolved, policy_reason="no_eligible_candidate")
        selected = eligible[0]
        estimated_cost = None
        if (
            selected.input_cost_per_million is not None
            and selected.output_cost_per_million is not None
        ):
            estimated_cost = (
                input_tokens * selected.input_cost_per_million
                + (output_tokens or 0) * selected.output_cost_per_million
            ) / 1_000_000
        decision = replace(
            resolved,
            policy_reason="ordered_candidates",
            proposed_model=selected.model,
            estimated_cost=estimated_cost,
        )
        if self._settings.routing_mode == "shadow":
            return decision
        fallbacks = (
            (ProviderModelCandidate(self._target_from_ref(eligible[1].model)),)
            if self._settings.routing_failover_enabled and len(eligible) > 1
            else ()
        )
        return replace(
            decision,
            primary=self._target_from_ref(selected.model),
            fallbacks=fallbacks,
            policy_deadline_seconds=self._settings.routing_deadline_seconds,
            policy_first_event_timeout_seconds=self._settings.routing_first_event_timeout_seconds,
        )

    def route_native_messages(
        self,
        request: NativeMessagesRequest,
        resolved: ResolvedModelRoute,
    ) -> RoutedNativeMessagesRequest:
        if (
            self._settings.routing_mode == "shadow"
            and resolved.policy_reason != "explicit_model"
        ):
            try:
                parsed = MessagesRequest.model_validate(request.body)
            except ValidationError:
                resolved = replace(resolved, policy_reason="unsupported_inspection")
            else:
                resolved = self.resolve_messages_request(parsed).resolved
        eligible = tuple(
            candidate
            for candidate in resolved.fallbacks
            if supports_native_messages(candidate.target.provider_id)
        )
        trace_event(
            stage="routing",
            event="free_claude_code.api.route.messages_contract",
            source="application",
            contract="native",
            excluded_fallbacks=tuple(
                candidate.target.provider_model_ref
                for candidate in resolved.fallbacks
                if candidate not in eligible
            ),
        )
        return RoutedNativeMessagesRequest(
            request, replace(resolved, fallbacks=eligible)
        )

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        routed.model = resolved.primary.provider_model
        reasoning = resolve_reasoning_policy(routed, resolved.reasoning_preference)
        if (
            self._settings.routing_mode != "off"
            and resolved.policy_reason != "explicit_model"
        ):
            resolved = self._apply_policy(
                resolved,
                payload=request.model_dump(mode="json"),
                input_tokens=get_token_count(
                    request.messages, request.system, request.tools
                ),
                output_tokens=request.max_tokens,
                reasoning=reasoning,
            )
            routed.model = resolved.primary.provider_model
        return RoutedMessagesRequest(
            request=routed,
            resolved=resolved,
            reasoning=reasoning,
        )

    def resolve_messages_request_with_policy(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> RoutedMessagesRequest:
        """Route an application-owned Messages request with resolved intent."""

        resolved = self.resolve(request.model)
        routed = request.model_copy(
            update={"model": resolved.primary.provider_model},
            deep=True,
        )
        return RoutedMessagesRequest(
            request=routed,
            resolved=resolved,
            reasoning=reasoning,
        )

    def resolve_token_count_request(
        self,
        request: TokenCountRequest | NativeTokenCountRequest,
        *,
        resolved: ResolvedModelRoute | None = None,
    ) -> RoutedTokenCountRequest:
        """Return an internal token-count request context."""
        resolved = resolved or self.resolve(request.model)
        routed = request.model_copy(
            update={"model": resolved.primary.provider_model}, deep=True
        )
        return RoutedTokenCountRequest(request=routed, resolved=resolved)

    def resolve_responses_request(
        self,
        request: OpenAIResponsesRequest,
    ) -> RoutedResponsesRequest:
        """Route a Responses request without converting its protocol shape."""

        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        routed.model = resolved.primary.provider_model
        reasoning = resolve_responses_reasoning_policy(
            routed, resolved.reasoning_preference
        )
        if (
            self._settings.routing_mode != "off"
            and resolved.policy_reason != "explicit_model"
        ):
            resolved = self._apply_policy(
                resolved,
                payload=request.model_dump(mode="json"),
                input_tokens=estimate_responses_input_tokens(request),
                output_tokens=request.max_output_tokens,
                reasoning=reasoning,
            )
            routed.model = resolved.primary.provider_model
        return RoutedResponsesRequest(
            request=routed,
            resolved=resolved,
            reasoning=reasoning,
        )

    def resolve_gemini_request(self, request: GeminiRequest) -> RoutedGeminiRequest:
        resolved = self.resolve(request.model)
        if self.requires_policy_inspection(request.model):
            raise InvalidRequestError(
                "Automatic policy routing currently supports Messages and Responses only."
            )
        return RoutedGeminiRequest(
            request=request.model_copy(
                update={"model": resolved.primary.provider_model}, deep=True
            ),
            resolved=resolved,
            reasoning=resolve_gemini_reasoning_policy(
                request, resolved.reasoning_preference
            ),
        )
