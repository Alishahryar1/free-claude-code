"""OpenRouter provider implementation."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.open_router_preferences import (
    OpenRouterProviderPreferences,
)
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_chat import estimate_chat_input_tokens
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.model_listing import extract_tool_capable_model_infos
from free_claude_code.providers.openai_chat import (
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ReasoningObject,
    validate_extra_body_does_not_override_canonical_fields,
)
from free_claude_code.providers.reasoning_compatibility import (
    reasoning_control_rejected,
)

from .pricing import CheapestProviderIndex, endpoints_from_payload

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENROUTER",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


def _extra_body_mapping(body: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the body's caller-supplied extra fields, if any."""

    extra = body.get("extra_body")
    return extra if isinstance(extra, Mapping) else {}


def _required_endpoint_parameters(body: Mapping[str, Any]) -> frozenset[str]:
    """Return the endpoint parameters the request itself uses.

    The pin runs under ``allow_fallbacks: false``, so a provider that
    cannot serve one of these would fail the request outright. Only the
    live body knows what the request uses: the tool catalog and the
    encoded reasoning control both vary per request.
    """

    required: set[str] = set()
    if body.get("tools"):
        required.add("tools")
    reasoning = _extra_body_mapping(body).get("reasoning")
    if isinstance(reasoning, Mapping) and reasoning.get("enabled") is not False:
        # ``reasoning: {"enabled": false}` is the disable request, not a
        # capability requirement; the disable-retry path strips it anyway.
        required.add("reasoning")
    return frozenset(required)


class OpenRouterChatBehavior(OpenAIChatBehavior):
    """OpenRouter Chat adaptation without HTTP ownership."""

    def __init__(
        self,
        profile: OpenAIChatProfile,
        *,
        provider_preferences: OpenRouterProviderPreferences | None = None,
        cheapest_index: CheapestProviderIndex | None = None,
    ) -> None:
        super().__init__(profile)
        self._provider_preferences = provider_preferences
        self._cheapest_index = cheapest_index

    def finalize_chat_body(
        self, body: dict[str, Any], *, reasoning: ReasoningPolicy
    ) -> dict[str, Any]:
        """Attach configured static routing preferences to every request.

        The block travels inside ``extra_body``: the SDK merges it into the
        wire JSON verbatim, while a top-level key would be passed to
        ``chat.completions.create`` as a kwarg it does not accept.
        """

        body = super().finalize_chat_body(body, reasoning=reasoning)
        if self._provider_preferences is not None:
            block = self._provider_preferences.to_request_field()
            body["extra_body"] = {
                **_extra_body_mapping(body),
                "provider": block,
            }
        return body

    def prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Pin the request to the cheapest provider when pricing resolves.

        The capacity constraint counts the body being sent, not the
        executor's original count: continuation and recovery paths grow the
        message list after that count is taken, and only the body reflects
        the request the provider actually receives.
        """

        if self._cheapest_index is None:
            return body
        static_block = _extra_body_mapping(body).get("provider")
        constraints = static_block if isinstance(static_block, Mapping) else None
        cheapest = self._cheapest_index.cheapest_provider_block(
            body.get("model") or "",
            constraints=constraints,
            prompt_tokens=estimate_chat_input_tokens(body),
            required_parameters=_required_endpoint_parameters(body),
        )
        if cheapest is None:
            return body
        merged = {**(constraints or {}), **cheapest}
        prepared = dict(body)
        prepared["extra_body"] = {
            **_extra_body_mapping(body),
            "provider": merged,
        }
        return prepared

    def failure_override(self, error: Exception) -> ExecutionFailure | None:
        if getattr(error, "status_code", None) != 404:
            return None
        body = getattr(error, "body", None)
        if isinstance(body, Mapping) and "error" in body:
            if any(field in body for field in ("code", "message", "metadata")):
                return None
            body = body["error"]
        if not isinstance(body, Mapping):
            return None
        code = body.get("code")
        if not isinstance(code, int) or code != 404:
            return None
        if not isinstance(body.get("message"), str):
            return None
        metadata = body.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("failed_routing_step") != "Filter by Image Support"
        ):
            return None
        return ExecutionFailure(
            FailureKind.INVALID_REQUEST,
            400,
            "No OpenRouter endpoint for this request supports image input. "
            "Remove the image or choose an image-capable model.",
            False,
        )

    def reasoning_disable_rejected(self, error: Exception) -> bool:
        detail = extract_upstream_error_detail(error)
        if detail.status_code not in {None, 200}:
            return False
        try:
            payload = json.loads(detail.body_text or "")
        except ValueError:
            return False
        if not isinstance(payload, dict):
            return False
        error_body = payload.get("error", payload)
        if not isinstance(error_body, dict):
            return False
        code = error_body.get("code")
        return (
            isinstance(code, int)
            and code == 400
            and reasoning_control_rejected(error_body, ("reasoning",))
        )


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_preferences: OpenRouterProviderPreferences | None = None,
        quantizations: tuple[str, ...] | None = None,
        use_cheapest_provider: bool = False,
        admission: ProviderAdmissionController,
    ) -> None:
        # The index fetches through self._client, which the base constructor
        # creates; no fetch runs before a request streams.
        cheapest_index = (
            CheapestProviderIndex(
                fetch=self._fetch_endpoints, quantizations=quantizations
            )
            if use_cheapest_provider
            else None
        )
        super().__init__(
            config,
            behavior=OpenRouterChatBehavior(
                _PROFILE,
                provider_preferences=provider_preferences,
                cheapest_index=cheapest_index,
            ),
            admission=admission,
        )
        self._cheapest_index = cheapest_index

    async def _fetch_endpoints(
        self, base_model: str
    ) -> Sequence[Mapping[str, Any]] | None:
        """Fetch one model's endpoint list for cheapest-provider pricing."""

        author, _, slug = base_model.partition("/")
        payload = await self._client.get(
            f"/models/{author}/{slug}/endpoints", cast_to=object
        )
        return endpoints_from_payload(payload)

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        """Build eagerly; refresh pricing first when the cache is cold."""

        stream = super().stream_messages(
            request,
            input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            model_info=model_info,
            endpoint_context=endpoint_context,
            request_headers=request_headers,
            continuation=continuation,
        )
        if self._cheapest_index is None or not self._cheapest_index.needs_refresh(
            request.model
        ):
            return stream
        return self._refresh_then_stream(request.model, stream)

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        """Build eagerly; refresh pricing first when the cache is cold."""

        stream = super().stream_responses(
            request,
            input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            request_headers=request_headers,
            model_info=model_info,
            continuation=continuation,
        )
        if self._cheapest_index is None or not self._cheapest_index.needs_refresh(
            request.model
        ):
            return stream
        return self._refresh_then_stream(request.model, stream)

    async def _refresh_then_stream(
        self, model: str, stream: AsyncIterator[str]
    ) -> AsyncGenerator[str]:
        """Refresh pricing, then stream, closing the inner stream on exit."""

        assert self._cheapest_index is not None
        await self._cheapest_index.ensure(model)
        try:
            async for event in stream:
                yield event
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Advertise OpenRouter tool models with reasoning capability metadata."""
        payload = await self._list_models_payload()
        return extract_tool_capable_model_infos(
            payload, provider_name=self._provider_name
        )


_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    ReasoningObject(tuple((effort, effort.value) for effort in ReasoningEffort)),
    structured_reasoning_details=True,
)
