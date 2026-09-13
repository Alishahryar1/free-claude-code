"""OpenCode provider with catalog-driven Chat/Responses dispatch."""

import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesToolPolicy,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint import EndpointContext
from free_claude_code.providers.http import close_provider_stream
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    create_chat_client,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport

from .catalog import (
    OpenCodeCatalog,
    OpenCodeCatalogSnapshot,
    OpenCodeModelRoute,
    OpenCodeUpstreamTransport,
)


@dataclass(frozen=True, slots=True)
class OpenCodeProfile:
    """Immutable identity and catalog configuration for one OpenCode plan."""

    provider_id: str
    provider_name: str
    catalog_key: str
    chat_profile: OpenAIChatProfile


def _profile(
    provider_id: str,
    provider_name: str,
    catalog_key: str,
) -> OpenCodeProfile:
    return OpenCodeProfile(
        provider_id=provider_id,
        provider_name=provider_name,
        catalog_key=catalog_key,
        chat_profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy(
                provider_name=provider_name,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            ),
            NO_REASONING,
            user_agent="opencode",
        ),
    )


_PROFILES = {
    "opencode_zen": _profile("opencode_zen", "OPENCODE_ZEN", "opencode"),
    "opencode_go": _profile("opencode_go", "OPENCODE_GO", "opencode-go"),
}


class OpenCodeProvider(BaseProvider):
    """Route OpenCode models through their catalog-declared OpenAI endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenCodeProfile,
        admission: ProviderAdmissionController,
        catalog_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client = create_chat_client(
            config,
            base_url=profile.chat_profile.base_url(config.base_url).rstrip("/"),
            provider_name=profile.provider_name,
            default_headers={"User-Agent": "opencode"},
        )
        self._chat = OpenAIChatTransport(
            client=self._client,
            admission=admission,
            behavior=OpenAIChatBehavior(profile.chat_profile),
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            log_api_error_tracebacks=config.log_api_error_tracebacks,
        )
        self._opencode_profile = profile
        self._catalog = OpenCodeCatalog(
            config,
            provider_key=profile.catalog_key,
            provider_name=profile.provider_name,
            admission=admission,
            client=catalog_client,
        )
        self._responses = OpenAIResponsesTransport(
            client=self._client,
            admission=admission,
            provider_name=profile.provider_name,
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            tool_policy=ResponsesToolPolicy(
                custom_tools_as_functions=True,
                explicit_search_parameters=True,
                text_only_web_search=True,
                client_tool_search=True,
                flatten_namespaces=True,
            ),
        )

    async def cleanup(self) -> None:
        """Close both owned clients even when one cleanup fails."""
        errors: list[Exception] = []
        try:
            await self._client.close()
        except Exception as exc:
            errors.append(exc)
        try:
            await self._catalog.cleanup()
        except Exception as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("OpenCode provider cleanup failed", errors)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        snapshot = await self._catalog.refresh()
        return snapshot.model_infos

    def _upstream_headers(
        self, request_headers: Mapping[str, str]
    ) -> Mapping[str, str]:
        headers = {name.lower(): value for name, value in request_headers.items()}
        for name in (
            "x-opencode-session",
            "session-id",
            "x-session-id",
            "x-claude-code-session-id",
            "session_id",
            "x-grok-session-id",
            "x-meta-ai-gateway-session-id",
            "x-tbh-session-id",
            "x-fcc-launch-id",
        ):
            session_id = headers.get(name)
            if session_id and session_id.strip():
                return {"x-opencode-session": session_id}
        return {}

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        return self._dispatch_stream(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            request_headers=request_headers,
        )

    async def _dispatch_stream(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        snapshot = await self._catalog.snapshot(request_id=request_id)
        route = self._require_route(snapshot, request.model)
        routed = _routed_messages_request(request, route)
        selected_stream: AsyncIterator[str] | None = None
        try:
            if route.transport is OpenCodeUpstreamTransport.RESPONSES:
                selected_stream = self._responses.stream_messages(
                    routed,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    endpoint_context=endpoint_context,
                    extra_headers=self._upstream_headers(request_headers or {}),
                    model_info=route.model_info,
                )
            else:
                selected_stream = self._chat.stream_messages(
                    routed,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    endpoint_context=endpoint_context,
                    extra_headers=self._upstream_headers(request_headers or {}),
                    model_info=route.model_info,
                )
            async for event in selected_stream:
                yield event
        finally:
            if selected_stream is not None:
                await close_provider_stream(
                    selected_stream,
                    active_error=sys.exception(),
                    provider_name=self._opencode_profile.provider_name,
                    request_id=request_id,
                )

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
    ) -> AsyncIterator[str]:
        return self._dispatch_responses_stream(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            request_headers=request_headers,
        )

    async def _dispatch_responses_stream(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        snapshot = await self._catalog.snapshot(request_id=request_id)
        route = self._require_route(snapshot, request.model)
        routed = _routed_responses_request(request, route)
        selected_stream: AsyncIterator[str] | None = None
        try:
            if route.transport is OpenCodeUpstreamTransport.RESPONSES:
                selected_stream = self._responses.stream_responses(
                    routed,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    endpoint_context=endpoint_context,
                    extra_headers=self._upstream_headers(request_headers or {}),
                )
            else:
                selected_stream = self._chat.stream_responses(
                    routed,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    endpoint_context=endpoint_context,
                    extra_headers=self._upstream_headers(request_headers or {}),
                )
            async for event in selected_stream:
                yield event
        finally:
            if selected_stream is not None:
                await close_provider_stream(
                    selected_stream,
                    active_error=sys.exception(),
                    provider_name=self._opencode_profile.provider_name,
                    request_id=request_id,
                )

    def _require_route(
        self,
        snapshot: OpenCodeCatalogSnapshot,
        selector_id: str,
    ) -> OpenCodeModelRoute:
        route = snapshot.route(selector_id)
        if route is not None:
            return route
        raise InvalidRequestError(
            f"{self._opencode_profile.provider_name} does not advertise "
            f"model {selector_id!r} in its active catalog."
        )


def create_opencode_provider(
    provider_id: str,
    config: ProviderConfig,
    admission: ProviderAdmissionController,
    *,
    catalog_client: httpx.AsyncClient | None = None,
) -> OpenCodeProvider:
    """Construct one catalog-aware OpenCode provider."""
    profile = _PROFILES.get(provider_id)
    if profile is None:
        raise KeyError(f"No OpenCode profile for {provider_id!r}")
    return OpenCodeProvider(
        config,
        profile=profile,
        admission=admission,
        catalog_client=catalog_client,
    )


def _routed_messages_request(
    request: MessagesRequest,
    route: OpenCodeModelRoute,
) -> MessagesRequest:
    return request.model_copy(
        update={"model": route.upstream_model_id},
        deep=True,
    )


def _routed_responses_request(
    request: OpenAIResponsesRequest,
    route: OpenCodeModelRoute,
) -> OpenAIResponsesRequest:
    return request.model_copy(
        update={"model": route.upstream_model_id},
        deep=True,
    )
