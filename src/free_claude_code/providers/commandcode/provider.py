"""Command Code provider: catalog-driven Messages / Chat Completions dispatch."""

import sys
from collections.abc import AsyncIterator, Mapping
from typing import override

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint_types import EndpointContext, HttpEndpoint
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.http import close_provider_stream
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    create_chat_client,
)

from .catalog import (
    CommandCodeCatalog,
    CommandCodeCatalogSnapshot,
    CommandCodeModelRoute,
    CommandCodeTransport,
)

PROVIDER_NAME = "COMMANDCODE"
_REPLAY_SCOPE = "commandcode"


async def _omit_cookies(request: httpx.Request) -> None:
    request.headers.pop("cookie", None)


class CommandCodeProvider(BaseProvider):
    """Send each advertised model to the Command Code endpoint that serves it."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        history_cache: HistoryModeCache | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config, history_cache=history_cache)
        self._admission = admission
        base_url = config.base_url.rstrip("/")
        self._http = http_client or httpx.AsyncClient(
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
            follow_redirects=False,
            event_hooks={"request": [_omit_cookies]},
        )
        self._endpoint = HttpEndpoint(
            base_url=base_url, api_key=config.api_key, headers={}
        )
        self._catalog = CommandCodeCatalog(
            config,
            provider_name=PROVIDER_NAME,
            admission=admission,
            client=self._http,
        )
        self._messages = AnthropicMessagesTransport(
            client=self._http,
            admission=admission,
            history_cache=self._history_cache,
            provider_name=PROVIDER_NAME,
            replay_scope=_REPLAY_SCOPE,
            read_timeout_s=config.http_read_timeout,
        )
        self._chat_client = create_chat_client(
            config, base_url=base_url, provider_name=PROVIDER_NAME
        )
        self._chat = OpenAIChatTransport(
            client=self._chat_client,
            admission=admission,
            history_cache=self._history_cache,
            behavior=OpenAIChatBehavior(
                OpenAIChatProfile(
                    OpenAIChatRequestPolicy(
                        provider_name=PROVIDER_NAME,
                        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
                    ),
                    NO_REASONING,
                )
            ),
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            log_api_error_tracebacks=config.log_api_error_tracebacks,
        )

    async def cleanup(self) -> None:
        """Close both owned clients even when one cleanup fails."""
        errors: list[Exception] = []
        try:
            await self._chat_client.close()
        except Exception as exc:
            errors.append(exc)
        try:
            await self._http.aclose()
        except Exception as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Command Code provider cleanup failed", errors)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        snapshot = await self._catalog.refresh()
        return snapshot.model_infos

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
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        return self._dispatch(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
        )

    @override
    def stream_chat(
        self,
        request: ChatRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        return self._dispatch(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
        )

    @override
    def stream_gemini(
        self,
        request: GeminiRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        return self._dispatch(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
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
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        return self._dispatch(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
        )

    async def _dispatch(
        self,
        request: MessagesRequest | ChatRequest | GeminiRequest | OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        continuation: ContinuationSeed | None,
    ) -> AsyncIterator[StreamOutput]:
        snapshot = await self._catalog.snapshot(request_id=request_id)
        route = self._require_route(snapshot, request.model)
        selected: AsyncIterator[StreamOutput] | None = None
        try:
            selected = self._select(
                request,
                route,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                continuation=continuation,
            )
            async for event in selected:
                yield event
        finally:
            if selected is not None:
                await close_provider_stream(
                    selected,
                    active_error=sys.exception(),
                    provider_name=PROVIDER_NAME,
                    request_id=request_id,
                )

    def _select(
        self,
        request: MessagesRequest | ChatRequest | GeminiRequest | OpenAIResponsesRequest,
        route: CommandCodeModelRoute,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        continuation: ContinuationSeed | None,
    ) -> AsyncIterator[StreamOutput]:
        info = route.model_info
        if route.transport is CommandCodeTransport.MESSAGES:
            native = self._messages
            endpoint = self._endpoint
            if isinstance(request, ChatRequest):
                return native.stream_chat(
                    request,
                    endpoint_context=endpoint,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    model_info=info,
                    continuation=continuation,
                )
            if isinstance(request, GeminiRequest):
                return native.stream_gemini(
                    request,
                    endpoint_context=endpoint,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    model_info=info,
                    continuation=continuation,
                )
            if isinstance(request, MessagesRequest):
                return native.stream_messages(
                    request,
                    endpoint_context=endpoint,
                    request_id=request_id,
                    response_model=response_model,
                    reasoning=reasoning,
                    model_info=info,
                    continuation=continuation,
                )
            return native.stream_responses(
                request,
                endpoint_context=endpoint,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=info,
                continuation=continuation,
            )
        chat = self._chat
        if isinstance(request, ChatRequest):
            return chat.stream_chat(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                continuation=continuation,
            )
        if isinstance(request, GeminiRequest):
            return chat.stream_gemini(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                continuation=continuation,
            )
        if isinstance(request, MessagesRequest):
            return chat.stream_messages(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=info,
                continuation=continuation,
            )
        return chat.stream_responses(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            continuation=continuation,
        )

    def _require_route(
        self, snapshot: CommandCodeCatalogSnapshot, model_id: str
    ) -> CommandCodeModelRoute:
        route = snapshot.route(model_id)
        if route is not None:
            return route
        raise InvalidRequestError(
            f"{PROVIDER_NAME} does not advertise model {model_id!r} in its catalog."
        )


def create_commandcode_provider(
    config: ProviderConfig,
    admission: ProviderAdmissionController,
    history_cache: HistoryModeCache | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> CommandCodeProvider:
    """Construct one catalog-aware Command Code provider."""
    return CommandCodeProvider(
        config,
        admission=admission,
        history_cache=history_cache,
        http_client=http_client,
    )
