"""Copilot subscription models dispatched through FCC's three HTTP egresses."""

import asyncio
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress

import httpx
import httpx2
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint import EndpointContext, HttpEndpoint
from free_claude_code.providers.http import close_provider_stream
from free_claude_code.providers.openai_chat import (
    OpenAIChatBehavior,
    OpenAIChatTransport,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport

from .auth import CopilotAuthManager
from .lifecycle import drain_owned
from .request_policy import (
    PROVIDER_NAME,
    REPLAY_SCOPE,
    chat_profile,
    non_messages_reasoning,
)
from .responses_events import CopilotResponsesEvents
from .types import CopilotEgress, CopilotModel


class _BorrowedMessagesPool(httpx.AsyncBaseTransport):
    def __init__(self, pool: httpx.AsyncBaseTransport) -> None:
        self._pool = pool

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._pool.handle_async_request(request)


class _MessagesEndpoint:
    """A fresh snapshot replaces all cookie state for this request's next attempt."""

    def __init__(self, context: EndpointContext, client: httpx.AsyncClient) -> None:
        self._context = context
        self._client = client

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        snapshot = await self._context.endpoint(force_refresh=force_refresh)
        self._client.cookies.clear()
        return snapshot


class GitHubCopilotProvider(BaseProvider):
    """Own generation HTTP resources; borrow the process account for each stream."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: CopilotAuthManager,
        admission: ProviderAdmissionController,
        messages_transport: httpx.AsyncBaseTransport | None = None,
        openai_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._admission = admission
        self._messages_pool = messages_transport or httpx.AsyncHTTPTransport(
            proxy=config.proxy
        )
        self._openai_pool = openai_transport or httpx2.AsyncHTTPTransport(
            proxy=config.proxy
        )
        self._client = AsyncOpenAI(
            api_key=_endpoint_required,
            base_url=config.base_url,
            max_retries=0,
            timeout=httpx2.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
        )
        self._responses = OpenAIResponsesTransport(
            client=self._client,
            admission=admission,
            provider_name=PROVIDER_NAME,
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            endpoint_transport=self._openai_pool,
            event_adapter_factory=CopilotResponsesEvents,
        )
        self._chats: dict[str, tuple[CopilotModel, OpenAIChatTransport]] = {}
        self._condition = asyncio.Condition()
        self._active = 0
        self._closing = False
        self._closed = False
        self._cleanup_task: asyncio.Task[None] | None = None

    def _check_model(self, model: str) -> None:
        if self._closing:
            raise ExecutionFailure(
                FailureKind.UNAVAILABLE, 503, "Copilot provider is closing.", False
            )
        if not model.strip() or model.strip().casefold() == "auto":
            raise InvalidRequestError(
                "Choose a concrete model from the connected Copilot account."
            )

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
    ) -> None:
        # Endpoint family and capabilities belong to the current account lease.
        # Conversion runs after acquisition and before opening the physical stream.
        self._check_model(request.model)

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        self._check_model(request.model)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._closing:
            raise ExecutionFailure(
                FailureKind.UNAVAILABLE, 503, "Copilot provider is closing.", False
            )
        return frozenset(
            model.info for model in (await self._auth.models(refresh=True)).values()
        )

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self.preflight_messages(request, reasoning=reasoning)
        return self._dispatch(
            request,
            input_tokens,
            request_id,
            response_model or request.model,
            reasoning,
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        self.preflight_responses(request, reasoning=reasoning)
        return self._dispatch(
            request,
            input_tokens,
            request_id,
            response_model or request.model,
            reasoning,
        )

    def _chat(self, model: CopilotModel) -> OpenAIChatTransport:
        cached = self._chats.get(model.info.model_id)
        if cached is None or cached[0] != model:
            transport = OpenAIChatTransport(
                behavior=OpenAIChatBehavior(chat_profile(model)),
                read_timeout_s=self._config.http_read_timeout,
                log_raw_sse_events=self._config.log_raw_sse_events,
                log_api_error_tracebacks=self._config.log_api_error_tracebacks,
                admission=self._admission,
                client=self._client,
                endpoint_transport=self._openai_pool,
            )
            self._chats[model.info.model_id] = (model, transport)
            return transport
        return cached[1]

    async def _dispatch(
        self,
        request: MessagesRequest | OpenAIResponsesRequest,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        async with self._condition:
            self._check_model(request.model)
            self._active += 1
        try:
            async with self._auth.lease(request.model) as lease:
                selected: AsyncIterator[str] | None = None
                http: httpx.AsyncClient | None = None
                try:
                    if lease.egress is CopilotEgress.MESSAGES:
                        http = httpx.AsyncClient(
                            transport=_BorrowedMessagesPool(self._messages_pool),
                            follow_redirects=False,
                            timeout=httpx.Timeout(
                                self._config.http_read_timeout,
                                connect=self._config.http_connect_timeout,
                                write=self._config.http_write_timeout,
                            ),
                        )
                        native = AnthropicMessagesTransport(
                            client=http,
                            admission=self._admission,
                            provider_name=PROVIDER_NAME,
                            replay_scope=REPLAY_SCOPE,
                            read_timeout_s=self._config.http_read_timeout,
                            capabilities=lease.model.messages,
                        )
                        endpoint = _MessagesEndpoint(lease, http)
                        if isinstance(request, MessagesRequest):
                            selected = native.stream_messages(
                                request,
                                endpoint_context=endpoint,
                                request_id=request_id,
                                response_model=response_model,
                                reasoning=reasoning,
                                model_info=lease.model.info,
                            )
                        else:
                            selected = native.stream_responses(
                                request,
                                endpoint_context=endpoint,
                                request_id=request_id,
                                response_model=response_model,
                                reasoning=reasoning,
                            )
                    else:
                        resolved = non_messages_reasoning(
                            request, reasoning, lease.model
                        )
                        transport = (
                            self._responses
                            if lease.egress is CopilotEgress.RESPONSES
                            else self._chat(lease.model)
                        )
                        if isinstance(request, MessagesRequest):
                            if lease.egress is CopilotEgress.RESPONSES:
                                selected = self._responses.stream_messages(
                                    request,
                                    input_tokens=input_tokens,
                                    request_id=request_id,
                                    response_model=response_model,
                                    reasoning=resolved,
                                    endpoint_context=lease,
                                    model_info=lease.model.info,
                                    can_disable_reasoning="none"
                                    in (lease.model.supported_efforts or ()),
                                )
                            else:
                                selected = self._chat(lease.model).stream_messages(
                                    request,
                                    input_tokens=input_tokens,
                                    request_id=request_id,
                                    response_model=response_model,
                                    reasoning=resolved,
                                    endpoint_context=lease,
                                    model_info=lease.model.info,
                                )
                        else:
                            selected = transport.stream_responses(
                                request,
                                input_tokens=input_tokens,
                                request_id=request_id,
                                response_model=response_model,
                                reasoning=resolved,
                                endpoint_context=lease,
                            )
                    while True:
                        try:
                            event = await _next_event(selected)
                        except StopAsyncIteration:
                            break
                        yield event
                finally:
                    await drain_owned(
                        asyncio.create_task(
                            _close_request(
                                selected,
                                http,
                                active_error=sys.exception(),
                                request_id=request_id,
                            )
                        )
                    )
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()

    async def cleanup(self) -> None:
        if self._closed:
            return
        self._closing = True
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup())
        await drain_owned(self._cleanup_task)

    async def _cleanup(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active == 0)
        results = await asyncio.gather(
            self._client.close(),
            self._messages_pool.aclose(),
            self._openai_pool.aclose(),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise ExceptionGroup("Copilot HTTP cleanup failed", failures)
        self._chats.clear()
        self._closed = True


async def _close_request(
    stream: AsyncIterator[str] | None,
    client: httpx.AsyncClient | None,
    *,
    active_error: BaseException | None,
    request_id: str | None,
) -> None:
    try:
        if stream is not None:
            await close_provider_stream(
                stream,
                active_error=active_error,
                provider_name=PROVIDER_NAME,
                request_id=request_id,
            )
    finally:
        if client is not None:
            await client.aclose()


async def _endpoint_required() -> str:
    raise ExecutionFailure(
        FailureKind.UNAVAILABLE,
        503,
        "Copilot inference requires a current account endpoint.",
        False,
    )


async def _advance(stream: AsyncIterator[str]) -> str:
    return await anext(stream)


async def _next_event(stream: AsyncIterator[str]) -> str:
    # Cancel transport work once, then protect its response-close finally blocks
    # from repeated caller cancellation before releasing the account lease.
    task = asyncio.create_task(_advance(stream))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        with suppress(Exception, asyncio.CancelledError):
            await drain_owned(task)
        raise
