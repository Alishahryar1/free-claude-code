"""Application services for the Claude-compatible API."""

from __future__ import annotations

import json
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)

from . import metrics, pricing, skillopt, trajectory
from .model_router import ModelRouter, RoutedMessagesRequest
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim", "opencode", "opencode_go"})


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


async def _metered_stream(
    agen: AsyncIterator[str],
    *,
    request_id: str,
    provider_id: str,
    model: str,
    input_tokens: int,
    skill: str = "chat",
    thinking_enabled: bool = False,
) -> AsyncIterator[str]:
    """Wrap a provider SSE stream to record latency + token usage metrics."""
    start = time.perf_counter()
    output_tokens = 0
    status = "ok"
    try:
        async for chunk in agen:
            if '"message_delta"' in chunk and '"output_tokens"' in chunk:
                for line in chunk.splitlines():
                    if line.startswith("data: "):
                        try:
                            obj = json.loads(line[6:])
                            if obj.get("type") == "message_delta":
                                tok = obj.get("usage", {}).get("output_tokens")
                                if isinstance(tok, int):
                                    output_tokens = tok
                        except Exception:
                            pass
            yield chunk
    except BaseException:
        status = "error"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record(
            request_id=request_id,
            provider_id=provider_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
        )
        if trajectory.is_enabled():
            trajectory.record(
                request_id=request_id,
                provider_id=provider_id,
                model=model,
                skill=skill,
                thinking_enabled=thinking_enabled,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=pricing.estimate_cost_usd(
                    provider_id, model, input_tokens, output_tokens
                ),
                status=status,
            )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


_RETRIABLE_PREFLIGHT_ERRORS: tuple[type[BaseException], ...] = (
    AuthenticationError,
    ServiceUnavailableError,
    RateLimitError,
)


def _preflight_with_fallbacks(
    *,
    original_request: MessagesRequest,
    original_model: str,
    primary_routed: RoutedMessagesRequest,
    fallback_refs: tuple[str, ...],
    router: ModelRouter,
    provider_getter: ProviderGetter,
) -> tuple[RoutedMessagesRequest, BaseProvider]:
    """Try the primary resolution first, then each SkillOpt fallback ref.

    Returns ``(routed, provider)`` for the first candidate that passes
    ``preflight_stream``. Re-raises the primary's error when every
    candidate fails, so callers see the same exception they would have
    seen without SkillOpt enabled.
    """
    candidates: list[RoutedMessagesRequest] = [primary_routed]
    for ref in fallback_refs:
        try:
            candidates.append(
                router.rebind_messages_request(
                    original_request, ref, original_model=original_model
                )
            )
        except ValueError:
            logger.warning("SkillOpt fallback skipped: unknown provider in '{}'", ref)

    first_error: BaseException | None = None
    for candidate in candidates:
        resolved = candidate.resolved
        provider = provider_getter(resolved.provider_id)
        try:
            provider.preflight_stream(
                candidate.request,
                thinking_enabled=resolved.thinking_enabled,
            )
            return candidate, provider
        except _RETRIABLE_PREFLIGHT_ERRORS as exc:
            logger.warning(
                "SkillOpt fallback: provider='{}' failed preflight ({})",
                resolved.provider_id,
                type(exc).__name__,
            )
            if first_error is None:
                first_error = exc
            continue

    assert first_error is not None  # loop had at least one candidate
    raise first_error


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            skill = trajectory.infer_skill(
                request_data.messages, request_data.tools
            )
            routed = self._model_router.resolve_messages_request(
                request_data, skill=skill
            )
            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.optimization.web_server_tool",
                    source="api",
                    model=routed.request.model,
                )
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                trace_event(
                    stage="routing",
                    event="api.optimization.short_circuit",
                    source="api",
                    model=routed.request.model,
                )
                return optimized
            logger.debug("No optimization matched, routing to provider")

            routed, provider = _preflight_with_fallbacks(
                original_request=request_data,
                original_model=request_data.model,
                primary_routed=routed,
                fallback_refs=skillopt.fallbacks(skill),
                router=self._model_router,
                provider_getter=self._provider_getter,
            )

            trace_event(
                stage="routing",
                event="api.route.resolved",
                source="api",
                provider_id=routed.resolved.provider_id,
                provider_model=routed.resolved.provider_model,
                provider_model_ref=routed.resolved.provider_model_ref,
                gateway_model=routed.request.model,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            with logger.contextualize(request_id=request_id):
                trace_event(
                    stage="ingress",
                    event="api.request.received",
                    source="api",
                    message_count=len(routed.request.messages),
                    snapshot=api_messages_request_snapshot(routed.request),
                )

                if self._settings.log_raw_api_payloads:
                    logger.debug(
                        "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                    )

                input_tokens = self._token_counter(
                    routed.request.messages,
                    routed.request.system,
                    routed.request.tools,
                )

                streamed = traced_async_stream(
                    provider.stream_response(
                        routed.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=routed.resolved.thinking_enabled,
                    ),
                    stage="egress",
                    source="api",
                    complete_event="api.response.stream_completed",
                    interrupted_event="api.response.stream_interrupted",
                    chunk_event=None,
                    extra={
                        "request_id": request_id,
                        "provider_id": routed.resolved.provider_id,
                        "gateway_model": routed.request.model,
                    },
                )
                metered = _metered_stream(
                    streamed,
                    request_id=request_id,
                    provider_id=routed.resolved.provider_id,
                    model=routed.request.model,
                    input_tokens=input_tokens,
                    skill=skill,
                    thinking_enabled=routed.resolved.thinking_enabled,
                )
                return anthropic_sse_streaming_response(metered)

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.route.resolved",
                    source="api",
                    kind="count_tokens",
                    provider_id=routed.resolved.provider_id,
                    provider_model=routed.resolved.provider_model,
                    provider_model_ref=routed.resolved.provider_model_ref,
                    gateway_model=routed.request.model,
                )
                trace_event(
                    stage="ingress",
                    event="api.count_tokens.completed",
                    source="api",
                    message_count=len(routed.request.messages),
                    input_tokens=tokens,
                    snapshot=api_messages_request_snapshot(routed.request),
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
