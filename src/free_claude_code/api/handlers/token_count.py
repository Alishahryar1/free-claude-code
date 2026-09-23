"""Anthropic token-count API product flow."""

from fastapi import HTTPException
from loguru import logger

from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import TokenCounter
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    TokenCountRequest,
    TokenCountResponse,
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.trace import trace_event


class TokenCountHandler:
    """Handle Anthropic-compatible token count requests."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def count(
        self, request_data: TokenCountRequest, *, request_id: str | None = None
    ) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = request_id or new_request_id()
        with logger.contextualize(request_id=request_id):
            try:
                require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="free_claude_code.api.route.resolved",
                    source="api",
                    request_id=request_id,
                    kind="count_tokens",
                    provider_id=routed.resolved.primary.provider_id,
                    provider_model=routed.resolved.primary.provider_model,
                    provider_model_ref=routed.resolved.primary.provider_model_ref,
                    gateway_model=routed.resolved.original_model,
                )
                request_snapshot = anthropic_request_snapshot(routed.request)
                request_snapshot["model"] = routed.resolved.original_model
                trace_event(
                    stage="ingress",
                    event="free_claude_code.api.count_tokens.completed",
                    source="api",
                    request_id=request_id,
                    message_count=len(routed.request.messages),
                    input_tokens=tokens,
                    snapshot=request_snapshot,
                )
                return TokenCountResponse(input_tokens=tokens)
            except ApplicationError:
                raise
            except Exception as exc:
                log_unexpected_api_exception(
                    self._settings,
                    exc,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=http_status_for_unexpected_api_exception(exc),
                    detail=safe_exception_message(exc),
                ) from exc
