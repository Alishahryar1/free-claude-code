"""Claude Messages API product flow."""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace

from fastapi.responses import JSONResponse, Response
from loguru import logger

from free_claude_code.api.detection import is_safety_classifier_request
from free_claude_code.api.optimization_handlers import try_optimizations
from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
    unexpected_http_exception,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import (
    EmptyStreamError,
    anthropic_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.api.web_tools.egress import (
    WebFetchEgressPolicy,
    web_fetch_allowed_scheme_set,
)
from free_claude_code.api.web_tools.execution import stream_local_web_tool_response
from free_claude_code.api.web_tools.request import (
    is_local_web_tool_request,
    is_web_server_tool_request,
    unsupported_server_tool_error,
)
from free_claude_code.api.web_tools.streaming import stream_web_server_tool_response
from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor, TokenCounter
from free_claude_code.application.ports import ProviderResolver
from free_claude_code.application.routing import ModelRouter, RoutedMessagesRequest
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    MessagesRequest,
    aggregate_anthropic_sse_to_message,
    anthropic_error_payload,
    anthropic_error_type_for_failure,
    anthropic_failure_payload,
    anthropic_status_for_error_type,
    get_token_count,
)
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.core.trace import trace_event


@dataclass(frozen=True)
class _MessagesStreamResult:
    body: AsyncIterator[str]


@dataclass(frozen=True)
class _MessagesCompleteResult:
    response: object


_MessagesResult = _MessagesStreamResult | _MessagesCompleteResult
MessageIntercept = Callable[[RoutedMessagesRequest], _MessagesResult | None]


class MessagesHandler:
    """Handle Anthropic-compatible Messages requests."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        provider_executor: ProviderExecutor | None = None,
        generation_id: int | None = None,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._provider_executor = provider_executor or ProviderExecutor(
            provider_resolver,
            token_counter=token_counter,
            generation_id=generation_id,
            log_raw_payloads=settings.log_raw_api_payloads,
        )
        self._message_intercepts: tuple[MessageIntercept, ...] = (
            self._intercept_local_optimization,
        )

    async def create(
        self, request_data: MessagesRequest, *, request_id: str | None = None
    ) -> object:
        """Create an Anthropic-compatible message response."""
        request_id = request_id or new_request_id()
        try:
            require_non_empty_messages(request_data.messages)
            routed = self._model_router.resolve_messages_request(request_data)
            routed = self._apply_message_routing_policies(routed)
            self._reject_unsupported_server_tools(routed)

            result = self._intercept_web_server_tool(routed, request_id=request_id)
            if result is None:
                result = self._run_message_intercepts(routed)
            if result is None:
                logger.debug("No optimization matched, routing to provider")
                result = _MessagesStreamResult(
                    self._provider_executor.stream(
                        routed,
                        wire_api="messages",
                        raw_log_label="FULL_PAYLOAD",
                        raw_log_payload=routed.request.model_dump(),
                        request_id=request_id,
                    )
                )
            return await self._to_public_response(
                result,
                stream=request_data.stream,
                request_id=request_id,
            )
        except ApplicationError:
            raise
        except ExecutionFailure as exc:
            return self._execution_failure_response(exc, request_id=request_id)
        except Exception as exc:
            failure = find_execution_failure(exc)
            if failure is not None:
                return self._execution_failure_response(failure, request_id=request_id)
            raise unexpected_http_exception(
                self._settings, exc, context="CREATE_MESSAGE_ERROR"
            ) from exc

    async def _to_public_response(
        self,
        result: _MessagesResult,
        *,
        stream: bool,
        request_id: str,
    ) -> object:
        if isinstance(result, _MessagesCompleteResult):
            return result.response
        if not stream:
            # Non-streaming clients (e.g. Claude Code utility calls) need a
            # complete JSON Message; the internal pipeline is always SSE, so
            # serving that raw here breaks the client SDK's response parse.
            try:
                message, error = await aggregate_anthropic_sse_to_message(result.body)
            except GeneratorExit:
                raise
            except asyncio.CancelledError:
                raise
            except ExecutionFailure as exc:
                return self._execution_failure_response(exc, request_id=request_id)
            except BaseExceptionGroup as exc:
                failure = find_execution_failure(exc)
                if failure is not None:
                    return self._execution_failure_response(
                        failure, request_id=request_id
                    )
                return self._unexpected_execution_error_response(
                    exc,
                    request_id=request_id,
                    context="CREATE_MESSAGE_NON_STREAM_ERROR",
                )
            except Exception as exc:
                return self._unexpected_execution_error_response(
                    exc,
                    request_id=request_id,
                    context="CREATE_MESSAGE_NON_STREAM_ERROR",
                )
            if error is not None:
                error_type, message_text = _stream_error_fields(error)
                status_code = anthropic_status_for_error_type(error_type)
                trace_terminal_execution_error(
                    wire_api="messages",
                    request_id=request_id,
                    status_code=status_code,
                    error_type=error_type,
                )
                return terminal_execution_error_response(
                    status_code=status_code,
                    content=anthropic_error_payload(
                        error_type=error_type,
                        message=message_text,
                        request_id=request_id,
                    ),
                )
            return JSONResponse(content=message)
        return await anthropic_sse_streaming_response(
            result.body,
            pre_start_error_response=lambda exc: self._pre_start_error_response(
                exc, request_id=request_id
            ),
            request_id=request_id,
        )

    def _pre_start_error_response(
        self, exc: BaseException, *, request_id: str
    ) -> Response:
        failure = find_execution_failure(exc)
        if failure is not None:
            return self._execution_failure_response(failure, request_id=request_id)
        context = (
            "CREATE_MESSAGE_EMPTY_STREAM"
            if isinstance(exc, EmptyStreamError)
            else "CREATE_MESSAGE_STREAM_START_ERROR"
        )
        return self._unexpected_execution_error_response(
            exc,
            request_id=request_id,
            context=context,
        )

    def _execution_failure_response(
        self, failure: ExecutionFailure, *, request_id: str
    ) -> JSONResponse:
        error_type = anthropic_error_type_for_failure(failure)
        trace_terminal_execution_error(
            wire_api="messages",
            request_id=request_id,
            status_code=failure.status_code,
            error_type=error_type,
            error=failure,
        )
        return terminal_execution_error_response(
            status_code=failure.status_code,
            content=anthropic_failure_payload(failure, request_id=request_id),
        )

    def _unexpected_execution_error_response(
        self,
        exc: BaseException,
        *,
        request_id: str,
        context: str,
    ) -> JSONResponse:
        log_unexpected_api_exception(
            self._settings,
            exc,
            context=context,
            request_id=request_id,
        )
        status_code = http_status_for_unexpected_api_exception(exc)
        trace_terminal_execution_error(
            wire_api="messages",
            request_id=request_id,
            status_code=status_code,
            error_type="api_error",
            error=exc,
        )
        return terminal_execution_error_response(
            status_code=status_code,
            content=anthropic_error_payload(
                error_type="api_error",
                message=safe_exception_message(exc),
                request_id=request_id,
            ),
        )

    def _reject_unsupported_server_tools(self, routed: RoutedMessagesRequest) -> None:
        tool_err = unsupported_server_tool_error(
            routed.request,
            web_tools_enabled=self._settings.enable_web_server_tools,
        )
        if tool_err is not None:
            raise InvalidRequestError(tool_err)

    def _apply_message_routing_policies(
        self, routed: RoutedMessagesRequest
    ) -> RoutedMessagesRequest:
        if not is_safety_classifier_request(routed.request):
            return routed
        changed = routed.reasoning.control is not ReasoningControl.OFF
        trace_event(
            stage="routing",
            event="free_claude_code.api.optimization.safety_classifier_no_thinking",
            source="api",
            model=routed.resolved.original_model,
            changed=changed,
        )
        if not changed:
            return routed
        return replace(routed, reasoning=ReasoningPolicy.off())

    def _run_message_intercepts(
        self, routed: RoutedMessagesRequest
    ) -> _MessagesResult | None:
        for intercept in self._message_intercepts:
            result = intercept(routed)
            if result is not None:
                return result
        return None

    def _intercept_web_server_tool(
        self, routed: RoutedMessagesRequest, *, request_id: str | None = None
    ) -> _MessagesResult | None:
        if not self._settings.enable_web_server_tools:
            return None
        if not is_local_web_tool_request(routed.request):
            return None

        input_tokens = self._token_counter(
            routed.request.messages, routed.request.system, routed.request.tools
        )
        trace_event(
            stage="routing",
            event="free_claude_code.api.optimization.web_server_tool",
            source="api",
            model=routed.resolved.original_model,
        )
        egress = WebFetchEgressPolicy(
            allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
            allowed_schemes=web_fetch_allowed_scheme_set(
                self._settings.web_fetch_allowed_schemes
            ),
        )
        if not is_web_server_tool_request(routed.request):
            return _MessagesStreamResult(
                stream_local_web_tool_response(
                    routed,
                    self._provider_executor,
                    web_fetch_egress=egress,
                    request_id=request_id or new_request_id(),
                )
            )
        return _MessagesStreamResult(
            stream_web_server_tool_response(
                routed.request,
                input_tokens=input_tokens,
                web_fetch_egress=egress,
                response_model=routed.resolved.original_model,
                verbose_client_errors=self._settings.log_api_error_tracebacks,
            ),
        )

    def _intercept_local_optimization(
        self, routed: RoutedMessagesRequest
    ) -> _MessagesResult | None:
        optimized = try_optimizations(
            routed.request,
            self._settings,
            response_model=routed.resolved.original_model,
        )
        if optimized is None:
            return None
        trace_event(
            stage="routing",
            event="free_claude_code.api.optimization.short_circuit",
            source="api",
            model=routed.resolved.original_model,
        )
        return _MessagesCompleteResult(optimized)


def _stream_error_fields(error: dict[str, object]) -> tuple[str, str]:
    raw_type = error.get("type")
    error_type = (
        raw_type.strip()
        if isinstance(raw_type, str) and raw_type.strip()
        else "api_error"
    )
    raw_message = error.get("message")
    message = (
        raw_message.strip()
        if isinstance(raw_message, str) and raw_message.strip()
        else "Provider request failed unexpectedly."
    )
    return error_type, message
