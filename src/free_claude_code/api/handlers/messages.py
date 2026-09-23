"""Claude Messages API product flow."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace

from fastapi.responses import JSONResponse, Response
from loguru import logger

from free_claude_code.api.detection import detect_safety_classifier_stop_sequence
from free_claude_code.api.optimization_handlers import try_optimizations
from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    ordinary_application_error_response,
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
from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import ProviderExecutor, TokenCounter
from free_claude_code.application.ports import ModelInfoLookup, ProviderResolver
from free_claude_code.application.routing import ModelRouter, RoutedMessagesRequest
from free_claude_code.application.web_tools.ports import WebToolsPort
from free_claude_code.application.web_tools.service import WebToolService
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

from .classifier_response import classifier_response


@dataclass(frozen=True)
class _MessagesStreamResult:
    body: AsyncIterator[str]


@dataclass(frozen=True)
class _MessagesCompleteResult:
    response: object


_MessagesResult = _MessagesStreamResult | _MessagesCompleteResult


class MessagesHandler:
    """Handle Anthropic-compatible Messages requests."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        web_tools: WebToolsPort,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        provider_executor: ProviderExecutor | None = None,
        generation_id: int | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info_lookup: ModelInfoLookup | None = None,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._provider_executor = provider_executor or ProviderExecutor(
            provider_resolver,
            progress_timeout_seconds=settings.provider_progress_timeout,
            token_counter=token_counter,
            generation_id=generation_id,
            log_raw_payloads=settings.log_raw_api_payloads,
            request_headers=request_headers,
            model_info_lookup=model_info_lookup,
        )
        self._web_tools = WebToolService(
            settings=settings,
            client=web_tools,
            executor=self._provider_executor,
            token_counter=token_counter,
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
            tool_body = self._web_tools.try_stream_messages(
                routed, request_id=request_id
            )
            result = (
                _MessagesStreamResult(tool_body)
                if tool_body is not None
                else self._intercept_local_optimization(routed)
            )
            if result is None:
                logger.debug("No optimization matched, routing to provider")
                result = _MessagesStreamResult(
                    self._provider_executor.stream_messages(
                        routed,
                        raw_log_payload=routed.request.model_dump(),
                        request_id=request_id,
                    )
                )
            if routed.reasoning.control is ReasoningControl.PREFER_OFF and isinstance(
                result, _MessagesStreamResult
            ):
                result = _MessagesStreamResult(classifier_response(result.body))
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
                message, error, _complete = await aggregate_anthropic_sse_to_message(
                    result.body
                )
            except GeneratorExit:
                raise
            except asyncio.CancelledError:
                raise
            except ApplicationError:
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
        if isinstance(exc, ApplicationError):
            return ordinary_application_error_response(
                exc,
                wire_api="messages",
                request_id=request_id,
            )
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

    def _apply_message_routing_policies(
        self, routed: RoutedMessagesRequest
    ) -> RoutedMessagesRequest:
        classifier_stop_sequence = detect_safety_classifier_stop_sequence(
            routed.request
        )
        if classifier_stop_sequence is None:
            return routed

        reasoning_changed = routed.reasoning.control is not ReasoningControl.PREFER_OFF
        stop_sequences = routed.request.stop_sequences
        remaining_stop_sequences = (
            [
                stop_sequence
                for stop_sequence in stop_sequences
                if stop_sequence != classifier_stop_sequence
            ]
            if stop_sequences is not None
            else None
        )
        stop_sequence_removed = remaining_stop_sequences != stop_sequences
        trace_event(
            stage="routing",
            event="free_claude_code.api.route.safety_classifier_policy",
            source="api",
            model=routed.resolved.original_model,
            classifier_stop_sequence=classifier_stop_sequence,
            reasoning_changed=reasoning_changed,
            stop_sequence_removed=stop_sequence_removed,
        )
        if not reasoning_changed and not stop_sequence_removed:
            return routed

        request = routed.request
        if stop_sequence_removed:
            request = request.model_copy(
                update={"stop_sequences": remaining_stop_sequences or None}
            )
        return replace(
            routed,
            request=request,
            reasoning=(
                ReasoningPolicy.prefer_off() if reasoning_changed else routed.reasoning
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
