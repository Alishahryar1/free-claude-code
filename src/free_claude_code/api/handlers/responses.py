"""OpenAI Responses API product flow for Codex clients."""

from collections.abc import Mapping

from fastapi.responses import JSONResponse

from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    ordinary_application_error_response,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import (
    openai_responses_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import ProviderResolver
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
from free_claude_code.core.openai_responses import (
    OPENAI_RESPONSES_SSE_HEADERS,
    OpenAIResponsesRequest,
    openai_error_payload,
    openai_error_type_for_failure,
    openai_failure_payload,
)


class ResponsesHandler:
    """Handle streaming OpenAI Responses-compatible requests."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        model_router: ModelRouter | None = None,
        provider_executor: ProviderExecutor | None = None,
        generation_id: int | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._provider_executor = provider_executor or ProviderExecutor(
            provider_resolver,
            progress_timeout_seconds=settings.provider_progress_timeout,
            generation_id=generation_id,
            log_raw_payloads=settings.log_raw_api_payloads,
            request_headers=request_headers,
        )

    async def create(
        self, request_data: OpenAIResponsesRequest, *, request_id: str | None = None
    ) -> object:
        """Create a streaming OpenAI Responses-compatible response."""
        request_id = request_id or new_request_id()
        request_payload = request_data.model_dump(mode="json", exclude_none=True)
        if request_data.stream is False:
            raise InvalidRequestError(
                "FCC /v1/responses supports streaming only; omit stream or set stream=true."
            )
        if not request_data.model.strip():
            raise InvalidRequestError("Responses request model must not be empty.")
        if (
            request_data.input is None
            or request_data.input == ""
            or request_data.input == []
        ):
            raise InvalidRequestError("Responses request input must not be empty.")

        try:
            routed = self._model_router.resolve_responses_request(request_data)
            streamed = self._provider_executor.stream_responses(
                routed,
                raw_log_payload=request_payload,
                request_id=request_id,
            )
            return await openai_responses_sse_streaming_response(
                streamed,
                headers=OPENAI_RESPONSES_SSE_HEADERS,
                pre_start_error_response=lambda exc: self._pre_start_error_response(
                    exc, request_id=request_id
                ),
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
            log_unexpected_api_exception(
                self._settings,
                exc,
                context="CREATE_RESPONSE_ERROR",
            )
            return JSONResponse(
                status_code=http_status_for_unexpected_api_exception(exc),
                content=openai_error_payload(
                    message=safe_exception_message(exc),
                    error_type="api_error",
                ),
            )

    def _pre_start_error_response(
        self, exc: BaseException, *, request_id: str
    ) -> JSONResponse:
        if isinstance(exc, ApplicationError):
            return ordinary_application_error_response(
                exc,
                wire_api="responses",
                request_id=request_id,
            )
        failure = find_execution_failure(exc)
        if failure is not None:
            return self._execution_failure_response(failure, request_id=request_id)
        log_unexpected_api_exception(
            self._settings,
            exc,
            context="CREATE_RESPONSE_STREAM_START_ERROR",
            request_id=request_id,
        )
        status_code = http_status_for_unexpected_api_exception(exc)
        trace_terminal_execution_error(
            wire_api="responses",
            request_id=request_id,
            status_code=status_code,
            error_type="api_error",
            error=exc,
        )
        return terminal_execution_error_response(
            status_code=status_code,
            content=openai_error_payload(
                message=safe_exception_message(exc),
                error_type="api_error",
            ),
        )

    def _execution_failure_response(
        self,
        failure: ExecutionFailure,
        *,
        request_id: str,
    ) -> JSONResponse:
        error_type = openai_error_type_for_failure(failure)
        trace_terminal_execution_error(
            wire_api="responses",
            request_id=request_id,
            status_code=failure.status_code,
            error_type=error_type,
            error=failure,
        )
        return terminal_execution_error_response(
            status_code=failure.status_code,
            content=openai_failure_payload(failure),
        )
