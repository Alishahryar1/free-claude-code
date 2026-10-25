"""Gemini JSON and SSE execution through the shared provider application."""

from collections.abc import Mapping

from fastapi.responses import JSONResponse, Response

from free_claude_code.api.request_errors import (
    log_unexpected_api_exception,
    ordinary_application_error_response,
)
from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    gemini_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import ModelInfoLookup, ProviderResolver
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.gemini import (
    GeminiConversionError,
    GeminiRequest,
    gemini_error_payload,
)
from free_claude_code.core.gemini.response import GeminiResponseAccumulator


class GeminiHandler:
    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        generation_id: int | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info_lookup: ModelInfoLookup | None = None,
    ) -> None:
        self._settings = settings
        self._router = ModelRouter(settings)
        self._executor = ProviderExecutor(
            provider_resolver,
            progress_timeout_seconds=settings.provider_progress_timeout,
            generation_id=generation_id,
            request_headers=request_headers,
            model_info_lookup=model_info_lookup,
            log_raw_payloads=settings.log_raw_api_payloads,
        )

    async def create(
        self, request: GeminiRequest, *, streaming: bool, request_id: str
    ) -> Response:
        try:
            routed = self._router.resolve_gemini_request(request)
            body = self._executor.stream_gemini(
                routed,
                request_id=request_id,
                raw_log_payload=lambda: request.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            )
            response = await gemini_sse_streaming_response(
                body,
                request_id=request_id,
                pre_start_error_response=lambda error: self._error(error, request_id),
            )
            if streaming or not isinstance(response, ManagedStreamingResponse):
                return response
            aggregate = GeminiResponseAccumulator()
            decoder = AnthropicSSEDecoder()
            try:
                async for chunk in response.body_iterator:
                    text = (
                        chunk
                        if isinstance(chunk, str)
                        else bytes(chunk).decode("utf-8")
                    )
                    for event in decoder.feed(text):
                        aggregate.feed(event.data)
                for event in decoder.finish():
                    aggregate.feed(event.data)
                return JSONResponse(content=aggregate.result())
            finally:
                await response.aclose()
        except (Exception, BaseExceptionGroup) as error:
            return self._error(error, request_id)

    def _error(self, error: BaseException, request_id: str) -> JSONResponse:
        if isinstance(error, ApplicationError):
            return ordinary_application_error_response(
                error, wire_api="gemini", request_id=request_id
            )
        if isinstance(error, GeminiConversionError):
            return JSONResponse(
                status_code=400, content=gemini_error_payload(400, str(error))
            )
        failure = find_execution_failure(error)
        status = failure.status_code if failure is not None else 500
        if failure is None:
            log_unexpected_api_exception(
                self._settings,
                error,
                context="CREATE_GEMINI_ERROR",
                request_id=request_id,
            )
        trace_terminal_execution_error(
            wire_api="gemini",
            request_id=request_id,
            status_code=status,
            error_type=failure.kind.value if failure else "api_error",
            error=error,
        )
        return terminal_execution_error_response(
            status_code=status,
            content=gemini_error_payload(
                status, safe_exception_message(failure or error)
            ),
        )
