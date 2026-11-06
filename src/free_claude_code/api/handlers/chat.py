"""Chat Completions JSON and SSE through the shared execution and delivery owners."""

from collections.abc import Mapping

from fastapi.responses import JSONResponse, Response

from free_claude_code.api.request_errors import (
    log_unexpected_api_exception,
    ordinary_application_error_response,
)
from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    chat_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.api.stream_frames import frame_data
from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import ModelInfoLookup, ProviderResolver
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.chat_completions.request import (
    ChatConversionError,
    ChatRequest,
)
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.openai_responses import (
    openai_error_payload,
    openai_error_type_for_failure,
)


class ChatHandler:
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
            generation_id=generation_id,
            request_headers=request_headers,
            model_info_lookup=model_info_lookup,
            log_raw_payloads=settings.log_raw_api_payloads,
        )

    async def create(self, request: ChatRequest, *, request_id: str) -> Response:
        try:
            routed = self._router.resolve_chat_request(request)
            body = self._executor.stream_chat(
                routed, request_id=request_id, raw_log_payload=request.body
            )
            response = await chat_sse_streaming_response(
                body,
                request_id=request_id,
                pre_start_error_response=lambda error: self._error(error, request_id),
            )
            if request.stream or not isinstance(response, ManagedStreamingResponse):
                return response
            aggregate = ChatResponseAccumulator()
            decoder = AnthropicSSEDecoder()

            def consume(frame: str) -> None:
                if frame.strip() == "data: [DONE]":
                    aggregate.finish()
                elif (payload := frame_data(frame)) is not None:
                    aggregate.feed(payload)

            try:
                async for chunk in response.body_iterator:
                    text = (
                        chunk
                        if isinstance(chunk, str)
                        else bytes(chunk).decode("utf-8")
                    )
                    for frame in decoder.feed_frames(text):
                        consume(frame)
                for frame in decoder.finish_frames():
                    consume(frame)
                return JSONResponse(content=aggregate.result())
            finally:
                await response.aclose()
        except (Exception, BaseExceptionGroup) as error:
            return self._error(error, request_id)

    def _error(self, error: BaseException, request_id: str) -> JSONResponse:
        if isinstance(error, ChatConversionError):
            error = InvalidRequestError(str(error))
        if isinstance(error, ApplicationError):
            return ordinary_application_error_response(
                error, wire_api="chat", request_id=request_id
            )
        failure = find_execution_failure(error)
        status = failure.status_code if failure is not None else 500
        if failure is None:
            log_unexpected_api_exception(
                self._settings,
                error,
                context="CREATE_CHAT_ERROR",
                request_id=request_id,
            )
        kind = openai_error_type_for_failure(failure) if failure else "api_error"
        trace_terminal_execution_error(
            wire_api="chat",
            request_id=request_id,
            status_code=status,
            error_type=kind,
            error=error,
        )
        return terminal_execution_error_response(
            status_code=status,
            content=openai_error_payload(
                message=safe_exception_message(failure or error), error_type=kind
            ),
        )
