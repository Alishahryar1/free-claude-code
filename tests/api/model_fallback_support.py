"""Controlled provider boundary for model-fallback API and product tests."""

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from tests.api.support import create_test_app


def execution_failure(message: str, *, retryable: bool = True) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message=message,
        retryable=retryable,
    )


def text_stream(text: str, *, model: str) -> list[str]:
    return [
        format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_fallback",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def responses_created_event(*, model: str) -> str:
    return _responses_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _responses_payload(
                model=model,
                status="in_progress",
            ),
        },
    )


def responses_text_stream(text: str, *, model: str) -> list[str]:
    output: JsonObject = {
        "id": "msg_fallback",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    return [
        responses_created_event(model=model),
        _responses_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_fallback",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
        ),
        _responses_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": _responses_payload(
                    model=model,
                    status="completed",
                    output=[output],
                ),
            },
        ),
    ]


def responses_failure_event(
    failure: ExecutionFailure,
    *,
    model: str,
) -> str:
    return _responses_event(
        "response.failed",
        {
            "type": "response.failed",
            "sequence_number": 1,
            "response": _responses_payload(
                model=model,
                status="failed",
                error={
                    "message": failure.message,
                    "type": "api_error",
                    "param": None,
                    "code": None,
                },
            ),
        },
    )


def _responses_event(event_type: str, payload: JsonObject) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _responses_payload(
    *,
    model: str,
    status: str,
    output: list[JsonObject] | None = None,
    error: JsonObject | None = None,
) -> JsonObject:
    return {
        "id": "resp_fallback",
        "object": "response",
        "model": model,
        "status": status,
        "output": output or [],
        "error": error,
        "usage": None,
    }


class ControlledFallbackProvider:
    def __init__(
        self,
        *,
        failure: ExecutionFailure | None = None,
        chunks_before_failure: tuple[str, ...] = (),
        responses_chunks_before_failure: tuple[str, ...] = (),
        text: str | None = None,
        validation_error: InvalidRequestError | None = None,
    ) -> None:
        self._failure = failure
        self._chunks_before_failure = chunks_before_failure
        self._responses_chunks_before_failure = responses_chunks_before_failure
        self._text = text
        self._validation_error = validation_error
        self.stream_models: list[str] = []
        self.response_models: list[str] = []
        self.close_calls = 0

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, reasoning
        if self._validation_error is not None:
            raise self._validation_error
        self.stream_models.append(request.model)
        public_model = response_model or request.model
        self.response_models.append(public_model)
        try:
            for chunk in self._chunks_before_failure:
                yield chunk
            if self._failure is not None:
                raise self._failure
            assert self._text is not None
            for chunk in text_stream(self._text, model=public_model):
                yield chunk
        finally:
            self.close_calls += 1

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, reasoning
        if self._validation_error is not None:
            raise self._validation_error
        self.stream_models.append(request.model)
        public_model = response_model or request.model
        self.response_models.append(public_model)
        try:
            for chunk in self._responses_chunks_before_failure:
                yield chunk
            if self._failure is not None:
                if self._responses_chunks_before_failure:
                    yield responses_failure_event(self._failure, model=public_model)
                    return
                raise self._failure
            assert self._text is not None
            for chunk in responses_text_stream(self._text, model=public_model):
                yield chunk
        finally:
            self.close_calls += 1


@contextmanager
def fallback_client(
    primary: ControlledFallbackProvider,
    fallback: ControlledFallbackProvider,
) -> Iterator[TestClient]:
    app = create_test_app(
        Settings(
            model="nvidia_nim/primary-model",
            model_fallbacks=("groq/fallback-model",),
        )
    )

    def resolve(
        provider_id: str,
        *,
        lease: object,
    ) -> ControlledFallbackProvider:
        del lease
        return {"nvidia_nim": primary, "groq": fallback}[provider_id]

    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            side_effect=resolve,
        ),
        TestClient(app) as client,
    ):
        yield client


def messages_payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "nvidia_nim/primary-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": stream,
    }


def responses_payload() -> dict[str, object]:
    return {
        "model": "nvidia_nim/primary-model",
        "input": "Hello",
        "max_output_tokens": 32,
        "stream": True,
    }
