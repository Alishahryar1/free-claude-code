"""Controlled provider boundary for model-fallback API and product tests."""

import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.recovery import AttemptFailure, RecoveryCheckpoint
from free_claude_code.core.recovery_request import continue_request
from free_claude_code.core.stream_events import (
    DecodedStreamEvent,
    RequestOutcome,
    StreamEvent,
)
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
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**output, "status": "in_progress", "content": []},
            },
        ),
        _responses_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "item_id": "msg_fallback",
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
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
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": output,
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

    @asynccontextmanager
    async def open_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[_ControlledCandidate]:
        candidate = _ControlledCandidate(
            self, request, response_model or request.model, "messages"
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()

    @asynccontextmanager
    async def open_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[_ControlledCandidate]:
        candidate = _ControlledCandidate(
            self, request, response_model or request.model, "responses"
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()


class _ControlledCandidate:
    def __init__(
        self,
        provider: ControlledFallbackProvider,
        request: MessagesRequest | OpenAIResponsesRequest,
        model: str,
        wire_api: str,
    ) -> None:
        self._provider = provider
        self._request = request
        self._model = model
        self._wire_api = wire_api
        self._attempted = False
        self._closed = False

    @property
    def can_attempt(self) -> bool:
        return not self._attempted

    async def prepare(self, checkpoint: RecoveryCheckpoint) -> None:
        if self._provider._validation_error is not None:
            raise self._provider._validation_error
        self.continued_request = continue_request(self._request, checkpoint)

    async def stream_attempt(
        self,
        checkpoint: RecoveryCheckpoint,
        *,
        wait_for_recovery: bool,
        can_correct: Callable[[], bool],
        on_rejected: Callable[[], None] | None = None,
    ) -> AsyncIterator[DecodedStreamEvent]:
        self._attempted = True
        provider = self._provider
        provider.stream_models.append(self._request.model)
        provider.response_models.append(self._model)
        origin = ReplayOrigin(
            "controlled",
            "messages" if self._wire_api == "messages" else "responses",
            "",
            "",
            self._request.model,
        )
        yield DecodedStreamEvent(origin, StreamEvent("request.dispatched", {}), ())
        prefix = (
            provider._chunks_before_failure
            if self._wire_api == "messages"
            else provider._responses_chunks_before_failure
        )
        frames = list(prefix)
        if provider._failure is None:
            assert provider._text is not None
            frames += (
                text_stream(provider._text, model=self._model)
                if self._wire_api == "messages"
                else responses_text_stream(provider._text, model=self._model)
            )
        for frame in frames:
            for parsed in parse_sse_text(frame):
                event = StreamEvent(parsed.event, parsed.data)
                yield DecodedStreamEvent(
                    origin,
                    event,
                    (event,),
                    progress=parsed.event
                    in {"content_block_delta", "response.output_text.delta"},
                    outcome=RequestOutcome.SUCCESS
                    if parsed.event
                    in {"message_stop", "response.completed", "response.incomplete"}
                    else None,
                )
        if provider._failure is not None:
            raise AttemptFailure(provider._failure)

    def finish(self, failure: ExecutionFailure | None) -> None:
        pass

    async def suspend(self) -> None:
        pass

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            self._provider.close_calls += 1


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
