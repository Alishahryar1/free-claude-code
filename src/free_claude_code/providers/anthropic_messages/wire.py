"""Shared Messages framing and upstream failure evidence."""

import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import cast

import httpx

from free_claude_code.core.anthropic.errors import anthropic_failure_from_error
from free_claude_code.core.anthropic.stream_contracts import SSEEvent
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
    is_context_window_finish_reason,
)
from free_claude_code.providers.stream_read import read_stream_event


async def messages_events(
    response: httpx.Response, *, read_timeout_s: float
) -> AsyncGenerator[tuple[str, JsonObject]]:
    async with aclosing(_decoded_events(response)) as source:
        while True:
            try:
                event = await read_stream_event(
                    source,
                    timeout=read_timeout_s,
                    is_keepalive=lambda event: (
                        event.event == "ping" or event.data.get("type") == "ping"
                    ),
                )
            except StopAsyncIteration:
                return
            payload = cast(JsonObject, event.data)
            kind = event.event or payload.get("type")
            if not isinstance(kind, str) or not kind:
                raise RetryableProviderProtocolError(
                    "Messages stream has an invalid event type."
                )
            yield kind, payload


async def _decoded_events(response: httpx.Response) -> AsyncGenerator[SSEEvent]:
    decoder = AnthropicSSEDecoder()
    async for chunk in response.aiter_text():
        for event in decoder.feed(chunk):
            yield event
    for event in decoder.finish():
        yield event


def is_messages_stop(event_type: str, payload: JsonObject) -> bool:
    """Capture a model stop before presentation or validation can reject it."""
    delta = payload.get("delta")
    return event_type == "message_stop" or (
        event_type == "message_delta"
        and isinstance(delta, Mapping)
        and delta.get("stop_reason") is not None
    )


def check_messages_failure(
    event_type: str, payload: JsonObject, *, native: bool = False
) -> None:
    delta = payload.get("delta")
    if (
        not native
        and event_type == "message_delta"
        and isinstance(delta, Mapping)
        and is_context_window_finish_reason(delta.get("stop_reason"))
    ):
        raise context_window_exceeded_provider_failure()
    if event_type != "error":
        return
    error = payload.get("error")
    if isinstance(error, Mapping) and any(
        is_context_window_error_code(error.get(key)) for key in ("type", "code")
    ):
        raise context_window_exceeded_provider_failure()
    fields = dict(error) if isinstance(error, Mapping) else {}
    message = fields.get("message")
    if isinstance(message, str):
        fields["message"] = message[:ERROR_DETAIL_DISPLAY_CAP_BYTES]
    failure = anthropic_failure_from_error(fields)
    attach_upstream_error_body(failure, json.dumps(payload))
    raise failure


async def messages_status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    limit = ERROR_DETAIL_DISPLAY_CAP_BYTES
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk[: limit + 1 - len(body)])
        if len(body) > limit:
            break
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        attach_upstream_error_body(
            error, bytes(body[:limit]), truncated=len(body) > limit
        )
        return error
    raise AssertionError("Expected an unsuccessful Messages response.")
