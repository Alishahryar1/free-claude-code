"""Shared Messages framing and upstream failure evidence."""

import json
from collections.abc import AsyncIterator, Mapping
from typing import cast

import httpx

from free_claude_code.core.anthropic.errors import anthropic_status_for_error_type
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
    redact_sensitive_error_text,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
    is_context_window_finish_reason,
)


async def messages_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, JsonObject]]:
    decoder = AnthropicSSEDecoder()
    async for chunk in response.aiter_text():
        for event in decoder.feed(chunk):
            payload = cast(JsonObject, event.data)
            kind = event.event or payload.get("type")
            if not isinstance(kind, str) or not kind:
                raise RetryableProviderProtocolError(
                    "Messages stream has an invalid event type."
                )
            yield kind, payload
    for event in decoder.finish():
        payload = cast(JsonObject, event.data)
        kind = event.event or payload.get("type")
        if not isinstance(kind, str) or not kind:
            raise RetryableProviderProtocolError(
                "Messages stream has an invalid final event."
            )
        yield kind, payload


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
    kind = error.get("type") if isinstance(error, Mapping) else None
    if isinstance(error, Mapping) and any(
        is_context_window_error_code(error.get(key)) for key in ("type", "code")
    ):
        raise context_window_exceeded_provider_failure()
    status = anthropic_status_for_error_type(kind if isinstance(kind, str) else "")
    failure_kind = {
        400: FailureKind.INVALID_REQUEST,
        401: FailureKind.AUTHENTICATION,
        402: FailureKind.PERMISSION,
        403: FailureKind.PERMISSION,
        404: FailureKind.INVALID_REQUEST,
        413: FailureKind.INVALID_REQUEST,
        429: FailureKind.RATE_LIMIT,
        504: FailureKind.TIMEOUT,
        529: FailureKind.OVERLOADED,
    }.get(status, FailureKind.UPSTREAM)
    message = error.get("message") if isinstance(error, Mapping) else None
    failure = ExecutionFailure(
        failure_kind,
        status,
        redact_sensitive_error_text(message[:ERROR_DETAIL_DISPLAY_CAP_BYTES])
        if isinstance(message, str) and message
        else "Messages upstream returned an error.",
        status == 429 or status >= 500,
    )
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
