"""OpenAI Responses SSE event formatting."""

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

import simplejson

from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import openai_error_from_failure, openai_error_payload

OPENAI_RESPONSES_SSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def format_response_sse_event(event_type: str, data: Mapping[str, Any]) -> str:
    """Format one OpenAI Responses SSE event."""

    return f"event: {event_type}\ndata: {simplejson.dumps(data, use_decimal=True)}\n\n"


def committed_response_failure_frame(
    first_chunk: str,
    latest_chunk: str,
    exc: BaseException,
) -> str:
    """Close an already-public Responses lifecycle after a boundary failure."""

    created = _response_event_payload(first_chunk)
    response_value = created.get("response") if created is not None else None
    if not isinstance(response_value, Mapping) or not isinstance(
        response_value.get("id"), str
    ):
        raise exc

    response = cast(JsonObject, deepcopy(dict(response_value)))
    failure = find_execution_failure(exc)
    if failure is not None:
        error = openai_error_from_failure(failure)
    else:
        error = openai_error_payload(
            message=safe_exception_message(exc),
            error_type="api_error",
        )["error"]
    response["status"] = "failed"
    response["error"] = cast(JsonValue, error)

    latest = _response_event_payload(latest_chunk)
    sequence_number = latest.get("sequence_number") if latest is not None else None
    next_sequence_number = (
        sequence_number + 1
        if isinstance(sequence_number, int) and not isinstance(sequence_number, bool)
        else 1
    )
    return format_response_sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "sequence_number": next_sequence_number,
            "response": response,
        },
    )


def _response_event_payload(chunk: str) -> JsonObject | None:
    for line in reversed(chunk.splitlines()):
        if not line.startswith("data:"):
            continue
        try:
            raw: object = json.loads(line.removeprefix("data:").lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(raw, Mapping):
            return cast(JsonObject, dict(raw))
    return None
