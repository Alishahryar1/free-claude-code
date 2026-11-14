"""Read status codes, error bodies, and marker text from raw upstream exceptions.

Every helper here is pure: it inspects one exception and never logs, sleeps, or
reads a clock.
"""

import json
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import httpx
import httpx2
import openai

from free_claude_code.core.anthropic.errors import anthropic_status_for_error_type
from free_claude_code.core.diagnostics import (
    attached_upstream_error_body,
    reported_http_status,
)
from free_claude_code.providers.canonical_failures import is_context_window_error_code

_RATE_LIMIT_MARKERS = frozenset({"rate_limit", "rate limit", "too many requests"})
_OVERLOAD_MARKERS = frozenset(
    {
        "resourceexhausted",
        "resource exhausted",
        "limit reached",
        "overloaded",
        "capacity",
    }
)
_INTERNAL_ERROR_MARKERS = frozenset({"internal_server_error", "internal server error"})


def is_backoff_status(status: int | None) -> bool:
    """Return whether a status asks the caller to back off and retry."""
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)


def reported_status(exc: Exception) -> int | None:
    """Return the status from the exception, its response, or its error body."""
    status = reported_http_status(exc)
    if status is not None:
        return status
    return _status_from_body(getattr(exc, "body", None))


def transient_status(exc: Exception) -> int | None:
    """Infer the backoff status an exception reports, if any."""
    if reports_context_window_exceeded(exc):
        return None
    if is_invalid_request(exc) or reported_status(exc) == 413:
        return None
    if isinstance(exc, openai.RateLimitError):
        return 429
    if isinstance(exc, httpx.HTTPStatusError | httpx2.HTTPStatusError):
        status = exc.response.status_code
        return status if is_backoff_status(status) else None

    status = getattr(exc, "status_code", None)
    if is_backoff_status(status):
        return status

    body_status = _status_from_body(getattr(exc, "body", None))
    if is_backoff_status(body_status):
        return body_status

    text = error_text(exc)
    if _has_marker(text, _RATE_LIMIT_MARKERS):
        return 429
    if _has_marker(text, _OVERLOAD_MARKERS):
        return 503
    if _has_marker(text, _INTERNAL_ERROR_MARKERS):
        return 500
    return None


def reports_overload(exc: Exception) -> bool:
    """Return whether an exception's text reports overload or capacity pressure."""
    return _has_marker(error_text(exc), _OVERLOAD_MARKERS)


def auth_status(exc: Exception) -> int | None:
    """Recognize HTTP and structured stream authentication failures, never prose."""
    status = reported_status(exc)
    if status is not None:
        return status if status in {401, 403} else None
    codes: list[object] = [getattr(exc, "code", None)]
    for item in _body_candidates(getattr(exc, "body", None)):
        if isinstance(item, Mapping):
            codes.extend((item.get("code"), item.get("type")))
    for code in codes:
        if isinstance(code, str) and (
            status := anthropic_status_for_error_type(code)
        ) in {401, 403}:
            return status
    if any(code in ("invalid_api_key", "unauthorized") for code in codes):
        return 401
    if any(code in ("permission_denied", "forbidden") for code in codes):
        return 403
    return None


def is_invalid_request(exc: Exception) -> bool:
    """Recognize invalid requests without masking status or credential failures."""
    status = reported_status(exc)
    if status is not None:
        return status == 400
    if auth_status(exc) is not None:
        return False
    return any(
        isinstance(value, str) and anthropic_status_for_error_type(value.lower()) == 400
        for item in _body_candidates(getattr(exc, "body", None))
        if isinstance(item, Mapping)
        for key in ("type", "code")
        if (value := item.get(key)) is not None
    )


def reports_context_window_exceeded(exc: Exception) -> bool:
    """Return whether a structured code anywhere on the failure means context exhaustion."""
    if is_context_window_error_code(getattr(exc, "code", None)):
        return True

    bodies = [attached_upstream_error_body(exc), getattr(exc, "body", None)]
    response = getattr(exc, "response", None)
    if response is not None:
        with suppress(Exception):
            bodies.append(response.content)
    return any(
        isinstance(item, Mapping)
        and any(is_context_window_error_code(item.get(key)) for key in ("code", "type"))
        for body in bodies
        for item in _body_candidates(body)
    )


def error_text(exc: Exception) -> str:
    """Combine exception, body, and response text for marker matching."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(_body_to_text(body))
    response = getattr(exc, "response", None)
    if response is not None:
        with suppress(Exception):
            parts.append(response.text)
    return " ".join(part for part in parts if part).lower()


def _status_from_body(body: Any) -> int | None:
    for item in _body_candidates(body):
        if not isinstance(item, Mapping):
            continue
        for key in ("status", "status_code", "code"):
            status = _coerce_status(item.get(key))
            if status is not None:
                return status
        type_status = _status_from_type_fields(item)
        if type_status is not None:
            return type_status
    return None


def _body_candidates(body: Any) -> tuple[Any, ...]:
    if isinstance(body, str):
        try:
            return _body_candidates(json.loads(body))
        except ValueError:
            return (body,)
    if isinstance(body, bytes):
        return _body_candidates(body.decode("utf-8", errors="replace"))
    if isinstance(body, Mapping):
        nested = body.get("error")
        return (body, nested) if isinstance(nested, Mapping) else (body,)
    return (body,)


def _coerce_status(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _status_from_type_fields(item: Mapping[str, Any]) -> int | None:
    text = " ".join(
        value.lower()
        for key in ("type", "code")
        if isinstance((value := item.get(key)), str)
    )
    if _has_marker(text, _RATE_LIMIT_MARKERS):
        return 429
    if _has_marker(text, _OVERLOAD_MARKERS):
        return 503
    if _has_marker(text, _INTERNAL_ERROR_MARKERS):
        return 500
    return None


def _body_to_text(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(body)


def _has_marker(text: str, markers: frozenset[str]) -> bool:
    return any(marker in text for marker in markers)
