"""Structured upstream-error evidence shared by routing and provider layers.

Pure helpers: each inspects one exception, never logs, sleeps or reads a clock.
Prose is deliberately not evidence; only HTTP status and structured body fields
(``code``/``type``/``message``) are read.
"""

import json
import re
from collections.abc import Mapping
from contextlib import suppress

from .diagnostics import attached_upstream_error_body, reported_http_status
from .json_types import load_json_value

_RATE_LIMIT_CODES = frozenset({"rate_limit_exceeded", "rate_limit_error", "tokens"})
_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "quota_exceeded",
        "rate_limit_exceeded",
        "rate_limit_error",
        "resource_exhausted",
    }
)
_LIMIT_PATTERN = re.compile(r"limit\s+(\d+)\s*,\s*requested\s+(\d+)", re.IGNORECASE)


def _candidates(body: object) -> tuple[object, ...]:
    if isinstance(body, str):
        try:
            return _candidates(load_json_value(body))
        except ValueError:
            return (body,)
    if isinstance(body, bytes):
        return _candidates(body.decode("utf-8", errors="replace"))
    if isinstance(body, Mapping):
        nested = body.get("error")
        return (body, nested) if isinstance(nested, Mapping) else (body,)
    return (body,)


def _error_mappings(exc: BaseException) -> tuple[Mapping[str, object], ...]:
    sources: list[object] = [
        attached_upstream_error_body(exc),
        getattr(exc, "body", None),
    ]
    response: object = getattr(exc, "response", None)
    if response is not None:
        for attribute in ("content", "text"):
            with suppress(Exception):
                sources.append(getattr(response, attribute, None))
    return tuple(
        item
        for source in sources
        if source is not None
        for item in _candidates(source)
        if isinstance(item, Mapping)
    )


def error_text(exc: BaseException) -> str:
    """Combine exception, body and response text (lowercase) for display-only use."""
    parts = [str(exc)]
    body: object = getattr(exc, "body", None)
    if body is not None:
        try:
            parts.append(
                body.decode("utf-8", errors="replace")
                if isinstance(body, bytes)
                else body
                if isinstance(body, str)
                else json.dumps(body, ensure_ascii=False, separators=(",", ":"))
            )
        except TypeError:
            parts.append(str(body))
    response: object = getattr(exc, "response", None)
    if response is not None:
        with suppress(Exception):
            text: object = getattr(response, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return " ".join(part for part in parts if part).lower()


def reported_status(exc: BaseException) -> int | None:
    """HTTP status from the exception or its response (body status excluded)."""
    return reported_http_status(exc)


def structured_error_codes(exc: BaseException) -> frozenset[str]:
    """Lowercase structured ``code``/``type`` values from the error body."""
    codes: set[str] = set()
    direct = getattr(exc, "code", None)
    if isinstance(direct, str):
        codes.add(direct.lower())
    for item in _error_mappings(exc):
        for key in ("code", "type"):
            value = item.get(key)
            if isinstance(value, str):
                codes.add(value.lower())
    return frozenset(codes)


def structured_error_message(exc: BaseException) -> str:
    """Lowercase ``message`` field(s) of the structured error body."""
    return " ".join(
        value.lower()
        for item in _error_mappings(exc)
        if isinstance(value := item.get("message"), str)
    )


def reports_tpm_limit(exc: BaseException) -> bool:
    """Structured tokens-per-minute evidence on a 413 or 429 response.

    A plain payload-too-large 413 has no rate-limit code and is not matched.
    """
    status = reported_status(exc)
    if status not in {413, 429}:
        return False
    if not structured_error_codes(exc) & _RATE_LIMIT_CODES:
        return False
    if status == 413:
        return True
    message = structured_error_message(exc)
    return "tokens per minute" in message or "tpm" in message


def reports_quota_403(exc: BaseException) -> bool:
    """HTTP 403 whose structured body carries quota or rate-limit evidence."""
    return reported_status(exc) == 403 and bool(
        structured_error_codes(exc) & _QUOTA_CODES
    )


def tpm_limit_and_requested(exc: BaseException) -> tuple[int, int] | None:
    """Parse ``Limit N, Requested M`` from a structured provider message."""
    match = _LIMIT_PATTERN.search(structured_error_message(exc))
    return (int(match.group(1)), int(match.group(2))) if match else None
