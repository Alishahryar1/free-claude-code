"""Recover from upstream ``max_(completion_)tokens`` too-large 400 rejections.

Some OpenAI-compatible providers (Groq, NVIDIA NIM, ...) cap the per-request
output token count below what Claude Code asks for and reject the whole request
with an HTTP 400 that names the allowed maximum, e.g.::

    max_completion_tokens must be less than or equal to 40960, ...

This module parses that maximum and clamps the request body so the provider can
retry once and succeed. The provider also remembers the learned cap per model
so later requests clamp proactively instead of paying the 400 every time.
"""

import re
from typing import Any

import openai

# Body keys that carry the output-token budget across OpenAI-compatible policies.
_OUTPUT_TOKEN_FIELDS = ("max_completion_tokens", "max_tokens")
_STRUCTURED_TEXT_FIELDS = ("message", "detail", "error")
_STRUCTURED_CONTAINER_FIELDS = ("error", "errors", "detail", "details")

_CAP_VALUE_PATTERN = r"[`'\"]?(\d+)[`'\"]?"
_OUTPUT_TOKEN_FIELD_PATTERN = (
    r"(?<!\w)[`'\"]?(?:max_completion_tokens|max_tokens)[`'\"]?(?!\w)"
)

# An accepted grammar must bind the output-token field and its numeric limit.
# A field mentioned elsewhere in the response must never authorize an unrelated
# comparator.
_FIELD_BOUND_CAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"range\s+of\s+{_OUTPUT_TOKEN_FIELD_PATTERN}\s+should\s+be\s+"
        rf"\[\s*1\s*,\s*{_CAP_VALUE_PATTERN}\s*\]"
    ),
    re.compile(
        rf"{_OUTPUT_TOKEN_FIELD_PATTERN}\s*(?::\s*)?"
        rf"(?:"
        rf"(?:(?:must|should)\s+be\s+)?"
        rf"(?:less|smaller)\s+than\s+or\s+equal\s+to"
        rf"|(?:(?:must|should)\s+be\s*)?<="
        rf"|(?:(?:must|should)\s+be\s+)?at\s+most"
        rf"|(?:must|should)\s+not\s+exceed"
        rf"|maximum(?:\s+allowed)?(?:\s+value)?\s+(?:is|of)"
        rf")\s*{_CAP_VALUE_PATTERN}"
    ),
    re.compile(
        rf"maximum(?:\s+allowed)?(?:\s+value)?\s+for\s+"
        rf"{_OUTPUT_TOKEN_FIELD_PATTERN}\s+is\s+{_CAP_VALUE_PATTERN}"
    ),
    re.compile(
        rf"maximum(?:\s+allowed)?(?:\s+value)?\s+of\s+"
        rf"{_CAP_VALUE_PATTERN}\s+for\s+{_OUTPUT_TOKEN_FIELD_PATTERN}"
    ),
)


def _is_bad_request(error: Exception) -> bool:
    return isinstance(error, openai.BadRequestError) or (
        getattr(error, "status_code", None) == 400
    )


def _normalized_output_field(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    field = value.strip("`'\" ").lower()
    return field if field in _OUTPUT_TOKEN_FIELDS else None


def _structured_error_texts(body: dict[str, Any] | list[Any]) -> tuple[str, ...]:
    """Extract messages without discarding their structured parameter scope."""
    texts: list[str] = []
    pending: list[dict[str, Any] | list[Any]] = [body]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, (dict, list)):
                    pending.append(item)
            continue

        param = value.get("param")
        output_field = _normalized_output_field(param)
        messages = tuple(
            message
            for field in _STRUCTURED_TEXT_FIELDS
            if isinstance(message := value.get(field), str)
        )
        if output_field is not None:
            # Structured parameter metadata is authoritative. Prefixing the
            # paired field supports comparator-only messages such as "<= 8192".
            texts.extend(f"{output_field} {message}" for message in messages)
        if param is not None:
            # Do not let text nested below an explicitly scoped validation error
            # escape that scope and become a new unscoped candidate.
            continue

        texts.extend(messages)
        for field in _STRUCTURED_CONTAINER_FIELDS:
            nested = value.get(field)
            if isinstance(nested, (dict, list)):
                pending.append(nested)
    return tuple(texts)


def _error_texts(error: Exception) -> tuple[str, ...]:
    """Keep unstructured text separate from structured validation messages."""
    body = getattr(error, "body", None)
    if body is None:
        texts = (str(error),)
    elif isinstance(body, str):
        texts = (body,)
    elif isinstance(body, (dict, list)):
        # The OpenAI SDK embeds a representation of this body in ``str(error)``.
        # Parsing both would flatten the parameter scopes restored above.
        texts = _structured_error_texts(body)
    else:
        texts = ()
    return tuple(text.lower() for text in texts)


def _parse_cap(text: str) -> int | None:
    for pattern in _FIELD_BOUND_CAP_PATTERNS:
        match = pattern.search(text)
        if match:
            cap = int(match.group(1))
            if cap > 0:
                return cap
    return None


def parse_output_token_cap(error: Exception) -> int | None:
    """Return the allowed output-token maximum named in a 400 rejection, if any."""
    if not _is_bad_request(error):
        return None

    for text in _error_texts(error):
        cap = _parse_cap(text)
        if cap is not None:
            return cap
    return None


def clamp_output_tokens(body: dict[str, Any], cap: int) -> dict[str, Any] | None:
    """Return a shallow clone with output-token fields clamped to ``cap``.

    Returns ``None`` when nothing needs clamping (no output field exceeds the
    cap), so callers can avoid a pointless identical retry.
    """
    clamped: dict[str, Any] | None = None
    for field in _OUTPUT_TOKEN_FIELDS:
        value = body.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > cap:
            if clamped is None:
                clamped = dict(body)
            clamped[field] = cap
    return clamped
