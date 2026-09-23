"""OpenAI-chat streamed usage request and extraction helpers."""

import json
from collections.abc import Mapping
from typing import Any

import openai

_USAGE_OPTION_KEYS = ("stream_options", "include_usage")
_USAGE_REJECTION_WORDS = (
    "unsupported",
    "not supported",
    "unknown",
    "unrecognized",
    "unexpected",
    "invalid",
    "extra",
    "forbidden",
    "not permitted",
)


def request_stream_usage(body: dict[str, Any]) -> None:
    """Ask an OpenAI-compatible streaming endpoint for its final usage chunk."""
    stream_options = body.get("stream_options")
    if stream_options is None:
        body["stream_options"] = {"include_usage": True}
        return
    if isinstance(stream_options, dict):
        stream_options["include_usage"] = True


def clone_without_stream_usage(body: dict[str, Any]) -> dict[str, Any] | None:
    """Return a clone with only ``include_usage`` removed from stream options."""
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, dict):
        return None
    if "include_usage" not in stream_options:
        return None

    retry_body = dict(body)
    retry_stream_options = dict(stream_options)
    retry_stream_options.pop("include_usage", None)
    if retry_stream_options:
        retry_body["stream_options"] = retry_stream_options
    else:
        retry_body.pop("stream_options", None)
    return retry_body


def is_stream_usage_rejection(error: Exception) -> bool:
    """Return whether upstream rejected the optional streamed-usage request."""
    if not _is_bad_request_like(error):
        return False
    text = _error_text(error)
    if not any(key in text for key in _USAGE_OPTION_KEYS):
        return False
    return any(word in text for word in _USAGE_REJECTION_WORDS)


def usage_int(usage_info: Any, key: str) -> int | None:
    """Extract an integer usage field from OpenAI SDK objects or plain dicts."""
    if usage_info is None:
        return None
    if isinstance(usage_info, Mapping):
        value = usage_info.get(key)
    else:
        value = getattr(usage_info, key, None)
        if value is None:
            extra = getattr(usage_info, "model_extra", None)
            if isinstance(extra, Mapping):
                value = extra.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def nested_usage_int(usage_info: Any, parent: str, key: str) -> int | None:
    """Extract one nested integer from SDK usage objects or plain mappings."""
    if usage_info is None:
        return None
    if isinstance(usage_info, Mapping):
        details = usage_info.get(parent)
    else:
        details = getattr(usage_info, parent, None)
        if details is None:
            extra = getattr(usage_info, "model_extra", None)
            if isinstance(extra, Mapping):
                details = extra.get(parent)
    if isinstance(details, Mapping):
        value = details.get(key)
    else:
        value = getattr(details, key, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_bad_request_like(error: Exception) -> bool:
    if isinstance(error, openai.BadRequestError):
        return True
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = (
            getattr(response, "status_code", None) if response is not None else None
        )
    return status in (400, 422)


def _error_text(error: Exception) -> str:
    parts = [str(error)]
    body = getattr(error, "body", None)
    if body is not None:
        parts.append(json.dumps(body, default=str))
    response = getattr(error, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return " ".join(parts).lower()
