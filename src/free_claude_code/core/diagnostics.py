"""Credential-safe diagnostics shared across product boundaries."""

import json
import re
import traceback
from dataclasses import dataclass
from typing import Any

from .failures import ExecutionFailure

ERROR_DETAIL_DISPLAY_CAP_BYTES = 16_384
_MAX_CAUSE_CHAIN_DEPTH = 4
_UPSTREAM_BODY_ATTR = "_fcc_upstream_error_body"
_UPSTREAM_BODY_TRUNCATED_ATTR = "_fcc_upstream_error_body_truncated"

_SECRET_TEXT_REPLACEMENTS = (
    (
        re.compile(
            r"(?i)(?P<prefix>[\"']?authorization[\"']?\s*[:=]\s*)"
            r"(?P<quote>[\"']?)(?:(?:bearer|basic)\s+)?"
            r"[^\"'\s,;&}\]]+(?P=quote)"
        ),
        r"\g<prefix>\g<quote><redacted>\g<quote>",
    ),
    (
        re.compile(
            r"(?i)(?P<prefix>[\"']?(?:api[_-]?key|access[_-]?token|"
            r"refresh[_-]?token|token|client[_-]?secret|secret|password)"
            r"[\"']?\s*[:=]\s*)(?P<quote>[\"']?)"
            r"[^\"'\s,;&}\]]+(?P=quote)"
        ),
        r"\g<prefix>\g<quote><redacted>\g<quote>",
    ),
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1<redacted>"),
    (
        re.compile(
            r"(?i)(?<![a-z0-9])(?:sk-[a-z0-9._-]{8,}|"
            r"nvapi-[a-z0-9._-]{8,}|hf_[a-z0-9_-]{8,}|"
            r"gsk_[a-z0-9_-]{8,}|github_pat_[a-z0-9_]{8,}|"
            r"gh[pousr]_[a-z0-9]{8,}|AIza[a-z0-9_-]{20,})"
            r"(?![a-z0-9])"
        ),
        "<redacted>",
    ),
)


@dataclass(frozen=True, slots=True)
class UpstreamErrorDetail:
    """Sanitized diagnostic detail extracted from an upstream exception."""

    status_code: int | None = None
    body_text: str | None = None
    exception_text: str | None = None
    cause_chain_text: str | None = None
    category_hint: str | None = None
    body_truncated: bool = False

    def is_invalid_request(self, *, code: object = None) -> bool:
        """Recognize HTTP validation failures and explicit streamed equivalents."""
        if self.status_code is not None:
            return self.status_code in {400, 422}
        codes = {
            "invalid_request_error",
            "invalid_request",
            "unsupported_parameter",
            "unsupported_value",
        }
        return self.category_hint in codes or (isinstance(code, str) and code in codes)


def redact_sensitive_error_text(text: str) -> str:
    """Redact recognizable credentials while preserving diagnostic context."""
    sanitized = text
    for pattern, replacement in _SECRET_TEXT_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def safe_exception_message(
    exc: BaseException,
    *,
    fallback: str = "Provider request failed unexpectedly.",
) -> str:
    """Return a redacted, non-empty exception message."""
    message = redact_sensitive_error_text(str(exc).strip())
    return message or fallback


def format_user_error_preview(exc: BaseException, *, max_len: int = 200) -> str:
    """Return a short redacted exception preview for user-facing surfaces."""
    return safe_exception_message(exc)[:max_len]


def attach_upstream_error_body(
    exc: Exception,
    body: bytes | str,
    *,
    truncated: bool = False,
) -> None:
    """Attach a bounded streamed response body for later safe formatting."""
    setattr(exc, _UPSTREAM_BODY_ATTR, body)
    setattr(exc, _UPSTREAM_BODY_TRUNCATED_ATTR, truncated)


def attached_upstream_error_body(exc: BaseException) -> bytes | str | None:
    """Return a bounded streamed body previously attached by the transport."""
    body = getattr(exc, _UPSTREAM_BODY_ATTR, None)
    return body if isinstance(body, bytes | str) else None


def exception_cause_types(exc: BaseException) -> tuple[str, ...]:
    """Return exception cause type names without logging their contents."""
    return tuple(type(cause).__name__ for cause in _exception_causes(exc))


def redacted_exception_traceback(exc: BaseException) -> str:
    """Format a traceback while redacting recognizable credentials."""
    return redact_sensitive_error_text("".join(traceback.format_exception(exc)))


def extract_upstream_error_detail(exc: Exception) -> UpstreamErrorDetail:
    """Extract bounded, redacted body, exception, and cause-chain details."""
    raw_body = attached_upstream_error_body(exc)
    body_truncated = bool(getattr(exc, _UPSTREAM_BODY_TRUNCATED_ATTR, False))
    if raw_body is None:
        raw_body = getattr(exc, "body", None)
    if raw_body is None:
        raw_body = _body_from_response(exc)

    body_text = _normalize_body_text(raw_body)
    if body_text is not None:
        body_text = redact_sensitive_error_text(body_text)
        body_text, capped = _cap_text_bytes(body_text)
        body_truncated = body_truncated or capped

    exception_text = str(exc).strip() or None
    if exception_text is not None:
        exception_text = redact_sensitive_error_text(exception_text)
        exception_text, _ = _cap_text_bytes(exception_text)

    return UpstreamErrorDetail(
        status_code=_status_code_from_exception(exc),
        body_text=body_text,
        exception_text=exception_text,
        cause_chain_text=_exception_cause_chain_text(exc),
        category_hint=_category_hint_from_body(raw_body, body_text),
        body_truncated=body_truncated,
    )


def format_execution_failure_message(
    failure: ExecutionFailure,
    detail: UpstreamErrorDetail,
    *,
    upstream_name: str,
    request_id: str | None = None,
) -> str:
    """Build a copyable, redacted diagnostic for a finalized execution failure."""
    stable_message = failure.message
    has_upstream_detail = detail.status_code is not None or detail.body_text is not None
    if not has_upstream_detail:
        lines = [stable_message]
        if detail.exception_text and detail.exception_text != stable_message:
            lines.extend(("", "Provider exception:", detail.exception_text))
        if detail.cause_chain_text:
            lines.extend(("", "Caused by:", detail.cause_chain_text))
        _append_request_id_lines(lines, request_id)
        return "\n".join(lines)

    if detail.status_code == 405:
        lines = [
            f"Upstream provider {upstream_name} rejected the request method "
            "or endpoint (HTTP 405)."
        ]
    elif detail.status_code is not None:
        lines = [
            f"Upstream provider {upstream_name} returned HTTP {detail.status_code}."
        ]
    else:
        lines = [f"Upstream provider {upstream_name} returned an error."]

    lines.append(f"Category: {detail.category_hint or failure.kind.value}")
    if stable_message and stable_message != lines[0]:
        lines.append(f"Mapped message: {stable_message}")
    lines.extend(("", "Upstream error:"))
    lines.append(detail.body_text or "(empty upstream error body)")
    if _body_truncation_line_needed(detail):
        lines.append(f"... [truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes]")
    _append_request_id_lines(lines, request_id)
    return "\n".join(lines)


def _body_truncation_line_needed(detail: UpstreamErrorDetail) -> bool:
    """Return whether a separate truncation marker is needed."""
    return detail.body_truncated and (
        detail.body_text is None
        or f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes"
        not in detail.body_text
    )


def _status_code_from_exception(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _body_from_response(exc: Exception) -> Any:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        pass
    try:
        return response.text
    except Exception:
        return None


def _normalize_body_text(body: Any) -> str | None:
    if body is None:
        return None
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    elif isinstance(body, str):
        text = body
    else:
        try:
            return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            text = str(body)
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return stripped
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _cap_text_bytes(text: str) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= ERROR_DETAIL_DISPLAY_CAP_BYTES:
        return text, False
    capped = encoded[:ERROR_DETAIL_DISPLAY_CAP_BYTES].decode("utf-8", errors="replace")
    return (
        f"{capped}\n... [truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes]",
        True,
    )


def _exception_causes(exc: BaseException) -> tuple[BaseException, ...]:
    causes: list[BaseException] = []
    seen = {id(exc)}
    current: BaseException | None = exc
    while current is not None and len(causes) < _MAX_CAUSE_CHAIN_DEPTH:
        next_exc = current.__cause__ or current.__context__
        if next_exc is None or id(next_exc) in seen:
            break
        seen.add(id(next_exc))
        causes.append(next_exc)
        current = next_exc
    return tuple(causes)


def _exception_cause_chain_text(exc: BaseException) -> str | None:
    lines: list[str] = []
    for cause in _exception_causes(exc):
        raw_text = str(cause).strip()
        lines.append(
            f"{type(cause).__name__}: {redact_sensitive_error_text(raw_text)}"
            if raw_text
            else type(cause).__name__
        )
    if not lines:
        return None
    text, _ = _cap_text_bytes("\n".join(lines))
    return text


def _category_hint_from_body(body: Any, body_text: str | None) -> str | None:
    parsed = body
    if isinstance(parsed, bytes):
        parsed = parsed.decode("utf-8", errors="replace")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            for key in ("type", "code"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("type", "code"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if (
        body_text
        and "model" in body_text.lower()
        and "unsupported" in body_text.lower()
    ):
        return "upstream_model_error"
    return None


def _append_request_id_lines(lines: list[str], request_id: str | None) -> None:
    if request_id:
        lines.extend(("", f"Request ID: {request_id}"))
