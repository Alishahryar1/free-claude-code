"""Recognize Groq token-limit rejections and correct combined TPM output budgets."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from free_claude_code.core.json_types import JsonObject, JsonValue

_MAX_DECIMAL_DIGITS = 19
_TPM_PREFIX = r"(?<![a-z])tokens\s+per\s+minute\s*\(\s*tpm\s*\)\s*:\s*"
_TPM_MARKER = re.compile(_TPM_PREFIX, re.IGNORECASE)
_TPM_CLAUSE = re.compile(
    _TPM_PREFIX
    + rf"limit\s+([0-9]{{1,{_MAX_DECIMAL_DIGITS}}})(?![0-9])\s*,\s*"
    + rf"requested\s+([0-9]{{1,{_MAX_DECIMAL_DIGITS}}})"
    + r"(?=\s*(?:,(?!\s*[0-9])|$))",
    re.IGNORECASE,
)
_OUTPUT_FIELDS = frozenset({"max_completion_tokens", "max_tokens"})
_DETAIL_FIELDS = frozenset({"message", "type", "code"})


@dataclass(frozen=True, slots=True)
class GroqTpmCorrection:
    """One validated completion-budget correction and its safe diagnostics."""

    body: JsonObject
    limit: int
    requested: int
    previous_max_completion_tokens: int
    corrected_max_completion_tokens: int


def token_limit_message(error: Exception) -> str | None:
    """Recognize Groq's structured token-rate rejection, regardless of quota wording."""
    if _status_code(error) != 413:
        return None

    detail = _error_detail(getattr(error, "body", None))
    if detail is None:
        return None
    if detail.get("type") != "tokens":
        return None
    if detail.get("code") != "rate_limit_exceeded":
        return None

    message = detail.get("message")
    return message if isinstance(message, str) else None


def correct_tpm_completion_budget(
    error: Exception,
    body: Mapping[str, JsonValue],
) -> GroqTpmCorrection | None:
    """Subtract Groq's reported TPM overage from this request's output cap."""
    message = token_limit_message(error)
    if message is None:
        return None
    if sum(1 for _ in _TPM_MARKER.finditer(message)) != 1:
        return None
    matches = tuple(_TPM_CLAUSE.finditer(message))
    if len(matches) != 1:
        return None

    limit = int(matches[0].group(1))
    requested = int(matches[0].group(2))
    if limit <= 0 or requested <= limit:
        return None

    previous = body.get("max_completion_tokens")
    if not isinstance(previous, int) or isinstance(previous, bool) or previous <= 0:
        return None
    if requested <= previous:
        return None
    extra_body = body.get("extra_body")
    if isinstance(extra_body, Mapping) and any(
        field in extra_body for field in _OUTPUT_FIELDS
    ):
        return None

    corrected = previous - (requested - limit)
    if corrected <= 0 or corrected >= previous:
        return None

    corrected_body = dict(body)
    corrected_body["max_completion_tokens"] = corrected
    return GroqTpmCorrection(
        body=corrected_body,
        limit=limit,
        requested=requested,
        previous_max_completion_tokens=previous,
        corrected_max_completion_tokens=corrected,
    )


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _error_detail(value: object) -> Mapping[object, object] | None:
    if not isinstance(value, Mapping):
        return None
    candidates: list[Mapping[object, object]] = []
    if _DETAIL_FIELDS.issubset(value):
        candidates.append(value)
    nested = value.get("error")
    if isinstance(nested, Mapping) and _DETAIL_FIELDS.issubset(nested):
        candidates.append(nested)
    return candidates[0] if len(candidates) == 1 else None
