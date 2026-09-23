"""Pure smoke outcome classification shared by runners and reports."""

import re

_UPSTREAM_UNAVAILABLE_MARKERS = (
    "upstream_unavailable",
    "connection refused",
    "connecterror",
    "connect timeout",
    "connecttimeout",
    "proxyerror",
    "readtimeout",
    "remoteprotocolerror",
    "server disconnected",
    "service unavailable",
    "temporary failure",
    "timed out",
    "not reachable",
    "rate limit exceeded",
    "rate limited",
    "overloaded",
    "at capacity",
)
_RETRYABLE_HTTP_STATUS_PATTERNS = (
    r"\bhttp(?:/1\.[01])?[\"']?\s+(?:429|5\d{2})\b",
    r"\bstatus(?:_code| code)?(?:[=:]\s*|\s+)(?:429|5\d{2})\b",
    r"\berror code:\s*(?:429|5\d{2})\b",
    r"\b(?:client|server) error [\"'](?:429|5\d{2})\b",
    r"\b429 too many requests\b",
)


def is_upstream_unavailable_text(text: str) -> bool:
    """Return whether text contains a concrete transient-upstream signal."""
    normalized = text.lower()
    if any(marker in normalized for marker in _UPSTREAM_UNAVAILABLE_MARKERS):
        return True
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _RETRYABLE_HTTP_STATUS_PATTERNS
    )


def classify_outcome(*, nodeid: str, outcome: str, detail: str) -> str:
    """Classify one smoke outcome for triage reports."""
    if outcome == "passed":
        return "passed"

    text = f"{nodeid}\n{detail}"
    normalized = text.lower()
    if outcome == "skipped":
        if "smoke target disabled" in normalized:
            return "target_disabled"
        if "missing_env" in normalized:
            return "missing_env"
        if is_upstream_unavailable_text(text):
            return "upstream_unavailable"
        return "missing_env"

    if "harness_bug" in normalized:
        return "harness_bug"
    if "missing_env" in normalized:
        return "missing_env"
    if is_upstream_unavailable_text(text):
        return "upstream_unavailable"
    return "product_failure"
