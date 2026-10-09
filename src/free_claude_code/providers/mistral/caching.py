"""Mistral prompt-cache affinity headers.

Mistral's official CLI (mistral-vibe) enables prompt caching by sending the
conversation session id as the ``x-affinity`` HTTP header on every request
(``vibe/core/agent_loop/_loop.py`` in mistralai/mistral-vibe; mirrored by
zed-industries/zed#48584). The header is not part of Mistral's public API
contract, so this module mirrors the official client behavior rather than a
documented request field.
"""

from collections.abc import Mapping

MISTRAL_AFFINITY_HEADER = "x-affinity"

# Ordered by specificity: the first non-empty header wins.
MISTRAL_SESSION_HEADER_NAMES: tuple[str, ...] = (
    "x-opencode-session",
    "anthropic-session-id",
    "x-anthropic-session-id",
    "claude-session-id",
    "x-claude-session-id",
    "x-claude-code-session-id",
    "session-id",
    "x-session-id",
)


def extract_mistral_affinity_key(headers: Mapping[str, str]) -> str | None:
    """Return the first non-empty session header value, case-insensitive."""
    lowered = {
        str(name).lower(): value
        for name, value in headers.items()
        if isinstance(value, str)
    }
    for name in MISTRAL_SESSION_HEADER_NAMES:
        value = lowered.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def mistral_affinity_headers(
    request_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return upstream headers carrying the Mistral cache-affinity session id.

    Only the affinity header is forwarded; other client headers (credentials,
    user-agent) must never leak upstream through this path.
    """
    if not request_headers:
        return {}
    key = extract_mistral_affinity_key(request_headers)
    if key is None:
        return {}
    return {MISTRAL_AFFINITY_HEADER: key}
