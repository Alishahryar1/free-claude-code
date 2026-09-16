"""Limits and defaults for outbound web server tool HTTP."""

from free_claude_code.core.version import package_version

_REQUEST_TIMEOUT_S = 20.0
_MAX_SEARCH_RESULTS = 10
_MAX_FETCH_CHARS = 24_000
# Hard cap on raw bytes read from HTTP responses before decode / HTML parse (memory bound).
_MAX_WEB_FETCH_RESPONSE_BYTES = 2 * 1024 * 1024
# Drain at most this many bytes from redirect responses before following Location.
_REDIRECT_RESPONSE_BODY_CAP_BYTES = 65_536
_MAX_WEB_FETCH_REDIRECTS = 10
_WEB_FETCH_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

_WEB_TOOL_HTTP_HEADERS = {
    "User-Agent": f"Mozilla/5.0 compatible; free-claude-code/{package_version()}",
}
