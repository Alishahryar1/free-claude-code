"""Outbound capabilities consumed by the local web-tool workflow."""

from dataclasses import dataclass
from typing import Protocol

from free_claude_code.core.web_tools import WebFetchResult, WebSearchResult


@dataclass(frozen=True, slots=True)
class WebFetchEgressPolicy:
    """Egress rules for user-influenced web_fetch URLs."""

    allow_private_network_targets: bool
    allowed_schemes: frozenset[str]


class WebFetchEgressViolation(ValueError):
    """Raised when a web_fetch URL is rejected by egress policy (SSRF guard)."""


def web_fetch_allowed_scheme_set(raw_schemes: str) -> frozenset[str]:
    """Return normalized schemes allowed for web_fetch."""
    return frozenset(
        part.strip().lower() for part in raw_schemes.split(",") if part.strip()
    )


class WebToolsPort(Protocol):
    """Complete each operation before returning data, retaining no response resource."""

    async def search(self, query: str) -> list[WebSearchResult]: ...

    async def fetch(
        self, url: str, *, egress: WebFetchEgressPolicy
    ) -> WebFetchResult: ...
