"""SDK-independent endpoint snapshots shared with account owners."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HttpEndpoint:
    """A resolved API root and credentials; callers own validation and lifetime."""

    base_url: str
    headers: Mapping[str, str] = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    account_id: str | None = field(default=None, repr=False)


class EndpointContext(Protocol):
    """Borrow a current snapshot without changing credentials on shared clients."""

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint: ...
