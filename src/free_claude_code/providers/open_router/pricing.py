"""Cheapest-provider resolution from OpenRouter per-model endpoint pricing."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from loguru import logger

DEFAULT_ENDPOINTS_TTL_S = 600.0

EndpointFetcher = Callable[[str], Awaitable[Sequence[Mapping[str, Any]] | None]]


def base_model_slug(model_id: str) -> str | None:
    """Return the variant-stripped author/slug for the endpoints API."""

    author, separator, slug = model_id.strip().partition("/")
    if not separator or not author:
        return None
    base = slug.split(":", 1)[0]
    return f"{author}/{base}" if base else None


def endpoints_from_payload(payload: object) -> Sequence[Mapping[str, Any]] | None:
    """Extract the endpoint list from an endpoints API response."""

    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        return None
    return [endpoint for endpoint in endpoints if isinstance(endpoint, Mapping)]


def blended_price(pricing: object) -> float | None:
    """Average of per-token prompt and completion prices; None when unknown."""

    if not isinstance(pricing, Mapping):
        return None
    try:
        prompt = float(pricing["prompt"])
        completion = float(pricing["completion"])
    except KeyError, TypeError, ValueError:
        return None
    return (prompt + completion) / 2


def cheapest_provider_name(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    quantizations: Sequence[str] | None = None,
    free_only: bool = False,
) -> str | None:
    """Return the provider of the cheapest blended-rate endpoint."""

    best_name: str | None = None
    best_price: float | None = None
    for endpoint in endpoints:
        if endpoint.get("status", 0):
            # A non-zero status marks a disrupted endpoint; pinning one would
            # fail the request outright under allow_fallbacks: false.
            continue
        name = endpoint.get("provider_name")
        if not isinstance(name, str) or not name:
            continue
        price = blended_price(endpoint.get("pricing"))
        if price is None:
            continue
        if free_only and price != 0:
            continue
        if quantizations is not None:
            level = endpoint.get("quantization") or "unknown"
            if level not in quantizations:
                continue
        if best_price is None or price < best_price:
            best_name, best_price = name, price
    return best_name


class CheapestProviderIndex:
    """TTL cache of per-model endpoint pricing with in-flight coalescing."""

    def __init__(
        self,
        fetch: EndpointFetcher,
        *,
        quantizations: tuple[str, ...] | None = None,
        ttl_s: float = DEFAULT_ENDPOINTS_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._quantizations = quantizations or None
        self._ttl_s = ttl_s
        self._clock = clock
        self._entries: dict[str, tuple[float, Sequence[Mapping[str, Any]] | None]] = {}
        self._inflight: dict[str, asyncio.Task[None]] = {}

    async def ensure(self, model_id: str) -> None:
        """Refresh one model's cached endpoints when missing or stale."""

        base = base_model_slug(model_id)
        if base is None:
            return
        entry = self._entries.get(base)
        if entry is not None and self._clock() - entry[0] < self._ttl_s:
            return
        task = self._inflight.get(base)
        if task is None:
            task = asyncio.create_task(self._load(base))
            self._inflight[base] = task
        try:
            await task
        finally:
            if self._inflight.get(base) is task:
                del self._inflight[base]

    def needs_refresh(self, model_id: str) -> bool:
        """Return whether a request for this model must refresh pricing first."""

        base = base_model_slug(model_id)
        if base is None:
            return False
        entry = self._entries.get(base)
        return entry is None or self._clock() - entry[0] >= self._ttl_s

    async def _load(self, base: str) -> None:
        try:
            endpoints = await self._fetch(base)
        except Exception as exc:
            # Pricing lookup is best-effort; degrade to unconstrained routing.
            logger.warning(
                "OpenRouter endpoint pricing fetch failed for {}: {}", base, exc
            )
            endpoints = None
        self._entries[base] = (self._clock(), endpoints)

    def cheapest_provider_block(self, model_id: str) -> dict[str, Any] | None:
        """Return the routing block pinning the cheapest provider, if resolved."""

        base = base_model_slug(model_id)
        if base is None:
            return None
        entry = self._entries.get(base)
        if entry is None or entry[1] is None:
            return None
        name = cheapest_provider_name(
            entry[1],
            quantizations=self._quantizations,
            free_only=model_id.strip().endswith(":free"),
        )
        if name is None:
            return None
        block: dict[str, Any] = {"only": [name], "allow_fallbacks": False}
        if self._quantizations is not None:
            block["quantizations"] = list(self._quantizations)
        return block
