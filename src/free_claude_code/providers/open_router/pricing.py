"""Cheapest-provider resolution from OpenRouter per-model endpoint pricing."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from loguru import logger

DEFAULT_ENDPOINTS_TTL_S = 600.0
PRICING_FETCH_TIMEOUT_S = 5.0

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
    except KeyError, TypeError, ValueError, OverflowError:
        return None
    if not (math.isfinite(prompt) and math.isfinite(completion)):
        # NaN and infinities would poison the comparisons in
        # cheapest_provider_name: every ordering test against them is
        # false, so a non-finite first candidate would never be replaced.
        return None
    blended = (prompt + completion) / 2
    return blended if math.isfinite(blended) else None


def prompt_capacity(endpoint: Mapping[str, Any]) -> int | None:
    """Return the endpoint's served prompt capacity, when the payload states it.

    ``max_prompt_tokens`` is the input ceiling when present; otherwise the
    total context length. Missing or malformed fields leave capacity unknown.
    """

    capacities: list[int] = []
    for key in ("max_prompt_tokens", "context_length"):
        value = endpoint.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            capacities.append(value)
    return min(capacities) if capacities else None


def supports_required_parameters(
    endpoint: Mapping[str, Any], required: AbstractSet[str] | None
) -> bool:
    """Return whether the endpoint declares support for every required parameter.

    ``required`` holds the endpoint parameters the request itself uses, such
    as ``tools`` or ``reasoning``; a request that uses none needs no check.
    A request that uses some only qualifies endpoints that explicitly list
    them: the pin runs under ``allow_fallbacks: false``, so an endpoint
    that cannot serve the request would fail it outright.
    """

    if not required:
        return True
    supported = endpoint.get("supported_parameters")
    if not isinstance(supported, list) or not all(
        isinstance(name, str) for name in supported
    ):
        return False
    return required <= set(supported)


def cheapest_provider_name(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    quantizations: Sequence[str] | None = None,
    free_only: bool = False,
    excluded: Sequence[str] = (),
    prompt_tokens: int = 0,
    required_parameters: AbstractSet[str] | None = None,
) -> str | None:
    """Return the provider of the cheapest blended-rate endpoint."""

    excluded_names = frozenset(excluded)
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
        if name in excluded_names:
            # A static routing block may ignore this provider; pinning it
            # would leave the merged block with no eligible endpoint.
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
        capacity = prompt_capacity(endpoint)
        if capacity is not None and prompt_tokens > int(capacity * 0.9):
            # cl100k estimates undercount some tokenizers (~8% observed), so
            # a request past 90% of the served ceiling is already unsafe;
            # pinning it under allow_fallbacks: false would fail outright.
            continue
        if not supports_required_parameters(endpoint, required_parameters):
            # The endpoint does not declare a parameter the request uses;
            # pinning it under allow_fallbacks: false would leave the
            # request with no eligible endpoint.
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
        fetch_timeout_s: float = PRICING_FETCH_TIMEOUT_S,
    ) -> None:
        self._fetch = fetch
        self._quantizations = quantizations or None
        self._ttl_s = ttl_s
        self._clock = clock
        self._fetch_timeout_s = fetch_timeout_s
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
            task.add_done_callback(self._drop_inflight(base))
        # Shield the shared fetch: one caller disconnecting must not cancel
        # the lookup that concurrent requests for the same model still need.
        await asyncio.shield(task)

    def needs_refresh(self, model_id: str) -> bool:
        """Return whether a request for this model must refresh pricing first."""

        base = base_model_slug(model_id)
        if base is None:
            return False
        entry = self._entries.get(base)
        return entry is None or self._clock() - entry[0] >= self._ttl_s

    def _drop_inflight(self, base: str) -> Callable[[asyncio.Task[None]], None]:
        """Return a done-callback that retires this model's finished fetch."""

        def drop(task: asyncio.Task[None]) -> None:
            if self._inflight.get(base) is task:
                del self._inflight[base]

        return drop

    async def _load(self, base: str) -> None:
        try:
            endpoints = await asyncio.wait_for(
                self._fetch(base), timeout=self._fetch_timeout_s
            )
        except Exception as exc:
            # Pricing lookup is best-effort; degrade to unconstrained routing.
            # The bounded timeout keeps a stalled metadata response from
            # holding up chat startup behind the client's read timeout.
            logger.warning(
                "OpenRouter endpoint pricing fetch failed for {}: {}", base, exc
            )
            endpoints = None
        self._entries[base] = (self._clock(), endpoints)

    def cheapest_provider_block(
        self,
        model_id: str,
        *,
        constraints: Mapping[str, Any] | None = None,
        prompt_tokens: int = 0,
        required_parameters: AbstractSet[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return the routing block pinning the cheapest provider, if resolved.

        ``constraints`` is the static routing block that survives the merge;
        its ``ignore`` and (when no dynamic selection applies) ``quantizations``
        filters constrain candidacy so the pin cannot contradict them.
        ``prompt_tokens`` is the estimated request size; pinning a provider
        whose served capacity cannot fit it would leave the chat with no
        eligible endpoint, so only providers that can serve it are eligible.
        ``required_parameters`` are the endpoint parameters the request uses
        (tools, reasoning); the pin may only land on a provider that declares
        support for them.
        """

        base = base_model_slug(model_id)
        if base is None:
            return None
        entry = self._entries.get(base)
        if entry is None or entry[1] is None:
            return None
        excluded: tuple[str, ...] = ()
        quantizations = self._quantizations
        if constraints is not None:
            ignored = constraints.get("ignore")
            if isinstance(ignored, list):
                excluded = tuple(name for name in ignored if isinstance(name, str))
            if self._quantizations is None:
                static_levels = constraints.get("quantizations")
                if isinstance(static_levels, list) and all(
                    isinstance(level, str) for level in static_levels
                ):
                    quantizations = tuple(static_levels)
        name = cheapest_provider_name(
            entry[1],
            quantizations=quantizations,
            free_only=model_id.strip().endswith(":free"),
            excluded=excluded,
            prompt_tokens=prompt_tokens,
            required_parameters=required_parameters,
        )
        if name is None:
            return None
        block: dict[str, Any] = {"only": [name], "allow_fallbacks": False}
        if self._quantizations is not None:
            block["quantizations"] = list(self._quantizations)
        return block
