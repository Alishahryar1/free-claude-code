"""Tests for OpenRouter cheapest-provider resolution."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.open_router.pricing import (
    CheapestProviderIndex,
    base_model_slug,
    blended_price,
    cheapest_provider_name,
    endpoints_from_payload,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, make_provider_config


def endpoint(name, prompt, completion, *, quantization="fp8", status=0):
    return {
        "provider_name": name,
        "pricing": {"prompt": str(prompt), "completion": str(completion)},
        "quantization": quantization,
        "status": status,
    }


MIXED_ENDPOINTS = [
    endpoint("Expensive", "0.000001", "0.000002", quantization="fp16"),
    endpoint("Cheap", "0.0000001", "0.0000002"),
    endpoint("Down", "0.00000005", "0.0000001", status=-2),
]


def test_base_model_slug_strips_routing_variants():
    assert base_model_slug("z-ai/glm-5.3-flash:floor") == "z-ai/glm-5.3-flash"
    assert base_model_slug("moonshotai/kimi-k2.6:free") == "moonshotai/kimi-k2.6"
    assert base_model_slug(" vendor/model ") == "vendor/model"
    assert base_model_slug("noslash") is None
    assert base_model_slug("vendor/") is None


def test_endpoints_from_payload_reads_the_documented_shape():
    payload = {"data": {"endpoints": [endpoint("Chutes", "0", "0")]}}
    assert endpoints_from_payload(payload) == [endpoint("Chutes", "0", "0")]
    assert endpoints_from_payload({"data": {"endpoints": "bad"}}) is None
    assert endpoints_from_payload("nope") is None


def test_blended_price_averages_prompt_and_completion():
    assert (
        blended_price({"prompt": "0.0000001", "completion": "0.0000003"}) == 0.0000002
    )
    assert blended_price({"prompt": "x", "completion": "1"}) is None
    assert blended_price({"prompt": "1"}) is None
    assert blended_price(None) is None


@pytest.mark.parametrize(
    "rate",
    ["NaN", "Infinity", "-Infinity", "1e400", 10**400, float("nan")],
)
def test_blended_price_rejects_non_finite_rates(rate):
    assert blended_price({"prompt": rate, "completion": rate}) is None


def test_blended_price_rejects_a_blend_that_overflows():
    assert blended_price({"prompt": 1e308, "completion": 1e308}) is None


@pytest.mark.parametrize("rate", ["NaN", "Infinity", "-Infinity", 10**400])
def test_unusable_endpoint_pricing_is_ignored(rate):
    endpoints = [
        {
            "provider_name": "invalid",
            "status": 0,
            "pricing": {"prompt": rate, "completion": rate},
        }
    ]
    assert cheapest_provider_name(endpoints) is None


def test_non_finite_candidate_does_not_block_a_later_finite_one():
    endpoints = [
        endpoint("NaN-priced", "NaN", "NaN"),
        endpoint("Finite", "0.0000001", "0.0000002"),
    ]
    assert cheapest_provider_name(endpoints) == "Finite"


def test_cheapest_provider_skips_disrupted_endpoints():
    assert cheapest_provider_name(MIXED_ENDPOINTS) == "Cheap"


def test_cheapest_provider_filters_by_quantization():
    assert (
        cheapest_provider_name(MIXED_ENDPOINTS, quantizations=("fp16",)) == "Expensive"
    )
    assert cheapest_provider_name(MIXED_ENDPOINTS, quantizations=("int8",)) is None


def test_cheapest_provider_free_only_keeps_zero_priced_endpoints():
    endpoints = [endpoint("Paid", "0.1", "0.1"), endpoint("Free", "0", "0")]
    assert cheapest_provider_name(endpoints, free_only=True) == "Free"
    assert cheapest_provider_name(endpoints) == "Free"


def test_cheapest_provider_treats_null_quantization_as_unknown():
    endpoints = [
        {"provider_name": "Null", "pricing": {"prompt": "1", "completion": "1"}}
    ]
    assert cheapest_provider_name(endpoints, quantizations=("unknown",)) == "Null"
    assert cheapest_provider_name(endpoints, quantizations=("fp8",)) is None


@pytest.mark.asyncio
async def test_index_caches_within_ttl_and_refetches_after_expiry():
    calls: list[str] = []
    now = [1000.0]

    async def fetch(base):
        calls.append(base)
        return MIXED_ENDPOINTS

    index = CheapestProviderIndex(fetch=fetch, ttl_s=60, clock=lambda: now[0])

    await index.ensure("vendor/model:free")
    await index.ensure("vendor/model")
    assert calls == ["vendor/model"]

    now[0] += 61
    await index.ensure("vendor/model")
    assert calls == ["vendor/model", "vendor/model"]


@pytest.mark.asyncio
async def test_index_coalesces_concurrent_fetches():
    calls: list[str] = []
    gate = asyncio.Event()

    async def fetch(base):
        calls.append(base)
        await gate.wait()
        return MIXED_ENDPOINTS

    index = CheapestProviderIndex(fetch=fetch)
    first = asyncio.create_task(index.ensure("vendor/model"))
    second = asyncio.create_task(index.ensure("vendor/model"))
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(first, second)

    assert calls == ["vendor/model"]


@pytest.mark.asyncio
async def test_index_degrades_when_the_fetch_fails():
    calls: list[str] = []

    async def fetch(base):
        calls.append(base)
        raise RuntimeError("upstream down")

    index = CheapestProviderIndex(fetch=fetch)

    await index.ensure("vendor/model")

    assert index.cheapest_provider_block("vendor/model") is None
    await index.ensure("vendor/model")
    assert calls == ["vendor/model"]


def test_provider_block_pins_the_cheapest_provider():
    index = CheapestProviderIndex(
        fetch=_endpoints_fetch(MIXED_ENDPOINTS), quantizations=("fp8",)
    )
    asyncio.run(index.ensure("vendor/model"))

    assert index.cheapest_provider_block("vendor/model") == {
        "only": ["Cheap"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
    }


def test_provider_block_omits_quantizations_when_unset():
    index = CheapestProviderIndex(fetch=_endpoints_fetch(MIXED_ENDPOINTS))
    asyncio.run(index.ensure("vendor/model"))

    assert index.cheapest_provider_block("vendor/model") == {
        "only": ["Cheap"],
        "allow_fallbacks": False,
    }


def test_provider_block_honors_the_free_variant():
    endpoints = [endpoint("Paid", "0.1", "0.1"), endpoint("Free", "0", "0")]
    index = CheapestProviderIndex(fetch=_endpoints_fetch(endpoints))
    asyncio.run(index.ensure("vendor/model"))

    assert index.cheapest_provider_block("vendor/model:free") == {
        "only": ["Free"],
        "allow_fallbacks": False,
    }


def test_provider_block_is_absent_without_resolvable_pricing():
    index = CheapestProviderIndex(fetch=_endpoints_fetch([]))
    asyncio.run(index.ensure("vendor/model"))

    assert index.cheapest_provider_block("vendor/model") is None


def _endpoints_fetch(endpoints):
    async def fetch(base):
        return endpoints

    return fetch


def _cheapest_provider(**kwargs):
    return OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        use_cheapest_provider=True,
        admission=immediate_admission(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_provider_pins_cheapest_provider_in_the_request_body():
    provider = _cheapest_provider(quantizations=("fp8",))
    assert provider._cheapest_index is not None
    endpoints_payload = {"data": {"endpoints": MIXED_ENDPOINTS}}

    with patch.object(
        provider._client, "get", new_callable=AsyncMock, return_value=endpoints_payload
    ) as get_mock:
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request("moonshotai/kimi-k2.6")
            )
        )

    assert get_mock.call_args.args[0] == "/models/moonshotai/kimi-k2.6/endpoints"
    assert body["provider"] == {
        "only": ["Cheap"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
    }


def test_index_reports_refresh_need_by_cache_state():
    async def fetch(base):
        return MIXED_ENDPOINTS

    now = [1000.0]
    index = CheapestProviderIndex(fetch=fetch, ttl_s=60, clock=lambda: now[0])

    assert index.needs_refresh("moonshotai/kimi-k2.6") is True
    assert index.needs_refresh("unparseable") is False

    asyncio.run(index.ensure("moonshotai/kimi-k2.6"))

    assert index.needs_refresh("moonshotai/kimi-k2.6") is False
    now[0] += 61
    assert index.needs_refresh("moonshotai/kimi-k2.6") is True


@pytest.mark.asyncio
async def test_provider_merges_static_preferences_with_the_cheapest_pin():
    from free_claude_code.config.open_router_preferences import (
        OpenRouterProviderPreferences,
    )

    preferences = OpenRouterProviderPreferences.model_validate(
        {"data_collection": "deny", "only": ["Ignored"], "allow_fallbacks": True}
    )
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        provider_preferences=preferences,
        use_cheapest_provider=True,
        admission=immediate_admission(),
    )
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value={"data": {"endpoints": MIXED_ENDPOINTS}},
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request("moonshotai/kimi-k2.6")
            )
        )

    assert body["provider"] == {
        "data_collection": "deny",
        "only": ["Cheap"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_provider_omits_the_provider_field_when_cheapest_is_unresolved():
    provider = _cheapest_provider()
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client, "get", new_callable=AsyncMock, return_value={"data": {}}
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request("moonshotai/kimi-k2.6")
            )
        )

    assert "provider" not in body


def test_provider_without_cheapest_mode_never_fetches_pricing():
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        admission=immediate_admission(),
    )

    assert provider._cheapest_index is None


def _tracking_stream(closed, *, started=None, block=False):
    async def inner():
        try:
            if started is not None:
                started.set()
            yield "first"
            if block:
                await asyncio.Event().wait()
            yield "second"
        finally:
            closed.append("closed")

    return inner()


@pytest.mark.asyncio
async def test_refresh_wrapper_forwards_all_events_and_closes_once():
    closed: list[str] = []
    provider = _cheapest_provider()
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value={"data": {"endpoints": MIXED_ENDPOINTS}},
    ):
        events = [
            event
            async for event in provider._refresh_then_stream(
                "moonshotai/kimi-k2.6", _tracking_stream(closed)
            )
        ]

    assert events == ["first", "second"]
    assert closed == ["closed"]


@pytest.mark.asyncio
async def test_refresh_wrapper_closes_the_inner_stream_when_abandoned():
    closed: list[str] = []
    provider = _cheapest_provider()
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value={"data": {"endpoints": MIXED_ENDPOINTS}},
    ):
        stream = provider._refresh_then_stream(
            "moonshotai/kimi-k2.6", _tracking_stream(closed)
        )
        assert await anext(stream) == "first"
        await stream.aclose()

    assert closed == ["closed"]


@pytest.mark.asyncio
async def test_refresh_wrapper_closes_the_inner_stream_on_cancellation():
    closed: list[str] = []
    started = asyncio.Event()
    provider = _cheapest_provider()
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value={"data": {"endpoints": MIXED_ENDPOINTS}},
    ):
        stream = provider._refresh_then_stream(
            "moonshotai/kimi-k2.6",
            _tracking_stream(closed, started=started, block=True),
        )

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert closed == ["closed"]
