"""Tests for OpenRouter cheapest-provider resolution."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.providers.continuation import ContinuationRequest
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.open_router.pricing import (
    CheapestProviderIndex,
    base_model_slug,
    blended_price,
    cheapest_provider_name,
    endpoints_from_payload,
    prompt_capacity,
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


def test_prompt_capacity_uses_the_stricter_stated_ceiling():
    assert prompt_capacity({"max_prompt_tokens": 4000, "context_length": 8192}) == 4000
    assert prompt_capacity({"max_prompt_tokens": 9000, "context_length": 8192}) == 8192
    assert prompt_capacity({"context_length": 8192}) == 8192
    assert prompt_capacity({"max_prompt_tokens": 4096}) == 4096
    assert prompt_capacity({}) is None
    assert prompt_capacity({"context_length": "8192"}) is None
    assert prompt_capacity({"context_length": 0}) is None
    assert prompt_capacity({"context_length": True}) is None


def test_cheapest_provider_skips_endpoints_too_small_for_the_prompt():
    endpoints = [
        endpoint("Tiny", "0.0000001", "0.0000001") | {"context_length": 32_000},
        endpoint("Roomy", "0.000001", "0.000001") | {"context_length": 200_000},
    ]
    assert cheapest_provider_name(endpoints, prompt_tokens=1_000) == "Tiny"
    assert cheapest_provider_name(endpoints, prompt_tokens=100_000) == "Roomy"
    assert cheapest_provider_name(endpoints, prompt_tokens=1_000_000) is None


def test_cheapest_provider_applies_the_capacity_margin():
    tiny = endpoint("Tiny", "0.0000001", "0.0000001") | {"context_length": 32_000}
    # Below 90% of 32_000 fits; at 90% it still fits; past it, the estimate
    # may already undercount the real token count, so the endpoint is out.
    assert cheapest_provider_name([tiny], prompt_tokens=28_799) == "Tiny"
    assert cheapest_provider_name([tiny], prompt_tokens=28_801) is None


def test_cheapest_provider_keeps_endpoints_with_unknown_capacity():
    endpoints = [endpoint("Unknown", "0.0000001", "0.0000001")]
    assert cheapest_provider_name(endpoints, prompt_tokens=1_000_000) == "Unknown"


def test_cheapest_provider_uses_max_prompt_tokens_as_capacity():
    roomy_context = endpoint("Capped", "0.0000001", "0.0000001") | {
        "context_length": 200_000,
        "max_prompt_tokens": 8_000,
    }
    assert cheapest_provider_name([roomy_context], prompt_tokens=5_000) == "Capped"
    assert cheapest_provider_name([roomy_context], prompt_tokens=50_000) is None


@pytest.mark.asyncio
async def test_long_chat_moves_the_pin_to_a_provider_that_fits():
    endpoints_payload = {
        "data": {
            "endpoints": [
                endpoint("TinyCheap", "0.0000001", "0.0000001")
                | {"context_length": 32_000},
                endpoint("RoomyExpensive", "0.000001", "0.000001")
                | {"context_length": 200_000},
            ]
        }
    }
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        use_cheapest_provider=True,
        admission=immediate_admission(),
    )
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=endpoints_payload,
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request(
                    "moonshotai/kimi-k2.6",
                    messages=[{"role": "user", "content": "word " * 40_000}],
                )
            )
        )

    assert body["extra_body"]["provider"] == {
        "only": ["RoomyExpensive"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_tools_count_toward_the_capacity_requirement():
    endpoints_payload = {
        "data": {
            "endpoints": [
                endpoint("TinyCheap", "0.0000001", "0.0000001")
                | {"context_length": 32_000},
                endpoint("RoomyExpensive", "0.000001", "0.000001")
                | {"context_length": 200_000},
            ]
        }
    }
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        use_cheapest_provider=True,
        admission=immediate_admission(),
    )
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=endpoints_payload,
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request(
                    "moonshotai/kimi-k2.6",
                    tools=[
                        {
                            "name": f"tool_{index}",
                            "description": "word " * 4_000,
                            "input_schema": {"type": "object"},
                        }
                        for index in range(20)
                    ],
                )
            )
        )

    assert body["extra_body"]["provider"] == {
        "only": ["RoomyExpensive"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_continuation_growth_moves_the_pin_to_a_provider_that_fits():
    endpoints_payload = {
        "data": {
            "endpoints": [
                endpoint("TinyCheap", "0.0000001", "0.0000001")
                | {"context_length": 32_000},
                endpoint("RoomyExpensive", "0.000001", "0.000001")
                | {"context_length": 200_000},
            ]
        }
    }
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        use_cheapest_provider=True,
        admission=immediate_admission(),
    )
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=endpoints_payload,
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        original = provider._chat._build_request_body(
            make_messages_request("moonshotai/kimi-k2.6")
        )
        # The original request fits the cheapest provider...
        first = provider._chat._behavior.prepare_create_body(original)
        # ...and the continuation path appends the interrupted reply and the
        # continuation instructions to the same body. The pin must re-resolve
        # against the grown body, not the original request's size.
        grown = ContinuationRequest("chat").build(original, "reply " * 40_000, "")
        second = provider._chat._behavior.prepare_create_body(grown)

    assert first["extra_body"]["provider"] == {
        "only": ["TinyCheap"],
        "allow_fallbacks": False,
    }
    assert second["extra_body"]["provider"] == {
        "only": ["RoomyExpensive"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_pin_is_dropped_when_no_provider_can_fit_the_chat():
    endpoints_payload = {
        "data": {
            "endpoints": [
                endpoint("Tiny", "0.0000001", "0.0000001") | {"context_length": 32_000},
                endpoint("AlsoTiny", "0.000001", "0.000001")
                | {"context_length": 32_000},
            ]
        }
    }
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        use_cheapest_provider=True,
        admission=immediate_admission(),
    )
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=endpoints_payload,
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request(
                    "moonshotai/kimi-k2.6",
                    messages=[{"role": "user", "content": "word " * 40_000}],
                )
            )
        )

    assert "provider" not in body.get("extra_body", {})


@pytest.mark.asyncio
async def test_a_small_body_keeps_the_cheapest_pin():
    endpoints_payload = {
        "data": {
            "endpoints": [
                endpoint("TinyCheap", "0.0000001", "0.0000001")
                | {"context_length": 32_000},
                endpoint("RoomyExpensive", "0.000001", "0.000001")
                | {"context_length": 200_000},
            ]
        }
    }
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
        ),
        use_cheapest_provider=True,
        admission=immediate_admission(),
    )
    assert provider._cheapest_index is not None

    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=endpoints_payload,
    ):
        await provider._cheapest_index.ensure("moonshotai/kimi-k2.6")
        body = provider._chat._behavior.prepare_create_body(
            provider._chat._build_request_body(
                make_messages_request("moonshotai/kimi-k2.6")
            )
        )

    assert body["extra_body"]["provider"] == {
        "only": ["TinyCheap"],
        "allow_fallbacks": False,
    }


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
    assert body["extra_body"]["provider"] == {
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

    assert body["extra_body"]["provider"] == {
        "data_collection": "deny",
        "only": ["Cheap"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_static_ignore_excludes_a_pinned_candidate():
    from free_claude_code.config.open_router_preferences import (
        OpenRouterProviderPreferences,
    )

    preferences = OpenRouterProviderPreferences.model_validate({"ignore": ["Cheap"]})
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

    assert body["extra_body"]["provider"] == {
        "ignore": ["Cheap"],
        "only": ["Expensive"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_static_quantizations_constrain_candidacy_when_unset():
    from free_claude_code.config.open_router_preferences import (
        OpenRouterProviderPreferences,
    )

    preferences = OpenRouterProviderPreferences.model_validate(
        {"quantizations": ["fp16"]}
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

    assert body["extra_body"]["provider"] == {
        "quantizations": ["fp16"],
        "only": ["Expensive"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_static_filters_that_exclude_every_candidate_drop_the_pin():
    from free_claude_code.config.open_router_preferences import (
        OpenRouterProviderPreferences,
    )

    preferences = OpenRouterProviderPreferences.model_validate(
        {"ignore": ["Cheap", "Expensive"]}
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

    assert body["extra_body"]["provider"] == {"ignore": ["Cheap", "Expensive"]}


@pytest.mark.asyncio
async def test_one_callers_cancellation_does_not_cancel_the_shared_fetch():
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def fetch(base):
        calls.append(base)
        started.set()
        await release.wait()
        return MIXED_ENDPOINTS

    index = CheapestProviderIndex(fetch=fetch)

    first = asyncio.create_task(index.ensure("moonshotai/kimi-k2.6"))
    await asyncio.wait_for(started.wait(), 1)
    second = asyncio.create_task(index.ensure("moonshotai/kimi-k2.6"))
    await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    release.set()
    await asyncio.wait_for(second, 1)

    assert calls == ["moonshotai/kimi-k2.6"]
    assert index.needs_refresh("moonshotai/kimi-k2.6") is False
    assert index._inflight == {}


@pytest.mark.asyncio
async def test_stalled_pricing_degrades_without_blocking_the_refresh():
    started = asyncio.Event()

    async def fetch(base):
        started.set()
        await asyncio.Event().wait()

    index = CheapestProviderIndex(fetch=fetch, fetch_timeout_s=0.05)

    await asyncio.wait_for(index.ensure("moonshotai/kimi-k2.6"), 5)

    assert index.needs_refresh("moonshotai/kimi-k2.6") is False
    assert index.cheapest_provider_block("moonshotai/kimi-k2.6") is None
    assert index._inflight == {}


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

    assert "provider" not in body.get("extra_body", {})


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
