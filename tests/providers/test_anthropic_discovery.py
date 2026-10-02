"""Discovery snapshots remain authoritative for each compatibility request."""

import asyncio
import json

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.anthropic.models import model_record
from free_claude_code.providers.model_listing import ModelListResponseError
from tests.providers.support import stream_messages, stream_responses
from tests.providers.test_anthropic_messages_transport import _events, _sse
from tests.providers.test_anthropic_provider import provider


def metadata(model, adaptive, *, max_tokens=8192):
    return {
        "id": model,
        "max_tokens": max_tokens,
        "max_input_tokens": 200000,
        "capabilities": {
            "thinking": {
                "supported": True,
                "types": {
                    "adaptive": {"supported": adaptive},
                    "enabled": {"supported": not adaptive},
                },
            },
            "effort": {
                "supported": adaptive,
                "low": {"supported": adaptive},
                "high": {"supported": adaptive},
                "max": {"supported": False},
            },
            "image_input": {"supported": True},
        },
    }


def test_metadata_uses_advertised_types_and_efforts_without_name_inference():
    adaptive = model_record(metadata("manual-sounding-name", True))
    manual = model_record(metadata("adaptive-sounding-name", False))
    unknown = model_record({"id": "claude-future"})
    assert adaptive.messages.adaptive_thinking == "required"
    assert adaptive.messages.supported_efforts == ("low", "high")
    assert adaptive.info.reasoning_capability is not ReasoningCapability.REQUIRED
    assert adaptive.info.context_window_tokens == 200000
    assert adaptive.messages.supports_vision is True
    assert manual.messages.adaptive_thinking == "unsupported"
    assert unknown.messages.adaptive_thinking is None
    assert unknown.info.supports_thinking is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_page",
    [
        {"data": [], "has_more": True, "last_id": "old"},
        {"data": [{"id": "old"}], "has_more": True, "last_id": "old"},
        {"data": [], "has_more": "false"},
        {"data": [{"id": ""}], "has_more": False},
        {"has_more": False},
    ],
)
async def test_malformed_refresh_never_publishes_a_partial_catalog(bad_page):
    refresh = False
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer upstream-secret"
        assert request.headers["anthropic-workspace-id"] == "wrkspc_test"
        assert request.headers["anthropic-version"] == "2023-06-01"
        if not refresh:
            return httpx.Response(
                200, json={"data": [metadata("old", False)], "has_more": False}
            )
        return httpx.Response(200, json=bad_page)

    p = provider(handle)
    try:
        await p.list_model_infos()
        before = await p.model_record("old")
        refresh = True
        with pytest.raises(ModelListResponseError):
            await p.list_model_infos()
        assert await p.model_record("old") is before
        assert len(calls) <= 3
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_valid_empty_catalog_is_published_without_fabricated_models():
    p = provider(
        lambda request: httpx.Response(200, json={"data": [], "has_more": False})
    )
    try:
        assert await p.list_model_infos() == frozenset()
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_concurrent_alias_lookup_is_cached_and_uses_same_credentials():
    calls = []

    async def handle(request):
        calls.append(request)
        await asyncio.sleep(0)
        assert request.url.path == "/v1/models/latest-alias"
        assert request.headers["authorization"] == "Bearer upstream-secret"
        assert request.headers["anthropic-workspace-id"] == "wrkspc_test"
        return httpx.Response(200, json=metadata("resolved-model", True))

    p = provider(handle)
    try:
        first, second = await asyncio.gather(
            p.model_record("latest-alias"), p.model_record("latest-alias")
        )
        assert first is second
        assert first.info.model_id == "resolved-model"
        assert len(calls) == 1
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_concurrent_models_use_independent_conversion_capabilities():
    sent = []

    async def handle(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [metadata("manual", False), metadata("adaptive", True)],
                    "has_more": False,
                },
            )
        sent.append(json.loads(request.content))
        await asyncio.sleep(0)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(*_events())
        )

    p = provider(handle)
    try:
        await p.list_model_infos()

        async def messages():
            return [
                event
                async for event in stream_messages(
                    p,
                    MessagesRequest(
                        model="manual",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=16000,
                    ),
                    reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
                )
            ]

        async def responses():
            return [
                event
                async for event in stream_responses(
                    p,
                    OpenAIResponsesRequest(model="adaptive", input="hello"),
                    reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
                )
            ]

        message_output, response_output = await asyncio.gather(messages(), responses())
        assert "message_stop" in "".join(message_output)
        assert "response.completed" in "".join(response_output)
        by_model = {body["model"]: body for body in sent}
        assert by_model["manual"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }
        assert by_model["manual"]["max_tokens"] == 8192
        assert "output_config" not in by_model["manual"]
        assert by_model["adaptive"]["thinking"] == {"type": "adaptive"}
        assert by_model["adaptive"]["output_config"] == {"effort": "high"}
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_refresh_does_not_mutate_an_inflight_model_record():
    posted = asyncio.Event()
    release = asyncio.Event()
    refresh = False
    sent = []

    async def handle(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        metadata("same", refresh, max_tokens=4096 if refresh else 8192)
                    ],
                    "has_more": False,
                },
            )
        sent.append(json.loads(request.content))
        posted.set()
        await release.wait()
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(*_events())
        )

    p = provider(handle)

    async def run():
        return [
            event
            async for event in stream_messages(
                p,
                MessagesRequest(
                    model="same", messages=[{"role": "user", "content": "hi"}]
                ),
                reasoning=ReasoningPolicy.on(),
            )
        ]

    try:
        await p.list_model_infos()
        before = await p.model_record("same")
        task = asyncio.create_task(run())
        await asyncio.wait_for(posted.wait(), 1)
        refresh = True
        await p.list_model_infos()
        after = await p.model_record("same")
        assert before.messages.adaptive_thinking == "unsupported"
        assert after.messages.adaptive_thinking == "required"
        release.set()
        await task
        await run()
        assert sent[0]["thinking"]["type"] == "enabled"
        assert sent[0]["max_tokens"] == 8192
        assert sent[1]["thinking"]["type"] == "adaptive"
        assert sent[1]["max_tokens"] == 4096
    finally:
        release.set()
        await p.cleanup()


@pytest.mark.asyncio
async def test_fresh_owner_does_not_trust_stale_generic_capabilities():
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        return httpx.Response(404, json={"error": {"type": "not_found_error"}})

    p = provider(handle)
    try:
        with pytest.raises(InvalidRequestError, match="thinking mode"):
            _ = [
                event
                async for event in stream_messages(
                    p,
                    MessagesRequest(
                        model="alias", messages=[{"role": "user", "content": "hi"}]
                    ),
                    model_info=ProviderModelInfo(
                        "alias", supports_thinking=True, max_output_tokens=8000
                    ),
                    reasoning=ReasoningPolicy.on(),
                )
            ]
        assert len(calls) == 1
    finally:
        await p.cleanup()
