"""Cooldown errors use the recovery episode after provider cleanup."""

import asyncio

import httpx
import httpx2
import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.core.diagnostics import ERROR_DETAIL_DISPLAY_CAP_BYTES
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.runtime.config import build_custom_provider_config
from tests.api.support import create_test_app
from tests.api.test_four_protocol_matrix import ingress_request
from tests.api.test_retry_after_cooldown import cooldown_clock as cooldown_clock
from tests.providers.test_anthropic_provider import native_body
from tests.providers.test_anthropic_provider import provider as anthropic_provider
from tests.providers.test_custom_provider import definition
from tests.providers.test_custom_provider import provider as custom_provider


class DelayedResponseClose(httpx.AsyncByteStream):
    def __init__(self, closing: asyncio.Event, release: asyncio.Event) -> None:
        self.closing = closing
        self.release = release

    async def __aiter__(self):
        # The bounded error reader leaves the response open for the attempt scope.
        yield b"x" * (ERROR_DETAIL_DISPLAY_CAP_BYTES + 1)

    async def aclose(self) -> None:
        self.closing.set()
        await self.release.wait()


def assert_cooldown(response: httpx.Response, detail: str) -> None:
    assert response.status_code == 429, response.text
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert "3600 seconds" in error["message"], error["message"].split(
        "Upstream error:"
    )[0]
    assert detail in error["message"]


@pytest.mark.asyncio
async def test_active_request_reports_extension_after_response_close(cooldown_clock):
    calls = 0
    first_entered = asyncio.Event()
    both_entered = asyncio.Event()
    first_closing = asyncio.Event()
    release_first = asyncio.Event()
    detail = "longer upstream cooldown"

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await both_entered.wait()
            return httpx.Response(
                429,
                headers={"retry-after": "120"},
                stream=DelayedResponseClose(first_closing, release_first),
            )
        both_entered.set()
        await first_closing.wait()
        return httpx.Response(
            429,
            headers={"retry-after": "3600"},
            json={"error": {"type": "rate_limit_error", "message": detail}},
        )

    app = create_test_app(
        Settings(MODEL="anthropic/primary"),
        providers={"anthropic": anthropic_provider(upstream, max_attempts=10)},
    )
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client,
        asyncio.timeout(3),
        asyncio.TaskGroup() as tasks,
    ):
        first = tasks.create_task(client.post("/v1/messages", json=native_body(False)))
        await first_entered.wait()
        try:
            second = await client.post("/v1/messages", json=native_body(False))
            fresh = await client.post("/v1/messages", json=native_body(False))
            assert not first.done()
        finally:
            release_first.set()

    assert calls == 2
    for response in (second, fresh, first.result()):
        assert_cooldown(response, detail)


@pytest.mark.parametrize("translated", [False, True])
@pytest.mark.asyncio
async def test_cooldown_during_retry_cleanup_reaches_wire(cooldown_clock, translated):
    calls = 0
    leader_entered = asyncio.Event()
    follower_entered = asyncio.Event()
    all_entered = asyncio.Event()
    leader_closing = asyncio.Event()
    follower_closing = asyncio.Event()
    release = asyncio.Event()
    detail = "shared upstream cooldown"

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            leader_entered.set()
            await all_entered.wait()
            return httpx.Response(
                503,
                headers={"retry-after": "2"},
                stream=DelayedResponseClose(leader_closing, release),
            )
        if calls == 2:
            follower_entered.set()
            await leader_closing.wait()
            return httpx.Response(
                503,
                headers={"retry-after": "2"},
                stream=DelayedResponseClose(follower_closing, release),
            )
        all_entered.set()
        await follower_closing.wait()
        return httpx.Response(
            429,
            headers={"retry-after": "3600"},
            json={"error": {"type": "rate_limit_error", "message": detail}},
        )

    controller = ProviderAdmissionController(provider_name="TEST", jitter=0)
    if translated:
        configured = definition(
            api_format="anthropic_messages", api_keys=["upstream-key"]
        )
        provider_id = configured.provider_id
        settings = Settings(
            MODEL=f"{provider_id}/primary", custom_providers=(configured,)
        )
        instance = CustomProvider(
            build_custom_provider_config(configured, settings),
            definition=configured,
            admission=controller,
            messages_transport=httpx.MockTransport(upstream),
        )
        path, payload = ingress_request("responses", True)
    else:
        provider_id = "anthropic"
        settings = Settings(MODEL="anthropic/primary")
        instance = anthropic_provider(upstream)
        instance._admission = controller
        path, payload = "/v1/messages", native_body(False)
    app = create_test_app(settings, providers={provider_id: instance})
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client,
        asyncio.timeout(3),
        asyncio.TaskGroup() as tasks,
    ):
        leader = tasks.create_task(client.post(path, json=payload))
        await leader_entered.wait()
        follower = tasks.create_task(client.post(path, json=payload))
        await follower_entered.wait()
        try:
            cooldown = await client.post(path, json=payload)
            assert not leader.done() and not follower.done()
        finally:
            release.set()

    assert calls == 3
    for response in (cooldown, leader.result(), follower.result()):
        assert_cooldown(response, detail)


@pytest.mark.parametrize("api_format", ["openai_chat", "openai_responses"])
@pytest.mark.asyncio
async def test_owned_client_close_precedes_cooldown_snapshot(
    api_format, cooldown_clock, monkeypatch
):
    first_entered = asyncio.Event()
    both_entered = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    original_close = OpenAIRequestClient.aclose
    calls = 0

    async def gated_close(self):
        if not closing.is_set():
            closing.set()
            await release.wait()
        await original_close(self)

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await both_entered.wait()
            retry_after = "120"
        else:
            both_entered.set()
            await closing.wait()
            retry_after = "3600"
        return httpx2.Response(
            429,
            headers={"retry-after": retry_after},
            json={"error": {"message": f"upstream cooldown {retry_after}"}},
        )

    monkeypatch.setattr(OpenAIRequestClient, "aclose", gated_close)
    configured = definition(api_format=api_format, api_keys=["upstream-key"])
    provider_id = configured.provider_id
    settings = Settings(MODEL=f"{provider_id}/primary", custom_providers=(configured,))
    app = create_test_app(
        settings,
        providers={provider_id: custom_provider(configured, openai_handler=upstream)},
    )
    wire = "messages" if api_format == "openai_chat" else "responses"
    path, payload = ingress_request(wire, wire == "responses")
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client,
        asyncio.timeout(3),
        asyncio.TaskGroup() as tasks,
    ):
        first = tasks.create_task(client.post(path, json=payload))
        await first_entered.wait()
        try:
            second = await client.post(path, json=payload)
            assert not first.done()
        finally:
            release.set()

    assert calls == 2
    for response in (second, first.result()):
        assert_cooldown(response, "upstream cooldown 3600")
