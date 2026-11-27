"""Request startup retains the conversion it validates before generation."""

from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import (
    RoutedMessagesRequest,
    RoutedResponsesRequest,
)
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.application.test_execution import _routed_request, _routed_responses_request
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_openai_chat_transport import _success
from tests.providers.test_openai_codex_provider import (
    OpenAICodexProvider,
    _complete_stream,
    _config,
    _FakeAuth,
)
from tests.providers.test_opencode import (
    _catalog_payload,
    _provider_with_wire_transports,
)


def _stream(
    executor: ProviderExecutor,
    routed: RoutedMessagesRequest | RoutedResponsesRequest,
    request_id: str,
):
    if isinstance(routed, RoutedMessagesRequest):
        return executor.stream_messages(
            routed, raw_log_payload=dict, request_id=request_id
        )
    return executor.stream_responses(
        routed, raw_log_payload=dict, request_id=request_id
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_executor_converts_chat_request_once(wire: str) -> None:
    generation_calls = []

    def reply(request: httpx2.Request):
        generation_calls.append(request)
        return _success()

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        provider = OpenAIChatProvider(
            make_provider_config(
                api_key="test", base_url="https://provider.invalid/v1"
            ),
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.REASONING_CONTENT),
                NO_REASONING,
            ),
            admission=immediate_admission(),
            client=client,
        )

        def resolve_provider(_: str, _api_key_number: int):
            return provider

        executor = ProviderExecutor(
            AsyncMock(side_effect=resolve_provider),
        )
        builder_name = (
            "_build_request_body"
            if wire == "messages"
            else "_build_responses_request_body"
        )
        routed = (
            _routed_request() if wire == "messages" else _routed_responses_request()
        )
        with patch.object(
            provider._chat,
            builder_name,
            wraps=provider._chat._build_request_body
            if wire == "messages"
            else provider._chat._build_responses_request_body,
        ) as build:
            stream = _stream(executor, routed, "conversion-once")
            output = "".join(
                [frame async for event in stream for frame in event.frames]
            )
        assert "hello" in output
        assert len(generation_calls) == 1
        assert build.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("egress", ["chat", "responses"])
@pytest.mark.parametrize("warm", [False, True])
async def test_executor_converts_selected_opencode_route_once(
    wire: str, egress: str, warm: bool
) -> None:
    provider, generation, catalog = _provider_with_wire_transports(_catalog_payload())
    transport = provider._chat if egress == "chat" else provider._responses
    builders = {
        ("chat", "messages"): "_build_request_body",
        ("chat", "responses"): "_build_responses_request_body",
        ("responses", "messages"): "_build_messages_body",
        ("responses", "responses"): "_build_native_body",
    }
    name = builders[egress, wire]
    implementations = {
        ("chat", "messages"): provider._chat._build_request_body,
        ("chat", "responses"): provider._chat._build_responses_request_body,
        ("responses", "messages"): provider._responses._build_messages_body,
        ("responses", "responses"): provider._responses._build_native_body,
    }
    update = {"model": f"{egress}-selector"}
    if wire == "messages":
        messages_route = _routed_request()
        routed = replace(
            messages_route, request=messages_route.request.model_copy(update=update)
        )
    else:
        responses_route = _routed_responses_request()
        routed = replace(
            responses_route, request=responses_route.request.model_copy(update=update)
        )
    try:
        if warm:
            await provider.list_model_infos()
        with patch.object(
            transport, name, wraps=implementations[egress, wire]
        ) as build:

            def resolve_provider(_: str, _api_key_number: int):
                return provider

            executor = ProviderExecutor(
                AsyncMock(side_effect=resolve_provider),
            )
            stream = _stream(executor, routed, "opencode-conversion-once")
            assert f"{egress}-ok" in "".join(
                [frame async for event in stream for frame in event.frames]
            )
        assert build.call_count == 1
        assert len(generation) == len(catalog) == 1
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_executor_converts_subscription_request_once(wire: str) -> None:
    generation = []

    def reply(request: httpx2.Request):
        generation.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_complete_stream("hello"),
        )

    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(),
        auth=auth,
        admission=immediate_admission(),
        transport=httpx2.MockTransport(reply),
    )
    name = "_build_messages_body" if wire == "messages" else "_build_native_body"
    routed = _routed_request() if wire == "messages" else _routed_responses_request()
    try:
        with patch.object(
            provider._responses,
            name,
            wraps=provider._responses._build_messages_body
            if wire == "messages"
            else provider._responses._build_native_body,
        ) as build:

            def resolve_provider(_: str, _api_key_number: int):
                return provider

            executor = ProviderExecutor(
                AsyncMock(side_effect=resolve_provider),
            )
            stream = _stream(executor, routed, "subscription-conversion-once")
            assert auth.access_calls == 0
            assert "hello" in "".join(
                [frame async for event in stream for frame in event.frames]
            )
        assert build.call_count == len(generation) == auth.access_calls == 1
    finally:
        await provider.cleanup()
