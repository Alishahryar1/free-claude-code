"""Tests for catalog-driven Command Code Messages / Chat Completions routing."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from unittest.mock import patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.providers.commandcode import (
    CommandCodeProvider,
    create_commandcode_provider,
)
from free_claude_code.providers.commandcode.catalog import (
    CommandCodeCatalog,
    CommandCodeTransport,
    parse_command_code_catalog,
)
from free_claude_code.providers.model_listing import ModelListResponseError
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_anthropic_messages_transport import Wire, _events, _sse
from tests.wire_payload import decode_object

BASE_URL = "https://api.commandcode.ai/provider/v1"


def _config():
    return make_provider_config(api_key="test_commandcode_key", base_url=BASE_URL)


def _payload(*models: dict[str, object]) -> dict[str, object]:
    return {
        "object": "list",
        "data": list(models)
        or [
            {
                "id": "claude-sonnet-5-5",
                "context_length": 1_000_000,
                "supported_endpoints": ["/messages"],
            },
            {
                "id": "deepseek/deepseek-v4-flash",
                "context_length": 256_000,
                "supported_endpoints": ["/chat/completions", "/responses"],
            },
        ],
    }


def _client(
    handler: Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request(model: str) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
        }
    )


def test_parse_routes_messages_and_chat_and_keeps_context_window() -> None:
    snapshot = parse_command_code_catalog(_payload(), provider_name="COMMANDCODE")

    claude = snapshot.route("claude-sonnet-5-5")
    open_model = snapshot.route("deepseek/deepseek-v4-flash")
    assert claude is not None
    assert open_model is not None
    assert claude.transport is CommandCodeTransport.MESSAGES
    assert open_model.transport is CommandCodeTransport.CHAT_COMPLETIONS
    assert claude.model_info.context_window_tokens == 1_000_000
    assert open_model.model_info.context_window_tokens == 256_000
    assert {info.model_id for info in snapshot.model_infos} == set(snapshot.routes)


def test_parse_prefers_messages_when_both_endpoints_are_advertised() -> None:
    snapshot = parse_command_code_catalog(
        _payload(
            {"id": "both", "supported_endpoints": ["/chat/completions", "/messages"]}
        ),
        provider_name="COMMANDCODE",
    )

    route = snapshot.route("both")
    assert route is not None
    assert route.transport is CommandCodeTransport.MESSAGES


def test_parse_skips_models_without_a_supported_endpoint() -> None:
    snapshot = parse_command_code_catalog(
        _payload(
            {"id": "responses-only", "supported_endpoints": ["/responses"]},
            {"id": "chat", "supported_endpoints": ["/chat/completions"]},
        ),
        provider_name="COMMANDCODE",
    )

    assert set(snapshot.routes) == {"chat"}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": "nope"},
        {"data": ["nope"]},
        {"data": [{"supported_endpoints": ["/messages"]}]},
        {"data": [{"id": "  ", "supported_endpoints": ["/messages"]}]},
        {"data": [{"id": "m"}]},
        {"data": [{"id": "m", "supported_endpoints": "/messages"}]},
        {"data": [{"id": "m", "supported_endpoints": [1]}]},
        {"data": []},
        {"data": [{"id": "m", "supported_endpoints": ["/responses"]}]},
        {
            "data": [
                {"id": "m", "supported_endpoints": ["/messages"]},
                {"id": "m", "supported_endpoints": ["/messages"]},
            ]
        },
    ],
)
def test_parse_rejects_malformed_catalogs(payload: object) -> None:
    with pytest.raises(ModelListResponseError):
        parse_command_code_catalog(payload, provider_name="COMMANDCODE")


@pytest.mark.asyncio
async def test_cold_catalog_load_is_coalesced_and_sends_bearer_key() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200, json=_payload())

    client = _client(handler)
    catalog = CommandCodeCatalog(
        _config(),
        provider_name="COMMANDCODE",
        admission=immediate_admission(provider_name="commandcode"),
        client=client,
    )
    try:
        first, second = await asyncio.gather(catalog.snapshot(), catalog.snapshot())
    finally:
        await client.aclose()

    assert first is second
    assert len(requests) == 1
    assert str(requests[0].url) == f"{BASE_URL}/models"
    assert requests[0].headers["authorization"] == "Bearer test_commandcode_key"


@pytest.mark.asyncio
async def test_refresh_retains_last_good_snapshot_on_malformed_response() -> None:
    payloads: list[object] = [
        _payload({"id": "first", "supported_endpoints": ["/messages"]}),
        {"data": [{"id": ""}]},
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads.pop(0))

    client = _client(handler)
    catalog = CommandCodeCatalog(
        _config(),
        provider_name="COMMANDCODE",
        admission=immediate_admission(provider_name="commandcode"),
        client=client,
    )
    try:
        first = await catalog.refresh()
        with pytest.raises(ModelListResponseError):
            await catalog.refresh()
        cached = await catalog.snapshot()
    finally:
        await client.aclose()

    assert cached is first


@pytest.mark.asyncio
async def test_cold_catalog_failure_is_canonical() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"code": "upgrade_required"}})

    client = _client(handler)
    catalog = CommandCodeCatalog(
        _config(),
        provider_name="COMMANDCODE",
        admission=immediate_admission(provider_name="commandcode", max_attempts=2),
        client=client,
    )
    try:
        with pytest.raises(ExecutionFailure):
            await catalog.snapshot(request_id="req_catalog")
    finally:
        await client.aclose()


async def _drain(stream: AsyncIterator[StreamOutput]) -> str:
    return "".join([frame async for event in stream for frame in event.frames])


@pytest.mark.asyncio
async def test_claude_model_is_sent_to_messages_endpoint_with_api_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_payload())
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*_events("from-claude"))]),
        )

    provider = create_commandcode_provider(
        _config(),
        immediate_admission(provider_name="commandcode"),
        http_client=_client(handler),
    )
    try:
        with patch.object(provider._chat, "stream_messages") as chat:
            text = await _drain(provider.stream_messages(_request("claude-sonnet-5-5")))
    finally:
        await provider.cleanup()

    chat.assert_not_called()
    assert "from-claude" in text
    post = next(request for request in seen if request.method == "POST")
    assert str(post.url) == f"{BASE_URL}/messages"
    assert post.headers["x-api-key"] == "test_commandcode_key"
    assert decode_object(post.content)["model"] == "claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_unknown_model_is_rejected_before_any_generation_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_payload())

    provider = CommandCodeProvider(
        _config(),
        admission=immediate_admission(provider_name="commandcode"),
        http_client=_client(handler),
    )
    try:
        with pytest.raises(InvalidRequestError, match="does not advertise"):
            await _drain(provider.stream_messages(_request("missing-model")))
    finally:
        await provider.cleanup()

    assert [request.method for request in seen] == ["GET"]


@pytest.mark.asyncio
async def test_list_model_infos_uses_the_catalog() -> None:
    provider = CommandCodeProvider(
        _config(),
        admission=immediate_admission(provider_name="commandcode"),
        http_client=_client(lambda _request: httpx.Response(200, json=_payload())),
    )
    try:
        infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert {info.model_id for info in infos} == {
        "claude-sonnet-5-5",
        "deepseek/deepseek-v4-flash",
    }


def _chat_sse(text: str) -> str:
    chunks: tuple[JsonObject, ...] = (
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def _chat_request(model: str) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
        }
    )


def _gemini_request(model: str) -> GeminiRequest:
    return GeminiRequest.model_validate(
        {"model": model, "contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    )


def _responses_request(model: str) -> OpenAIResponsesRequest:
    return OpenAIResponsesRequest.model_validate(
        {"model": model, "input": "hello", "max_output_tokens": 64}
    )


_Request = MessagesRequest | ChatRequest | GeminiRequest | OpenAIResponsesRequest
_METHODS = ("stream_messages", "stream_chat", "stream_gemini", "stream_responses")


def _call(
    provider: CommandCodeProvider, method: str, model: str
) -> AsyncIterator[StreamOutput]:
    if method == "stream_chat":
        return provider.stream_chat(_chat_request(model))
    if method == "stream_gemini":
        return provider.stream_gemini(_gemini_request(model))
    if method == "stream_responses":
        return provider.stream_responses(_responses_request(model))
    return provider.stream_messages(_request(model))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize(
    ("model", "messages_transport"),
    [("claude-sonnet-5-5", True), ("deepseek/deepseek-v4-flash", False)],
    ids=["messages-transport", "chat-transport"],
)
async def test_every_ingress_reaches_only_the_transport_that_serves_the_model(
    method: str, model: str, messages_transport: bool
) -> None:
    provider = CommandCodeProvider(
        _config(),
        admission=immediate_admission(provider_name="commandcode"),
        http_client=_client(lambda _request: httpx.Response(200, json=_payload())),
    )
    seen: list[str] = []

    def spy(request: _Request, **_kwargs: object) -> AsyncIterator[StreamOutput]:
        seen.append(request.model)

        async def empty() -> AsyncIterator[StreamOutput]:
            return
            yield

        return empty()

    target = provider._messages if messages_transport else provider._chat
    other = provider._chat if messages_transport else provider._messages
    try:
        with (
            patch.object(target, method, side_effect=spy),
            patch.object(other, method) as untouched,
        ):
            await _drain(_call(provider, method, model))
    finally:
        await provider.cleanup()

    assert seen == [model]
    untouched.assert_not_called()


@pytest.mark.asyncio
async def test_chat_model_sends_authenticated_chat_completions_request() -> None:
    chat_requests: list[httpx2.Request] = []

    def chat_handler(request: httpx2.Request) -> httpx2.Response:
        chat_requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_sse("from-chat"),
        )

    sdk = AsyncOpenAI(
        api_key="test_commandcode_key",
        base_url=BASE_URL,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(chat_handler)),
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", return_value=sdk
    ):
        provider = CommandCodeProvider(
            _config(),
            admission=immediate_admission(provider_name="commandcode"),
            http_client=_client(lambda _request: httpx.Response(200, json=_payload())),
        )
    try:
        text = await _drain(
            provider.stream_messages(_request("deepseek/deepseek-v4-flash"))
        )
    finally:
        await provider.cleanup()

    assert "from-chat" in text
    assert len(chat_requests) == 1
    sent = chat_requests[0]
    assert str(sent.url) == f"{BASE_URL}/chat/completions"
    assert sent.headers["authorization"] == "Bearer test_commandcode_key"
    body = decode_object(sent.content)
    assert body["model"] == "deepseek/deepseek-v4-flash"
    assert body["stream"] is True
