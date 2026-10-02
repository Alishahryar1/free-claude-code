import json

import httpx
import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.anthropic import AnthropicProvider
from tests.providers.support import immediate_admission, make_provider_config


def provider(handler, **kwargs):
    return AnthropicProvider(
        make_provider_config("upstream-secret", "https://api.anthropic.com/v1"),
        workspace_id="wrkspc_test",
        admission=immediate_admission(max_attempts=2),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def native_body(stream=False):
    return {
        "model": "claude-test",
        "stream": stream,
        "max_tokens": 12345,
        "thinking": {"type": "adaptive", "future": "keep"},
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "future_tool_result", "opaque": [1, 2]}],
            }
        ],
        "tools": [{"type": "web_search_20260318", "name": "web_search", "max_uses": 1}],
        "future_option": {"keep": True},
    }


@pytest.mark.asyncio
async def test_native_json_preserves_body_and_owns_auth():
    body = native_body()
    received = []

    def handle(request):
        received.append(request)
        return httpx.Response(
            200,
            json={
                "type": "message",
                "model": "claude-test",
                "content": [{"type": "future_tool_result", "opaque": "kept"}],
                "usage": {
                    "output_tokens": 4,
                    "server_tool_use": {"web_search_requests": 1},
                },
            },
        )

    p = provider(handle)
    try:
        result = [
            x
            async for x in p.stream_native_messages(
                NativeMessagesRequest(body),
                request_id="r",
                response_model="public",
                request_headers={
                    "Authorization": "Bearer client",
                    "Cookie": "secret",
                    "anthropic-workspace-id": "foreign",
                    "anthropic-beta": "test-beta,test-beta",
                },
            )
        ]
        message = json.loads("".join(result))
        assert message["model"] == "public"
        assert message["content"][0]["opaque"] == "kept"
        assert message["usage"]["server_tool_use"]["web_search_requests"] == 1
        assert json.loads(received[0].content) == body
        assert received[0].headers["authorization"] == "Bearer upstream-secret"
        assert received[0].headers["anthropic-workspace-id"] == "wrkspc_test"
        assert received[0].headers["anthropic-beta"] == "test-beta"
        assert "cookie" not in received[0].headers
        assert "x-api-key" not in received[0].headers
        assert body == native_body()
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_native_sse_relays_unknown_tool_events_without_replay():
    calls = []
    events = [
        {"type": "message_start", "message": {"id": "m", "model": "claude-test"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "s",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"hi"}'},
        },
        {"type": "future_event", "opaque": "keep"},
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ]

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(
                f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
            ),
        )

    p = provider(handle)
    output = []
    try:
        with pytest.raises(ExecutionFailure):
            async for chunk in p.stream_native_messages(
                NativeMessagesRequest(native_body(True)),
                request_id="r",
                response_model="public",
            ):
                assert isinstance(chunk, str)
                output.append(chunk)
        assert len(calls) == 1
        assert '"model": "public"' in output[0]
        assert "input_json_delta" in "".join(output)
        assert "future_event" in "".join(output)
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_model_discovery_is_atomic_and_retains_thinking_types():
    fail = False

    def handle(request):
        if "after_id" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "adaptive",
                            "max_tokens": 8192,
                            "capabilities": {
                                "thinking": {
                                    "supported": True,
                                    "types": {
                                        "adaptive": {"supported": True},
                                        "enabled": {"supported": False},
                                    },
                                }
                            },
                        }
                    ],
                    "has_more": True,
                    "last_id": "adaptive",
                },
            )
        return httpx.Response(
            503 if fail else 200, json={"data": [{"id": "manual"}], "has_more": False}
        )

    p = provider(handle)
    try:
        assert {x.model_id for x in await p.list_model_infos()} == {
            "adaptive",
            "manual",
        }
        before = await p.model_record("adaptive")
        assert before.messages.adaptive_thinking == "required"
        fail = True
        with pytest.raises(httpx.HTTPStatusError):
            await p.list_model_infos()
        assert await p.model_record("adaptive") == before
    finally:
        await p.cleanup()


def test_anthropic_is_configuration_provider_with_native_messages():
    descriptor = PROVIDER_CATALOG["anthropic"]
    assert descriptor.credential_env == "ANTHROPIC_API_KEY"
    assert descriptor.native_messages_passthrough
