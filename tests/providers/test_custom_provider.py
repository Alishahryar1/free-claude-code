import asyncio
import json

import httpx
import httpx2
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.runtime.config import build_custom_provider_config

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "thinking",
    [
        None,
        {"type": "adaptive", "display": "omitted"},
        {"type": "enabled", "budget_tokens": 1536},
    ],
)
async def test_native_default_preserves_controls_without_advertised_support(thinking):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=completion("anthropic_messages"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="anthropic_messages"), messages_handler=reply
    )
    request = MessagesRequest(
        model="m",
        max_tokens=4096,
        messages=[{"role": "user", "content": "Hello"}],
        thinking=thinking,
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
    )
    before = request.model_dump()
    try:
        assert [
            item
            async for item in instance.stream_messages(
                request, reasoning=ReasoningPolicy.off()
            )
        ]
        assert bodies[0].get("thinking") == thinking
        assert bodies[0]["output_config"] == request.output_config
        assert request.model_dump() == before
    finally:
        await instance.cleanup()


@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize(
    "limit,cap,expected",
    [(2048, 4096, 2048), (8192, 4096, 4096), (None, 4096, 4096), (8192, None, 8192)],
)
async def test_messages_egress_obeys_request_model_limit(ingress, limit, cap, expected):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=completion("anthropic_messages"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="anthropic_messages"), messages_handler=reply
    )
    info = ProviderModelInfo("m", max_output_tokens=cap) if cap else None
    try:
        if ingress == "messages":
            request = MessagesRequest(
                model="m",
                max_tokens=limit,
                messages=[{"role": "user", "content": "Hello"}],
            )
            stream = instance.stream_messages(request, model_info=info)
        else:
            request = OpenAIResponsesRequest(
                model="m",
                max_output_tokens=limit,
                input="Hello",
                reasoning={"effort": "high"},
            )
            stream = instance.stream_responses(request, model_info=info)
        before = request.model_dump()
        assert [item async for item in stream]
        assert bodies[0]["max_tokens"] == expected
        assert "thinking" not in bodies[0]
        assert "effort" not in bodies[0].get("output_config", {})
        assert request.model_dump() == before
    finally:
        await instance.cleanup()


@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("cap", [1024, 4096])
async def test_manual_preset_respects_cap_before_resolving_budget(ingress, exact, cap):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=completion("anthropic_messages"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="anthropic_messages", reasoning_format="messages_manual"),
        messages_handler=reply,
    )
    policy = (
        ReasoningPolicy.on(budget_tokens=4096)
        if exact
        else ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    )
    info = ProviderModelInfo("m", max_output_tokens=cap)

    async def consume():
        if ingress == "messages":
            stream = instance.stream_messages(
                MessagesRequest(
                    model="m",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": "Hello"}],
                ),
                reasoning=policy,
                model_info=info,
            )
        else:
            stream = instance.stream_responses(
                OpenAIResponsesRequest(
                    model="m", max_output_tokens=8192, input="Hello"
                ),
                reasoning=policy,
                model_info=info,
            )
        return [item async for item in stream]

    try:
        if exact or cap == 1024:
            with pytest.raises(InvalidRequestError, match="below max_tokens"):
                await consume()
            assert not bodies
        else:
            assert await consume()
            assert bodies[0]["max_tokens"] == cap
            assert 1024 <= bodies[0]["thinking"]["budget_tokens"] < cap
    finally:
        await instance.cleanup()


@pytest.mark.parametrize(
    "thinking,error",
    [
        ({"type": "enabled"}, "budget"),
        ({"type": "enabled", "budget_tokens": 6000}, "below max_tokens"),
    ],
)
async def test_native_default_does_not_invent_or_shrink_exact_budget(thinking, error):
    bodies = []
    instance = provider(
        definition(api_format="anthropic_messages"),
        messages_handler=lambda request: bodies.append(request),
    )
    try:
        with pytest.raises(InvalidRequestError, match=error):
            stream = instance.stream_messages(
                MessagesRequest(
                    model="m",
                    max_tokens=8192,
                    thinking=thinking,
                    messages=[{"role": "user", "content": "Hello"}],
                ),
                model_info=ProviderModelInfo("m", max_output_tokens=4096),
            )
            [item async for item in stream]
        assert not bodies
    finally:
        await instance.cleanup()


async def test_concurrent_models_keep_independent_limits():
    bodies = []
    both_started = asyncio.Event()

    async def reply(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return httpx.Response(
            200,
            text=completion("anthropic_messages"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="anthropic_messages"), messages_handler=reply
    )

    async def call(model, cap):
        stream = instance.stream_messages(
            MessagesRequest(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": "Hello"}],
            ),
            model_info=ProviderModelInfo(model, max_output_tokens=cap),
        )
        return [item async for item in stream]

    try:
        await asyncio.gather(call("small", 2048), call("large", 6000))
        assert {body["model"]: body["max_tokens"] for body in bodies} == {
            "small": 2048,
            "large": 6000,
        }
    finally:
        await instance.cleanup()


def definition(**changes):
    return CustomProviderDefinition.model_validate(
        {
            "provider_id": "custom_" + "a" * 32,
            "display_name": "Gateway",
            "base_url": "https://custom.example/team/api",
            **changes,
        }
    )


def provider(config, *, openai_handler=None, messages_handler=None):
    settings = Settings(provider_rate_limit=1000)
    return CustomProvider(
        build_custom_provider_config(config, settings),
        definition=config,
        admission=ProviderAdmissionController(
            provider_name=config.provider_id,
            rate_limit=1000,
            rate_window=1,
            max_concurrency=2,
        ),
        openai_transport=httpx2.MockTransport(openai_handler)
        if openai_handler
        else None,
        messages_transport=httpx.MockTransport(messages_handler)
        if messages_handler
        else None,
    )


def event(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def completion(api_format):
    if api_format == "openai_chat":
        data = {
            "id": "chat",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": "stop"}
            ],
        }
        return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"
    if api_format == "openai_responses":
        message = {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
        }
        return (
            event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**message, "content": []},
                },
            )
            + event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Hello",
                },
            )
            + event(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp",
                        "object": "response",
                        "created_at": 0,
                        "status": "completed",
                        "model": "m",
                        "output": [message],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                },
            )
        )
    return "".join(
        [
            event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "model": "m",
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ),
            event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
            ),
            event("message_stop", {"type": "message_stop"}),
        ]
    )


@pytest.mark.parametrize(
    "api_format,path",
    [
        ("openai_chat", "chat/completions"),
        ("openai_responses", "responses"),
        ("anthropic_messages", "messages"),
    ],
)
@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize("key", [None, "custom-key"])
async def test_six_paths_use_exact_endpoint_and_credentials(
    api_format, path, ingress, key, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-project")
    requests = []

    def reply(request):
        requests.append(request)
        cls = httpx.Response if api_format == "anthropic_messages" else httpx2.Response
        return cls(
            200,
            text=completion(api_format),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format=api_format, api_key=key),
        openai_handler=reply if api_format != "anthropic_messages" else None,
        messages_handler=reply if api_format == "anthropic_messages" else None,
    )
    try:
        policy = ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM)
        if ingress == "messages":
            request = MessagesRequest(
                model="m",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
            )
            stream = instance.stream_messages(request, reasoning=policy)
        else:
            request = OpenAIResponsesRequest(
                model="m", input="Hello", reasoning={"effort": "medium"}
            )
            stream = instance.stream_responses(request, reasoning=policy)
        output = "".join([chunk async for chunk in stream])
        assert "Hello" in output
        assert len(requests) == 1
        sent = requests[0]
        assert str(sent.url) == f"https://custom.example/team/api/{path}"
        assert sent.headers.get("x-api-key") == (
            key if api_format == "anthropic_messages" else None
        )
        assert sent.headers.get("authorization") == (
            f"Bearer {key}" if key and api_format != "anthropic_messages" else None
        )
        assert "openai-organization" not in sent.headers
        assert "openai-project" not in sent.headers
        body = json.loads(sent.content)
        if ingress == "responses" and api_format == "openai_responses":
            assert body["reasoning"]["effort"] == "medium"
        else:
            assert "reasoning_effort" not in body
            assert not body.get("reasoning")
            assert "thinking" not in body
        if ingress == "responses":
            assert isinstance(request, OpenAIResponsesRequest)
            assert request.reasoning == {"effort": "medium"}
    finally:
        await instance.cleanup()


@pytest.mark.parametrize(
    "api_format", ["openai_chat", "openai_responses", "anthropic_messages"]
)
async def test_discovery_is_bound_and_manual_inventory_needs_no_network(
    api_format, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send")
    requests = []

    def reply(request):
        requests.append(request)
        cls = httpx.Response if api_format == "anthropic_messages" else httpx2.Response
        return cls(200, json={"data": [{"id": "org/model"}], "has_more": False})

    instance = provider(
        definition(api_format=api_format),
        openai_handler=reply if api_format != "anthropic_messages" else None,
        messages_handler=reply if api_format == "anthropic_messages" else None,
    )
    try:
        assert {info.model_id for info in await instance.list_model_infos()} == {
            "org/model"
        }
        assert requests[0].url.path == "/team/api/models"
        assert "authorization" not in requests[0].headers
        assert "x-api-key" not in requests[0].headers
    finally:
        await instance.cleanup()
    requests.clear()
    manual = provider(
        definition(api_format=api_format, model_ids=["manual"]),
        openai_handler=reply if api_format != "anthropic_messages" else None,
        messages_handler=reply if api_format == "anthropic_messages" else None,
    )
    try:
        assert {info.model_id for info in await manual.list_model_infos()} == {"manual"}
        assert not requests
    finally:
        await manual.cleanup()


@pytest.mark.parametrize(
    "api_format,mode,field,expected",
    [
        ("openai_chat", "openai_effort", "reasoning_effort", "high"),
        ("openai_chat", "limited_effort", "reasoning_effort", "high"),
        ("openai_chat", "reasoning_object", "reasoning", {"effort": "high"}),
        ("openai_chat", "thinking", "thinking", {"type": "enabled"}),
        (
            "openai_chat",
            "chat_template",
            "chat_template_kwargs",
            {"enable_thinking": True},
        ),
        (
            "openai_responses",
            "native_responses",
            "reasoning",
            {"effort": "high", "summary": "auto"},
        ),
        (
            "anthropic_messages",
            "messages_manual",
            "thinking",
            {"type": "enabled", "budget_tokens": 2048},
        ),
        ("anthropic_messages", "messages_adaptive", "thinking", {"type": "adaptive"}),
    ],
)
async def test_reasoning_presets_encode_controls_on_the_wire(
    api_format, mode, field, expected
):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        cls = httpx.Response if api_format == "anthropic_messages" else httpx2.Response
        return cls(
            200,
            text=completion(api_format),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format=api_format, reasoning_format=mode),
        openai_handler=reply if api_format != "anthropic_messages" else None,
        messages_handler=reply if api_format == "anthropic_messages" else None,
    )
    try:
        request = MessagesRequest(
            model="m", max_tokens=4096, messages=[{"role": "user", "content": "Hello"}]
        )
        assert [
            chunk
            async for chunk in instance.stream_messages(
                request, reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
            )
        ]
        assert bodies[0][field] == expected
    finally:
        await instance.cleanup()


@pytest.mark.parametrize(
    "api_format", ["openai_chat", "openai_responses", "anthropic_messages"]
)
@pytest.mark.parametrize("ingress", ["messages", "responses"])
async def test_tool_continuation_keeps_tool_results_and_images(api_format, ingress):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        cls = httpx.Response if api_format == "anthropic_messages" else httpx2.Response
        return cls(
            200,
            text=completion(api_format),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format=api_format),
        openai_handler=reply if api_format != "anthropic_messages" else None,
        messages_handler=reply if api_format == "anthropic_messages" else None,
    )
    try:
        if ingress == "messages":
            request = MessagesRequest(
                model="m",
                max_tokens=100,
                tools=[
                    {
                        "name": "Read",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_read",
                                "name": "Read",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_read",
                                "content": [
                                    {"type": "text", "text": "tool-result-token"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "aGVsbG8=",
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ],
            )
            chunks = instance.stream_messages(request)
        else:
            request = OpenAIResponsesRequest(
                model="m",
                tools=[
                    {
                        "type": "function",
                        "name": "Read",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                input=[
                    {
                        "type": "function_call",
                        "call_id": "call_read",
                        "name": "Read",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_read",
                        "output": [
                            {"type": "input_text", "text": "tool-result-token"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,aGVsbG8=",
                            },
                        ],
                    },
                ],
            )
            chunks = instance.stream_responses(request)
        assert [chunk async for chunk in chunks]
        body = json.dumps(bodies[0])
        assert (
            "tool-result-token" in body and "aGVsbG8=" in body and "call_read" in body
        )
    finally:
        await instance.cleanup()


async def test_messages_discovery_rejects_repeating_cursor_without_partial_results():
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(
            200, json={"data": [{"id": "same"}], "has_more": True, "last_id": "same"}
        )

    instance = provider(
        definition(api_format="anthropic_messages"), messages_handler=reply
    )
    try:
        with pytest.raises(ValueError, match="cursor did not advance"):
            await instance.list_model_infos()
        assert len(calls) == 2
        assert calls[1].url.params["after_id"] == "same"
    finally:
        await instance.cleanup()


@pytest.mark.parametrize("vision", [True, False])
async def test_messages_discovery_retains_text_and_advertised_limits(vision):
    instance = provider(
        definition(api_format="anthropic_messages"),
        messages_handler=lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model",
                        "max_input_tokens": 200000,
                        "max_tokens": 32000,
                        "capabilities": {"image_input": {"supported": vision}},
                    }
                ],
                "has_more": False,
            },
        ),
    )
    try:
        (info,) = await instance.list_model_infos()
        expected = {ModelInputModality.TEXT}
        if vision:
            expected.add(ModelInputModality.IMAGE)
        assert info.input_modalities == frozenset(expected)
        assert info.context_window_tokens == 200000
        assert info.max_output_tokens == 32000
    finally:
        await instance.cleanup()
