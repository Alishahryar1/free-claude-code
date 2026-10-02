"""Alibaba Cloud discovery and configuration contracts."""

import json
from contextlib import asynccontextmanager

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.runtime import build_provider_config
from tests.providers.support import (
    immediate_admission,
    stream_messages,
    stream_responses,
)

BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def model(model_id="qwen-coder", **metadata):
    return {
        "model": model_id,
        "features": ["function-calling"],
        "capabilities": ["TG", "Reasoning"],
        "inference_metadata": {"request_modality": ["Text", "Image"]},
        "model_info": {"context_window": 131072, "max_output_tokens": 16384},
        **metadata,
    }


def page(models, *, number=1, total=None):
    return {
        "success": True,
        "output": {
            "page_no": number,
            "page_size": 100,
            "total": len(models) if total is None else total,
            "models": models,
        },
    }


@asynccontextmanager
async def provider_for(handler, base_url=BASE_URL):
    from free_claude_code.providers.alibaba_cloud import AlibabaCloudProvider

    settings = Settings(
        ALIBABA_CLOUD_API_KEY="test-key", ALIBABA_CLOUD_BASE_URL=base_url
    )
    config = build_provider_config(PROVIDER_CATALOG["alibaba_cloud"], settings)
    async with AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        provider = AlibabaCloudProvider(
            config,
            admission=immediate_admission(provider_name="alibaba_cloud"),
            client=client,
        )
        try:
            yield provider
        finally:
            await provider.cleanup()


def test_configuration_defaults_and_region_override():
    descriptor = PROVIDER_CATALOG["alibaba_cloud"]
    settings = Settings(ALIBABA_CLOUD_API_KEY=" key ")
    config = build_provider_config(descriptor, settings)
    assert config.api_key == "key"
    assert config.base_url == BASE_URL
    assert descriptor.configuration_attrs() == ("alibaba_cloud_api_key",)

    settings = Settings(
        ALIBABA_CLOUD_API_KEY="regional-key",
        ALIBABA_CLOUD_BASE_URL="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        ALIBABA_CLOUD_PROXY="http://proxy.test:8080",
    )
    config = build_provider_config(descriptor, settings)
    assert config.api_key == "regional-key"
    assert config.base_url == settings.alibaba_cloud_base_url
    assert config.proxy == "http://proxy.test:8080"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [BASE_URL, "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/"],
)
async def test_discovery_fetches_all_pages_with_same_region_and_auth(endpoint):
    requests = []

    def handler(request):
        requests.append(request)
        number = int(request.url.params["page_no"])
        return httpx2.Response(
            200, json=page([model(f"model-{number}")], number=number, total=2)
        )

    async with provider_for(handler, endpoint) as provider:
        infos = await provider.list_model_infos()

    assert {info.model_id for info in infos} == {"model-1", "model-2"}
    assert len(requests) == 2
    for request in requests:
        assert request.url.host == httpx2.URL(endpoint).host
        assert request.url.path == "/api/v1/models"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.url.params["features"] == "function-calling"
        assert request.url.params["supports"] == "inference"
    for info in infos:
        assert info.supports_thinking is True
        assert info.context_window_tokens == 131072
        assert info.max_output_tokens == 16384
        assert info.input_modalities == frozenset(
            {ModelInputModality.TEXT, ModelInputModality.IMAGE}
        )


@pytest.mark.asyncio
async def test_discovery_filters_transport_after_counting_every_page():
    requests = []
    pages = [
        [
            model("realtime", capabilities=["Realtime-Omni"]),
            model("voice", capabilities=["TG", "Realtime-Chatting"]),
            model("audio", capabilities=["ASR"]),
            model("unknown", capabilities=[]),
        ],
        [
            model("chat", capabilities=["TG"]),
            model("reasoner", capabilities=["Reasoning"]),
            model("vision", capabilities=["VU"]),
            model("omni", capabilities=["Multimodal-Omni"]),
        ],
    ]

    def handler(request):
        number = int(request.url.params["page_no"])
        requests.append(number)
        return httpx2.Response(
            200, json=page(pages[number - 1], number=number, total=8)
        )

    async with provider_for(handler) as provider:
        infos = await provider.list_model_infos()

    assert requests == [1, 2]
    assert {info.model_id for info in infos} == {"chat", "reasoner", "vision", "omni"}


@pytest.mark.asyncio
@pytest.mark.parametrize("capabilities", [None, [], ["New-API"], "TG", [1]])
async def test_discovery_does_not_assume_unknown_capabilities_support_chat(
    capabilities,
):
    payload = page([model(), model("unknown", capabilities=capabilities)])
    async with provider_for(lambda _: httpx2.Response(200, json=payload)) as provider:
        infos = await provider.list_model_infos()
    assert {info.model_id for info in infos} == {"qwen-coder"}


@pytest.mark.asyncio
@pytest.mark.parametrize("same_page", [False, True])
@pytest.mark.parametrize("capabilities", [["TG"], ["Realtime-Omni"]])
async def test_discovery_rejects_duplicate_ids_before_publishing_catalog(
    same_page, capabilities
):
    requests = []
    repeated = model("repeated", capabilities=capabilities)
    pages = (
        [[model("chat"), repeated, repeated]]
        if same_page
        else [[model("chat"), repeated], [repeated]]
    )

    def handler(request):
        number = int(request.url.params["page_no"])
        requests.append(number)
        return httpx2.Response(
            200, json=page(pages[number - 1], number=number, total=3)
        )

    async with provider_for(handler) as provider:
        with pytest.raises(ModelListResponseError, match="duplicate"):
            await provider.list_model_infos()
    assert requests == ([1] if same_page else [1, 2])


@pytest.mark.asyncio
async def test_discovery_filters_non_tool_models_without_guessing_missing_limits():
    payload = page(
        [model(), model("embedding", features=[]), model("unknown", model_info=None)]
    )
    async with provider_for(lambda _: httpx2.Response(200, json=payload)) as provider:
        infos = {info.model_id: info for info in await provider.list_model_infos()}
    assert set(infos) == {"qwen-coder", "unknown"}
    assert infos["unknown"].context_window_tokens is None
    assert infos["unknown"].max_output_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "code": "InvalidApiKey"},
        {"success": True},
        page([]),
        page([], total=2),
        page([model()], number=2),
        page([model()], total=-1),
        page([model()], total=True),
        page([model()], total="1"),
        page([{"features": ["function-calling"]}]),
    ],
)
async def test_discovery_rejects_malformed_or_empty_catalog(payload):
    async with provider_for(lambda _: httpx2.Response(200, json=payload)) as provider:
        with pytest.raises(ModelListResponseError):
            await provider.list_model_infos()


@pytest.mark.asyncio
async def test_later_page_failure_does_not_publish_partial_catalog():
    def handler(request):
        number = int(request.url.params["page_no"])
        payload = (
            page([model()], total=2) if number == 1 else page([], number=2, total=2)
        )
        return httpx2.Response(200, json=payload)

    async with provider_for(handler) as provider:
        with pytest.raises(ModelListResponseError):
            await provider.list_model_infos()


@pytest.mark.asyncio
async def test_discovery_retries_only_the_failed_page():
    pages = []

    def handler(request):
        number = int(request.url.params["page_no"])
        pages.append(number)
        if pages == [1, 2]:
            return httpx2.Response(503, json={"error": {"message": "Unavailable"}})
        return httpx2.Response(
            200, json=page([model(f"model-{number}")], number=number, total=2)
        )

    async with provider_for(handler) as provider:
        infos = await provider.list_model_infos()
    assert pages == [1, 2, 2]
    assert {info.model_id for info in infos} == {"model-1", "model-2"}


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_chat_stream_preserves_reasoning_tools_and_usage(wire):
    requests = []
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Inspecting."}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "inspect",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"README.md"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
        },
    ]

    def handler(request):
        requests.append(request)
        frames = [
            {
                "id": "chat_1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen-coder",
                **chunk,
            }
            for chunk in chunks
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
            + "data: [DONE]\n\n",
        )

    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    async with provider_for(handler) as provider:
        stream = (
            stream_messages(
                provider,
                MessagesRequest(
                    model="qwen-coder",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": "Inspect the README."}],
                    tools=[{"name": "inspect", "input_schema": schema}],
                ),
            )
            if wire == "messages"
            else stream_responses(
                provider,
                OpenAIResponsesRequest(
                    model="qwen-coder",
                    input="Inspect the README.",
                    max_output_tokens=2048,
                    tools=[
                        {"type": "function", "name": "inspect", "parameters": schema}
                    ],
                ),
            )
        )
        output = "".join([event async for event in stream])

    assert len(requests) == 1
    assert requests[0].url.path == "/compatible-mode/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == "qwen-coder"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 2048
    assert body["tools"][0]["function"]["parameters"] == schema
    assert "enable_thinking" not in body
    assert "Inspecting." in output
    assert "inspect" in output
    assert "README.md" in output
    events = [
        json.loads(line[6:])
        for line in output.splitlines()
        if line.startswith("data: {")
    ]
    if wire == "messages":
        terminal = next(event for event in events if event["type"] == "message_delta")
        assert terminal["delta"]["stop_reason"] == "tool_use"
        assert terminal["usage"]["output_tokens"] == 10
    else:
        terminal = next(
            event for event in events if event["type"] == "response.completed"
        )
        assert terminal["response"]["usage"]["input_tokens"] == 25
        assert terminal["response"]["usage"]["output_tokens"] == 10
