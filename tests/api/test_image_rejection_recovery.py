"""Image-bearing requests to models whose upstream rejects image input."""

import asyncio
import json
from collections.abc import Callable
from unittest.mock import patch

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.provider_catalog import NVIDIA_NIM_DEFAULT_BASE
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from tests.api.support import create_test_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.providers.support import immediate_admission, make_provider_config
from tests.wire_payload import (
    array_payload,
    decode_object,
    object_array,
    object_payload,
)

IMAGE_DATA = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6nGQAAAAASUVORK5CYII="
IMAGE_URL = f"data:image/png;base64,{IMAGE_DATA}"
WIRES = ("messages", "chat", "responses", "gemini")

# Hosted NIM reports the rejection inside an HTTP 200 stream; a vLLM server
# reached directly answers with HTTP 400.
VLLM_MULTIMODAL_REJECTION = (
    "ValueError: Received multimodal data but multimodal processing is not "
    "enabled. Use --enable-multimodal flag to enable multimodal processing."
)


def vllm_rejection(*, in_stream: bool) -> httpx2.Response:
    error = {
        "message": VLLM_MULTIMODAL_REJECTION,
        "type": "invalid_request_error",
        "code": 400,
    }
    if in_stream:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n",
        )
    return httpx2.Response(400, json={"error": error})


def chat_reply(text: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(
            f"data: {json.dumps(event)}\n\n" for event in text_events("chat", text)
        )
        + "data: [DONE]\n\n",
    )


def image_request(wire: str) -> tuple[str, dict[str, object]]:
    if wire == "gemini":
        return "/v1beta/models/public:generateContent", {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Read this image"},
                        {"inlineData": {"mimeType": "image/png", "data": IMAGE_DATA}},
                    ],
                }
            ]
        }
    if wire == "messages":
        content: list[dict[str, object]] = [
            {"type": "text", "text": "Read this image"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": IMAGE_DATA,
                },
            },
        ]
        return "/v1/messages", {
            "model": "public",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": content}],
        }
    if wire == "chat":
        return "/v1/chat/completions", {
            "model": "public",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this image"},
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    ],
                }
            ],
        }
    return "/v1/responses", {
        "model": "public",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Read this image"},
                    {"type": "input_image", "image_url": IMAGE_URL},
                ],
            }
        ],
    }


def sent_parts(request: httpx2.Request) -> list[JsonObject]:
    body = decode_object(request.content)
    return [
        part
        for message in array_payload(body["messages"])
        if isinstance(object_payload(message).get("content"), list)
        for part in object_array(object_payload(message)["content"])
    ]


def nim_app(reply: Callable[[httpx2.Request], httpx2.Response]):
    client = AsyncOpenAI(
        api_key="upstream-key",
        base_url=NVIDIA_NIM_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=client,
    ):
        provider = NvidiaNimProvider(
            make_provider_config(
                api_key="upstream-key", base_url=NVIDIA_NIM_DEFAULT_BASE
            ),
            admission=immediate_admission(max_attempts=3),
        )
    settings = Settings(MODEL="nvidia_nim/text-only")
    return provider, create_test_app(settings, providers={"nvidia_nim": provider})


NOTE = (
    "[image omitted: this model cannot accept image input. "
    "Tell the user the image was not seen.]"
)


def image_sent(request: httpx2.Request) -> bool:
    return {"type": "image_url", "image_url": {"url": IMAGE_URL}} in sent_parts(request)


def note_sent(request: httpx2.Request) -> bool:
    return {"type": "text", "text": NOTE} in sent_parts(request)


def text_only_upstream(requests: list[httpx2.Request], *, in_stream: bool):
    def reply(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if image_sent(request):
            return vllm_rejection(in_stream=in_stream)
        return chat_reply("I cannot see the image.")

    return reply


@pytest.mark.parametrize("in_stream", [True, False])
@pytest.mark.parametrize("wire", WIRES)
def test_rejected_image_is_replaced_by_a_note_and_remembered(
    wire: str, in_stream: bool
):
    requests: list[httpx2.Request] = []
    provider, app = nim_app(text_only_upstream(requests, in_stream=in_stream))
    path, payload = image_request(wire)
    try:
        with TestClient(app) as client:
            first = client.post(path, json=payload)
            second = client.post(path, json=payload)
    finally:
        asyncio.run(provider.cleanup())

    for response in (first, second):
        assert response.status_code == 200, response.text
        assert "I cannot see the image." in response.text
    # The rejected request is retried with the note; the next request for the
    # same model is sent with the note from the start.
    assert [image_sent(request) for request in requests] == [True, False, False]
    assert [note_sent(request) for request in requests] == [False, True, True]


def test_a_model_that_accepts_images_still_receives_them():
    requests: list[httpx2.Request] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if decode_object(request.content)["model"] == "text-only" and image_sent(
            request
        ):
            return vllm_rejection(in_stream=True)
        return chat_reply("Done.")

    provider, app = nim_app(reply)
    path, payload = image_request("messages")
    try:
        with TestClient(app) as client:
            assert client.post(path, json=payload).status_code == 200
            vision = {**payload, "model": "nvidia_nim/vision"}
            assert client.post(path, json=vision).status_code == 200
    finally:
        asyncio.run(provider.cleanup())

    assert [decode_object(request.content)["model"] for request in requests] == [
        "text-only",
        "text-only",
        "vision",
    ]
    assert [image_sent(request) for request in requests] == [True, False, True]


@pytest.mark.parametrize("in_stream", [True, False])
def test_rejection_of_a_request_without_media_is_not_retried(in_stream: bool):
    requests: list[httpx2.Request] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return vllm_rejection(in_stream=in_stream)

    provider, app = nim_app(reply)
    path, payload = image_request("messages")
    payload["messages"] = [{"role": "user", "content": "hello"}]
    try:
        with TestClient(app) as client:
            response = client.post(path, json=payload)
    finally:
        asyncio.run(provider.cleanup())

    assert response.status_code == 400, response.text
    assert "multimodal processing is not enabled" in response.text
    assert len(requests) == 1
