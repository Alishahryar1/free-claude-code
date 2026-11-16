import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.providers.open_router import OpenRouterProvider
from tests.api.support import create_test_app
from tests.providers.support import (
    SDKStreamDouble,
    immediate_admission,
    make_provider_config,
    sdk_chat_chunk,
)
from tests.providers.test_open_router import image_routing_error_body, routing_error

IMAGE_DATA = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6nGQAAAAASUVORK5CYII="
IMAGE_URL = f"data:image/png;base64,{IMAGE_DATA}"
NOTE = (
    "[image omitted: this model cannot accept image input. "
    "Tell the user the image was not seen.]"
)
WIRES = [("messages", False), ("messages", True), ("responses", True)]


def image_payload(wire, stream):
    payload = {"model": "open_router/test-model", "stream": stream}
    if wire == "messages":
        content = [
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
        payload.update(messages=[{"role": "user", "content": content}], max_tokens=100)
    else:
        content = [
            {"type": "input_text", "text": "Read this image"},
            {"type": "input_image", "image_url": IMAGE_URL},
        ]
        payload.update(
            input=[{"role": "user", "content": content}], max_output_tokens=100
        )
    return payload


def sent_parts(call):
    return [
        part
        for message in call.kwargs["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
    ]


async def answered():
    for content, finish_reason in (("I cannot see the image.", None), (None, "stop")):
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content, tool_calls=None),
                        finish_reason=finish_reason,
                    )
                ],
                usage=None,
            )
        )


def post(provider, create, wire, payload):
    try:
        with (
            patch(
                "free_claude_code.api.routes.resolve_provider", return_value=provider
            ),
            patch.object(provider._client.chat.completions, "create", create),
            TestClient(create_test_app()) as client,
        ):
            return client.post(f"/v1/{wire}", json=payload)
    finally:
        asyncio.run(provider.cleanup())


def openrouter():
    return OpenRouterProvider(
        make_provider_config("test-key", "https://openrouter.ai/api/v1"),
        admission=immediate_admission(),
    )


@pytest.mark.parametrize(("wire", "stream"), WIRES)
def test_openrouter_image_rejection_is_retried_with_a_note(wire, stream):
    create = AsyncMock(
        side_effect=[
            routing_error(image_routing_error_body()),
            SDKStreamDouble(answered()),
        ]
    )

    response = post(openrouter(), create, wire, image_payload(wire, stream))

    assert response.status_code == 200, response.text
    assert "I cannot see the image." in response.text
    first, retried = (sent_parts(call) for call in create.await_args_list)
    image = {"type": "image_url", "image_url": {"url": IMAGE_URL}}
    assert image in first and image not in retried
    assert {"type": "text", "text": NOTE} in retried


@pytest.mark.parametrize(("wire", "stream"), WIRES)
def test_openrouter_unrelated_404_keeps_the_image(wire, stream):
    body = {"code": 404, "message": "Model not found"}
    create = AsyncMock(side_effect=routing_error(body))

    response = post(openrouter(), create, wire, image_payload(wire, stream))

    assert response.status_code == 404
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == "api_error"
    assert body["message"] in error["message"]
    assert response.headers["request-id"] in error["message"]
    assert create.await_count == 1
    assert {"type": "image_url", "image_url": {"url": IMAGE_URL}} in sent_parts(
        create.await_args_list[0]
    )
