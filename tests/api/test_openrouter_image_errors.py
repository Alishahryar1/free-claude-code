import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.providers.open_router import OpenRouterProvider
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_open_router import image_routing_error_body, routing_error


@pytest.mark.parametrize(
    ("wire", "stream"), [("messages", False), ("messages", True), ("responses", True)]
)
@pytest.mark.parametrize("image_rejected", [True, False])
def test_openrouter_image_error_reaches_client_without_losing_image(
    wire, stream, image_rejected
):
    provider = OpenRouterProvider(
        make_provider_config("test-key", "https://openrouter.ai/api/v1"),
        admission=immediate_admission(),
    )
    body = (
        image_routing_error_body()
        if image_rejected
        else {"code": 404, "message": "Model not found"}
    )
    create = AsyncMock(side_effect=routing_error(body))
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6nGQAAAAASUVORK5CYII="
    url = f"data:image/png;base64,{image_data}"
    payload = {"model": "open_router/test-model", "stream": stream}
    if wire == "messages":
        content = [
            {"type": "text", "text": "Read this image"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
        ]
        payload.update(messages=[{"role": "user", "content": content}], max_tokens=100)
    else:
        content = [
            {"type": "input_text", "text": "Read this image"},
            {"type": "input_image", "image_url": url},
        ]
        payload.update(
            input=[{"role": "user", "content": content}], max_output_tokens=100
        )
    try:
        with (
            patch(
                "free_claude_code.api.routes.resolve_provider", return_value=provider
            ),
            patch.object(provider._client.chat.completions, "create", create),
            TestClient(create_test_app()) as client,
        ):
            response = client.post(f"/v1/{wire}", json=payload)
        assert response.status_code == (400 if image_rejected else 404)
        assert response.headers["x-should-retry"] == "false"
        error = response.json()["error"]
        assert error["type"] == (
            "invalid_request_error" if image_rejected else "api_error"
        )
        assert body["message"] in error["message"]
        assert response.headers["request-id"] in error["message"]
        if image_rejected:
            assert "Remove the image" in error["message"]
        assert create.await_count == 1
        assert any(
            part.get("image_url", {}).get("url") == url
            for message in create.await_args_list[0].kwargs["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
        )
    finally:
        asyncio.run(provider.cleanup())
