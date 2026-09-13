"""Claude Auto mode exercises the real handler/provider request boundary."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError, AsyncOpenAI, BadRequestError

from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.reasoning import ReasoningCapability
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.providers.support import immediate_admission, make_provider_config


class ClassifierStream:
    def __init__(self, *, starved: bool = False, error: Exception | None = None):
        self.starved = starved
        self.error = error
        self.closed = False

    async def __aiter__(self):
        if self.error is not None:
            raise self.error
        for text, reasoning, finish in (
            (None, "Internal provider analysis", None),
            (None if self.starved else "<severity>0</severity>", None, None),
            (None, None, "length" if self.starved else "stop"),
        ):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=text,
                            reasoning=reasoning,
                            reasoning_content=reasoning,
                            tool_calls=None,
                        ),
                        finish_reason=finish,
                    )
                ],
                usage=None,
            )

    async def aclose(self):
        self.closed = True


def classifier_request():
    return MessagesRequest.model_validate(
        {
            "model": "open_router/dynamic-route",
            "system": "Output <severity>N</severity> for this command's safety.",
            "messages": [
                {
                    "role": "user",
                    "content": "<transcript>Read the requested file.</transcript>",
                }
            ],
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "stop_sequences": ["</severity>"],
            "stream": False,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "corrects"),
    [
        ({"code": 400, "message": "Reasoning cannot be disabled."}, True),
        (
            {
                "code": 400,
                "message": "Reasoning cannot be disabled.",
                "metadata": {"error_type": "invalid_request"},
            },
            True,
        ),
        ({"code": 400, "message": "Messages contain an invalid role."}, False),
        (
            {"code": 400, "param": "tools", "message": "Reasoning is required."},
            False,
        ),
        ({"code": 403, "message": "Reasoning cannot be disabled."}, False),
    ],
    ids=["numeric", "metadata", "unrelated", "other-field", "permission"],
)
async def test_openrouter_numeric_sse_rejection_uses_classifier_correction(
    error, corrects
):
    bodies = []

    def upstream(request):
        bodies.append(json.loads(request.content))
        chunk = {
            "id": "gen-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "dynamic-route",
            "provider": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<severity>0</severity>"},
                    "finish_reason": "stop",
                }
            ],
        }
        if len(bodies) == 1:
            chunk["error"] = error
            chunk["choices"] = [
                {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
            ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=AsyncOpenAI(
            api_key="test",
            base_url="https://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
        ),
    ):
        provider = OpenRouterProvider(
            make_provider_config("test", "https://provider.invalid/v1"),
            admission=immediate_admission(),
        )
    try:
        response = await MessagesHandler(
            Settings(), provider_resolver=lambda _: provider
        ).create(classifier_request())
        assert isinstance(response, JSONResponse)
        assert len(bodies) == (2 if corrects else 1)
        if corrects:
            assert response.status_code == 200
            assert json.loads(bytes(response.body))["content"] == [
                {"type": "text", "text": "<severity>0</severity>"}
            ]
            assert bodies[0]["reasoning"] == {"enabled": False}
            assert "reasoning" not in bodies[1]
            assert bodies[1]["max_tokens"] == 81920
        else:
            assert response.status_code >= 400
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "reject_off",
    [False, True, "stream"],
    ids=["off-ignored", "off-rejected", "off-stream-rejected"],
)
async def test_classifier_mandatory_reasoning_still_returns_verdict(reject_off, stream):
    provider = OpenRouterProvider(
        make_provider_config(api_key="test", base_url="https://openrouter.test/v1"),
        admission=immediate_admission(),
    )
    bodies = []
    streams = []

    async def upstream(**body):
        bodies.append(body)
        if (
            reject_off
            and body.get("extra_body", {}).get("reasoning", {}).get("enabled") is False
        ):
            error = {
                "error": {
                    "code": "invalid_request_error",
                    "message": "Reasoning is mandatory for this endpoint and cannot be disabled.",
                }
            }
            if reject_off == "stream":
                rejected = ClassifierStream(
                    error=APIError(
                        error["error"]["message"],
                        request=httpx.Request(
                            "POST", "https://openrouter.test/v1/chat/completions"
                        ),
                        body=error["error"],
                    )
                )
                streams.append(rejected)
                return rejected
            raise BadRequestError(
                error["error"]["message"],
                response=httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST", "https://openrouter.test/v1/chat/completions"
                    ),
                    json=error,
                ),
                body=error,
            )
        result = ClassifierStream(starved=body.get("max_tokens", 0) < 2048)
        streams.append(result)
        return result

    try:
        handler = MessagesHandler(Settings(), provider_resolver=lambda _: provider)
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=upstream),
        ):
            response = await handler.create(
                classifier_request().model_copy(update={"stream": stream})
            )
            assert isinstance(response, (JSONResponse, StreamingResponse))
            assert response.status_code == 200
            if stream:
                assert isinstance(response, StreamingResponse)
                raw = "".join([str(chunk) async for chunk in response.body_iterator])
                events = parse_sse_text(raw)
                assert text_content(events) == "<severity>0</severity>"
                assert thinking_content(events) == ""
                assert events[-1].event == "message_stop"
            else:
                assert isinstance(response, JSONResponse)
                message = json.loads(bytes(response.body))
                assert message["content"] == [
                    {"type": "text", "text": "<severity>0</severity>"}
                ]
                assert message["stop_reason"] == "end_turn"
        assert bodies[0]["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
        assert len(bodies) == (2 if reject_off else 1)
        assert all(stream.closed for stream in streams)
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability", "off", "limit"),
    [
        ("none", False, 64),
        ("optional", True, 64),
        ("required", False, 81920),
        ("unknown", True, 81920),
    ],
)
async def test_handler_uses_cached_capabilities_for_the_selected_provider(
    capability, off, limit
):
    provider = OpenRouterProvider(
        make_provider_config("test", "https://provider.invalid/v1"),
        admission=immediate_admission(),
    )
    bodies = []

    async def upstream(**body):
        bodies.append(body)
        return ClassifierStream()

    try:
        handler = MessagesHandler(
            Settings(),
            provider_resolver=lambda _: provider,
            model_infos=(
                ProviderModelInfo(
                    "open_router/dynamic-route",
                    reasoning_capability=ReasoningCapability(capability),
                ),
            ),
        )
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=upstream),
        ):
            response = await handler.create(classifier_request())
        assert isinstance(response, JSONResponse) and response.status_code == 200
        assert json.loads(bytes(response.body))["content"] == [
            {"type": "text", "text": "<severity>0</severity>"}
        ]
        assert (
            bodies[0].get("extra_body", {}).get("reasoning", {}).get("enabled") is False
        ) is off
        assert bodies[0]["max_tokens"] == limit
    finally:
        await provider.cleanup()
