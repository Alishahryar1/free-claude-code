import json

import pytest

from free_claude_code.core.chat_completions.request import ChatRequest
from tests.providers.test_history_transports import _harness


@pytest.mark.asyncio
async def test_direct_chat_preserves_native_request_and_stream_through_sdk():
    request = {
        "model": "native",
        "messages": [
            {"role": "developer", "content": "rules", "name": "instructions"},
            {
                "role": "assistant",
                "content": "prior",
                "reasoning": "native reasoning",
                "vendor_message": 17,
            },
            {"role": "user", "content": "go"},
        ],
        "n": 2,
        "temperature": None,
        "logprobs": True,
        "vendor_option": {"keep": True},
        "extra_body": {
            "model": "literal-extension",
            "google": {"thinking_config": {"include_thoughts": True}},
        },
        "extra_headers": {"x-native-extension": "body-value"},
        "stream": True,
    }
    events = [
        {
            "id": "native-id",
            "object": "chat.completion.chunk",
            "created": 17,
            "model": "native",
            "vendor_reply": 42,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>literal</think>"},
                    "finish_reason": "stop",
                    "logprobs": {"content": []},
                },
                {"index": 1, "delta": {"content": "other"}, "finish_reason": "stop"},
            ],
        }
    ]
    async with _harness("chat", lambda bodies: (200, events)) as (_, bodies, provider):
        chunks = [
            event
            async for event in provider.stream_chat(
                ChatRequest.model_validate(request), response_model="public"
            )
        ]
        assert bodies == [request]
        assert json.loads(chunks[0].removeprefix("data: ")) == {
            **events[0],
            "model": "public",
        }
        assert chunks[-1] == "data: [DONE]\n\n"
