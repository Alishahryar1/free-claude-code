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
            frame
            async for event in provider.stream_chat(
                ChatRequest.model_validate(request), response_model="public"
            )
            for frame in event.frames
        ]
        assert bodies == [request]
        assert json.loads(chunks[0].removeprefix("data: ")) == {
            **events[0],
            "model": "public",
        }
        assert chunks[-1] == "data: [DONE]\n\n"


def _chunk(model, choices, **extra):
    return {
        "id": "native-id",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        **extra,
        "choices": choices,
    }


async def _native_finishes(events, max_attempts=1):
    from unittest.mock import patch

    from free_claude_code.providers.openai_chat.transport import _ChatAttempt

    request = ChatRequest.model_validate(
        {
            "model": "native",
            "messages": [{"role": "user", "content": "go"}],
            "stream": True,
        }
    )
    finishes = []
    reported = []
    original = _ChatAttempt.completed_fields

    def spy(self):
        fields = original(self)
        reported.append(fields["finish_reason"])
        return fields

    async with _harness("chat", lambda _: (200, events), max_attempts=max_attempts) as (
        _,
        _,
        provider,
    ):
        with patch.object(_ChatAttempt, "completed_fields", spy):
            async for event in provider.stream_chat(request, response_model="public"):
                for frame in event.frames:
                    if not frame.startswith("data: ") or "[DONE]" in frame:
                        continue
                    data = json.loads(frame.removeprefix("data: "))
                    finishes.extend(
                        choice.get("finish_reason")
                        for choice in data.get("choices", [])
                        if choice.get("finish_reason") is not None
                    )
    return finishes, reported


@pytest.mark.asyncio
@pytest.mark.parametrize("with_trailer", [False, True])
async def test_native_chat_empty_tool_calls_finish_is_relayed_as_stop(with_trailer):
    events = [
        _chunk(
            "native",
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "done"},
                    "finish_reason": None,
                }
            ],
        ),
        _chunk("native", [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    ]
    if with_trailer:
        events.append(
            _chunk(
                "native",
                [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )
        )
    finishes, reported = await _native_finishes(events)
    assert finishes == ["stop"]
    assert reported == ["stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_trailer", [False, True])
async def test_native_chat_real_tool_call_keeps_tool_calls_finish(with_trailer):
    events = [
        _chunk(
            "native",
            [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        ),
        _chunk("native", [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    ]
    if with_trailer:
        events.append(
            _chunk(
                "native",
                [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )
        )
    finishes, reported = await _native_finishes(events)
    assert finishes == ["tool_calls"]
    assert reported == ["tool_calls"]
