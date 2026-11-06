import json

import pytest

from free_claude_code.core.chat_completions.response import (
    ChatResponseAccumulator,
    NativeChatRelay,
)


def test_native_chat_preserves_choices_unknown_fields_and_literal_thought_tags():
    payload = {
        "id": "completion",
        "created": 1,
        "model": "upstream",
        "object": "chat.completion.chunk",
        "vendor": {"native": True},
        "choices": [
            {
                "index": 0,
                "delta": {"content": "<think>literal</think>", "refusal": None},
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "literal", "logprob": -1.0}]},
            },
            {
                "index": 1,
                "delta": {"content": "other", "audio": {"data": "abc", "id": "audio"}},
                "finish_reason": "stop",
            },
        ],
    }
    relay = NativeChatRelay("public", choices=2)
    result = json.loads(relay.feed(payload).removeprefix("data: "))
    assert result == {**payload, "model": "public"}
    assert payload["model"] == "upstream"
    assert relay.finish() == ["data: [DONE]\n\n"]


def test_chat_json_keeps_late_usage_and_both_choices():
    accumulator = ChatResponseAccumulator()
    accumulator.feed(
        {
            "id": "id",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"},
                {"index": 1, "delta": {"content": "world"}, "finish_reason": "stop"},
            ],
        }
    )
    accumulator.feed({"id": "id", "choices": [], "usage": {"total_tokens": 12}})
    accumulator.finish()
    response = accumulator.result()
    assert response["usage"] == {"total_tokens": 12}
    assert [choice["message"]["content"] for choice in response["choices"]] == [
        "hello",
        "world",
    ]


def test_native_gemini_json_preserves_metadata_and_late_usage():
    from free_claude_code.core.gemini.response import GeminiResponseAccumulator

    accumulator = GeminiResponseAccumulator()
    accumulator.feed(
        {
            "responseId": "id",
            "native_extension": {"kept": True},
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ],
        }
    )
    accumulator.feed({"usageMetadata": {"totalTokenCount": 12}})
    assert accumulator.result()["native_extension"] == {"kept": True}
    assert accumulator.result()["usageMetadata"] == {"totalTokenCount": 12}


def test_chat_logprobs_accumulate_after_null_and_survive_trailing_null():
    accumulator = ChatResponseAccumulator()
    token = {"token": "hello", "logprob": -0.1, "bytes": None, "top_logprobs": []}
    for content in (None, [token], None):
        accumulator.feed(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": {"content": content},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
    accumulator.finish()
    assert accumulator.result()["choices"][0]["logprobs"]["content"] == [token]


def test_native_legacy_function_arguments_accumulate_without_replacement():
    accumulator = ChatResponseAccumulator()
    for call in [{"name": "read", "arguments": '{"pa'}, {"arguments": 'th":"kept"}'}]:
        accumulator.feed(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"function_call": call},
                        "finish_reason": None,
                    }
                ]
            }
        )
    accumulator.feed(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]}
    )
    accumulator.finish()
    assert accumulator.result()["choices"][0]["message"]["function_call"] == {
        "name": "read",
        "arguments": '{"path":"kept"}',
    }


@pytest.mark.parametrize(
    "details",
    [
        [
            {"type": "reasoning.summary", "id": "first", "index": 0, "summary": "one"},
            {"type": "reasoning.summary", "id": "second", "index": 0, "summary": "two"},
        ],
        [
            {"type": "reasoning.encrypted", "id": "same", "index": 0, "data": "one"},
            {"type": "reasoning.encrypted", "id": "same", "index": 0, "data": "two"},
        ],
        [
            {
                "type": "reasoning.text",
                "id": "same",
                "text": "one",
                "signature": "first",
            },
            {
                "type": "reasoning.text",
                "id": "same",
                "text": "two",
                "signature": "second",
            },
        ],
    ],
)
def test_distinct_native_reasoning_records_keep_their_boundaries(details):
    accumulator = ChatResponseAccumulator()
    for detail in details:
        accumulator.feed(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_details": [detail]},
                        "finish_reason": None,
                    }
                ]
            }
        )
    accumulator.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    accumulator.finish()
    assert accumulator.result()["choices"][0]["message"]["reasoning_details"] == details
