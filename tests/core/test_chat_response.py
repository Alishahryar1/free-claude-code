import json

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
