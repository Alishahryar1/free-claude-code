"""Chat client requests keep their native contract at each destination."""

from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.chat_completions.request import (
    ChatConversionError,
    ChatRequest,
    chat_messages_body,
    chat_responses_body,
)
from free_claude_code.core.gemini.native_request import chat_to_gemini


def test_chat_preserves_native_fields_and_nulls():
    body = {
        "model": "model",
        "messages": [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "hello"},
        ],
        "stream": True,
        "temperature": None,
        "logprobs": True,
        "metadata": {"tag": "native"},
        "provider_option": {"enabled": True},
    }
    assert ChatRequest.model_validate(body).body() == body


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_chat_tool_history_and_thinking_survive_conversion(destination):
    body = {
        "model": "model",
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Need the file",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ],
        "tool_choice": "required",
        "max_completion_tokens": 200,
    }
    original = deepcopy(body)
    result = destination(ChatRequest.model_validate(body))
    assert body == original
    assert "Need the file" in str(result)
    assert "contents" in str(result)
    assert "call_1" in str(result)
    assert result["model"] == "model"


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_translated_chat_rejects_unrepresentable_options(destination):
    with pytest.raises(ChatConversionError, match="logprobs"):
        destination(
            ChatRequest(
                model="model",
                messages=[{"role": "user", "content": "hi"}],
                logprobs=True,
            )
        )


def test_rich_tool_output_uses_responses_input_part_types():
    request = ChatRequest(
        model="model",
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call",
                "content": [
                    {"type": "text", "text": "result"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ],
    )
    output = cast(dict[str, Any], chat_responses_body(request))["input"][0]["output"]
    assert output == [
        {"type": "input_text", "text": "result"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
    ]


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_cross_protocol_stream_options_cannot_be_silently_discarded(destination):
    request = ChatRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        stream_options={"include_obfuscation": False},
    )
    with pytest.raises(ChatConversionError, match="stream_options"):
        destination(request)


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_nullable_chat_assistant_fields_are_absent_in_translation(destination):
    request = ChatRequest(
        model="model",
        messages=[
            {
                "role": "assistant",
                "content": "answer",
                "tool_calls": None,
                "function_call": None,
                "refusal": None,
                "audio": None,
            },
            {"role": "user", "content": "next"},
        ],
    )
    assert destination(request)


@pytest.mark.parametrize(
    "destination", [chat_messages_body, chat_responses_body, chat_to_gemini]
)
def test_allowed_tools_rejects_unsupported_translation_clearly(destination):
    request = ChatRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        tool_choice={
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "function", "function": {"name": "read"}}],
            },
        },
    )
    with pytest.raises(ChatConversionError, match="tool_choice"):
        destination(request)


@pytest.mark.parametrize("destination", [chat_messages_body, chat_to_gemini])
def test_refusal_history_keeps_its_text_across_other_protocols(destination):
    request = ChatRequest(
        model="model",
        messages=[
            {"role": "assistant", "content": None, "refusal": "I cannot do that."},
            {"role": "user", "content": "A different task"},
        ],
    )
    result = cast(dict[str, Any], destination(request))
    if destination is chat_messages_body:
        assert result["messages"][0]["content"] == [
            {"type": "text", "text": "I cannot do that."}
        ]
    else:
        assert result["contents"][0]["parts"] == [{"text": "I cannot do that."}]
