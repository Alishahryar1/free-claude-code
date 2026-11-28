"""Chat client requests keep their native contract at each destination."""

from collections.abc import Callable
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
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject
from tests.wire_payload import object_array


def test_chat_preserves_native_fields_and_nulls() -> None:
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
def test_chat_tool_history_and_thinking_survive_conversion(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
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
def test_translated_chat_rejects_unrepresentable_options(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    with pytest.raises(ChatConversionError, match="logprobs"):
        destination(
            ChatRequest(
                model="model",
                messages=[{"role": "user", "content": "hi"}],
                logprobs=True,
            )
        )


def test_rich_tool_output_uses_responses_input_part_types() -> None:
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
    body: JsonObject = chat_responses_body(request)
    output = object_array(body["input"])[0]["output"]
    assert output == [
        {"type": "input_text", "text": "result"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
    ]


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_cross_protocol_stream_options_cannot_be_silently_discarded(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    request = ChatRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        stream_options={"include_obfuscation": False},
    )
    with pytest.raises(ChatConversionError, match="stream_options"):
        destination(request)


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_nullable_chat_assistant_fields_are_absent_in_translation(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
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
def test_allowed_tools_rejects_unsupported_translation_clearly(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
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


_OPENAI_HINTS: JsonObject = {
    "prompt_cache_key": "session",
    "prompt_cache_retention": "in_memory",
    "safety_identifier": "person",
    "user": "person",
}


@pytest.mark.parametrize("destination", [chat_messages_body, chat_to_gemini])
def test_openai_request_hints_are_omitted_on_other_protocols(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    plain: JsonObject = {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    hinted = ChatRequest.model_validate({**plain, **_OPENAI_HINTS})
    assert destination(hinted) == destination(ChatRequest.model_validate(plain))


def test_openai_request_hints_reach_responses() -> None:
    result = chat_responses_body(
        ChatRequest.model_validate(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                **_OPENAI_HINTS,
            }
        )
    )
    assert {key: result.get(key) for key in _OPENAI_HINTS} == _OPENAI_HINTS


@pytest.mark.parametrize("destination", [chat_messages_body, chat_to_gemini])
def test_refusal_history_keeps_its_text_across_other_protocols(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
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


@pytest.mark.parametrize("destination", [chat_messages_body, chat_responses_body])
def test_echoed_chat_reasoning_keeps_signed_before_unsigned(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    origin = ReplayOrigin("test", "chat", "endpoint", "account", "model")
    record = ReplayRecord(
        origin,
        {
            "type": "reasoning",
            "encrypted_content": "cipher-signed",
            "reasoning_details": [{"type": "reasoning.text", "text": "A"}],
        },
    )
    request = ChatRequest(
        model="model",
        messages=[
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": [
                    {"type": "reasoning.text", "id": "signed", "text": "A"},
                    {
                        "type": "reasoning.encrypted",
                        "id": "signed",
                        "data": encode_replay(record),
                    },
                    {"type": "reasoning.text", "id": "unsigned", "text": "B"},
                ],
            }
        ],
    )
    result = destination(request)
    carrier = {"type": "reasoning", "encrypted_content": encode_replay(record)}
    if destination is chat_messages_body:
        parts = object_array(object_array(result["messages"])[0]["content"])
        assert parts[0] == {"type": "redacted_thinking", "data": encode_replay(record)}
        assert parts[1] == {"type": "thinking", "thinking": "B"}
        assert parts[2] == {"type": "text", "text": "answer"}
    else:
        items = object_array(result["input"])
        assert items[0] == carrier
        assert object_array(items[1]["content"])[0] == {
            "type": "reasoning_text",
            "text": "B",
        }
        assert object_array(items[2]["content"])[0] == {
            "type": "output_text",
            "text": "answer",
        }
