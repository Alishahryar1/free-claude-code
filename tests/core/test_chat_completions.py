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
from free_claude_code.core.json_types import JsonObject
from tests.wire_payload import object_array


def test_legacy_none_disables_modern_tools_and_auto_allows_them() -> None:
    request = ChatRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "read"}}],
        tool_choice="auto",
        functions=[{"name": "legacy"}],
    )
    for legacy in ("none", "auto"):
        body = chat_responses_body(
            ChatRequest.model_validate({**request.body(), "function_call": legacy})
        )
        assert body["tool_choice"] == legacy
        assert [tool["name"] for tool in object_array(body["tools"])] == ["read"]


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
def test_translated_chat_omits_unsupported_logprobs(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    request = ChatRequest(
        model="model", messages=[{"role": "user", "content": "hi"}], logprobs=True
    )
    assert "logprobs" not in destination(request)


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
def test_cross_protocol_stream_options_omit_unknown_hints(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    request = ChatRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        stream_options={"include_obfuscation": False},
    )
    assert "stream_options" not in destination(request)


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
def test_allowed_tools_omits_unavailable_automatic_tools(
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
    body = destination(request)
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "toolConfig" not in body


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


@pytest.mark.parametrize(
    "destination", [chat_messages_body, chat_responses_body, chat_to_gemini]
)
def test_untranslated_history_keeps_only_the_supported_call_and_its_result(
    destination: Callable[[ChatRequest], JsonObject],
) -> None:
    request = ChatRequest(
        model="m",
        messages=[
            {"role": "user", "content": "check"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "future_action"},
                    {"type": "future_action", "id": "future"},
                    {
                        "type": "function",
                        "id": "kept",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "future",
                "content": "untranslated result",
            },
            {"role": "tool", "tool_call_id": "kept", "content": ""},
        ],
    )
    assert request.reserved_tool_ids() == {"kept", "future"}
    body = destination(request)
    assert "kept" in str(body) and "untranslated result" not in str(body)


def test_result_cannot_be_assigned_to_an_ambiguous_source_call() -> None:
    request = ChatRequest(
        model="m",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "future_action", "id": "duplicate"},
                    {
                        "type": "function",
                        "id": "duplicate",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "duplicate", "content": "ambiguous"},
        ],
    )
    with pytest.raises(ChatConversionError, match="uplicate"):
        chat_responses_body(request)
