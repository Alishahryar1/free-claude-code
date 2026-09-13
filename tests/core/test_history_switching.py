"""Regressions for history loss when changing upstream providers."""

import json
from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.conversion import AnthropicToOpenAIConverter
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.history_replay import (
    HistoryScope,
    ReplayOrigin,
    prepare_history,
)
from free_claude_code.core.openai_responses.chat_request import (
    build_responses_chat_request,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.deepseek.client import DeepSeekProvider
from free_claude_code.providers.deepseek.compat import finalize_deepseek_chat_body
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.providers.support import immediate_admission, make_provider_config


def _chat(items):
    return build_responses_chat_request(
        OpenAIResponsesRequest(model="synthetic", input=items),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        structured_reasoning_details=True,
    ).body["messages"]


def _reasoning(text, opaque):
    return {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": opaque,
    }


def _call(call_id, arguments="{}"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
    }


def test_grouped_tool_calls_keep_both_reasoning_records():
    messages = _chat(
        [
            {"role": "user", "content": "look up both"},
            _reasoning("first", "opaque1"),
            _call("call_a"),
            _reasoning("second", "opaque2"),
            _call("call_b"),
            {"type": "function_call_output", "call_id": "call_a", "output": "17"},
            {"type": "function_call_output", "call_id": "call_b", "output": "18"},
        ]
    )
    assistant = messages[1]
    assert assistant["reasoning_content"] == "first\nsecond"
    assert [detail["data"] for detail in assistant["reasoning_details"]] == [
        "opaque1",
        "opaque2",
    ]


def test_quarantined_tool_call_does_not_donate_its_reasoning():
    messages = _chat(
        [
            {"role": "user", "content": "continue"},
            _reasoning("broken call only", "opaque-broken"),
            _call("broken", "{"),
            _call("valid"),
            {"type": "function_call_output", "call_id": "valid", "output": "17"},
        ]
    )
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert not assistant.get("reasoning_content")
    assert "reasoning_details" not in assistant


def test_completed_hosted_tool_history_remains_readable():
    messages = _chat(
        [
            {"role": "user", "content": "find the sample"},
            {
                "type": "web_search_call",
                "id": "search_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "sample",
                    "sources": [{"url": "https://example.org/result"}],
                },
            },
            {"role": "assistant", "content": "Found it."},
            {"role": "user", "content": "which source?"},
        ]
    )
    rendered = json.dumps(messages)
    assert "https://example.org/result" in rendered
    assert "web_search_call" in rendered
    assert any(
        message["role"] == "assistant"
        and "web_search_call" in str(message.get("content"))
        for message in messages
    )


def test_opaque_details_stay_with_the_assistant_that_produced_them():
    provider = OpenRouterProvider(
        make_provider_config(api_key="synthetic", base_url="https://example.org/v1"),
        admission=immediate_admission(),
    )
    detail = {"type": "reasoning.encrypted", "data": "second-only"}
    request = MessagesRequest.model_validate(
        {
            "model": "synthetic",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": json.dumps(detail)},
                        {"type": "text", "text": "second reply"},
                    ],
                },
                {"role": "user", "content": "continue again"},
            ],
        }
    )
    original = request.model_dump()
    assistants = [
        message
        for message in provider._chat._build_request_body(request)["messages"]
        if message["role"] == "assistant"
    ]
    assert "reasoning_details" not in assistants[0]
    assert assistants[1]["reasoning_details"] == [detail]
    assert request.model_dump() == original


def test_deepseek_tools_require_reasoning_on_earlier_plain_assistant_turns():
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "look up a value"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    }
    finalize_deepseek_chat_body(body, ReasoningPolicy.on())
    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][1]["reasoning_content"] == ""


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_deepseek_off_to_on_preserves_effort_and_all_available_history(wire):
    provider = DeepSeekProvider(
        make_provider_config(api_key="synthetic", base_url="https://example.invalid"),
        admission=immediate_admission(),
    )
    function = {"type": "object", "properties": {}}
    if wire == "messages":
        request = MessagesRequest.model_validate(
            {
                "model": "synthetic",
                "messages": [
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "lookup",
                                "id": "call_1",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_1",
                                "content": "17",
                            }
                        ],
                    },
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "second"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "real reasoning"},
                            {"type": "text", "text": "answer"},
                        ],
                    },
                    {"role": "user", "content": "new question, thinking on"},
                ],
                "tools": [{"name": "lookup", "input_schema": function}],
            }
        )
        original = deepcopy(request.model_dump())
        body = provider._chat._build_request_body(
            request, reasoning=ReasoningPolicy.on()
        )
    else:
        request = OpenAIResponsesRequest(
            model="synthetic",
            input=[
                {"role": "user", "content": "first"},
                _call("call_1"),
                {"type": "function_call_output", "call_id": "call_1", "output": "17"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "second"},
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "real reasoning"}],
                },
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "new question, thinking on"},
            ],
            tools=[{"type": "function", "name": "lookup", "parameters": function}],
        )
        original = deepcopy(request.model_dump())
        body = cast(
            dict[str, Any],
            provider._chat._build_responses_request_body(
                request, reasoning=ReasoningPolicy.on()
            ).body,
        )
    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assistants = [
        message for message in body["messages"] if message["role"] == "assistant"
    ]
    assert [message["reasoning_content"] for message in assistants] == [
        "",
        "",
        "real reasoning",
    ]
    assert request.model_dump() == original


def test_summary_is_context_even_when_destination_replays_full_reasoning():
    messages = _chat(
        [
            {"role": "user", "content": "first"},
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "A summary, not the hidden chain."}
                ],
            },
            {"role": "assistant", "content": "17"},
            {"role": "user", "content": "continue"},
        ]
    )
    assert "reasoning_content" not in messages[1]
    assert "[Earlier reasoning summary]" in messages[1]["content"]


def test_image_tool_result_does_not_start_a_new_user_turn():
    messages = _chat(
        [
            {"role": "user", "content": "look"},
            _reasoning("inspect the image", ""),
            _call("image"),
            {
                "type": "function_call_output",
                "call_id": "image",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.org/image.png",
                    }
                ],
            },
        ]
    )
    body = prepare_history(
        {"messages": messages},
        ReplayOrigin("a", "chat", "", "", "m"),
        scope=HistoryScope.TOOL_CONTINUATION,
    )
    assert body["messages"] == messages


def test_messages_interleaved_reasoning_belongs_to_the_tool_group():
    provider = OpenRouterProvider(
        make_provider_config(api_key="synthetic", base_url="https://example.org/v1"),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "look up both"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "first"},
                        {"type": "tool_use", "id": "a", "name": "lookup", "input": {}},
                        {"type": "thinking", "thinking": "second"},
                        {"type": "redacted_thinking", "data": "second-opaque"},
                        {"type": "tool_use", "id": "b", "name": "lookup", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": "17"},
                        {"type": "tool_result", "tool_use_id": "b", "content": "18"},
                    ],
                },
            ],
        }
    )
    messages = provider._chat._build_request_body(request)["messages"]
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["reasoning_content"] == "first\nsecond"
    assert assistant["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "second-opaque"}
    ]
    assert len([message for message in messages if message["role"] == "assistant"]) == 1


def test_completed_native_hosted_tools_become_readable_chat_history():
    provider = OpenRouterProvider(
        make_provider_config(api_key="synthetic", base_url="https://example.org/v1"),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "search",
                            "name": "web_search",
                            "input": {"query": "sample"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "search",
                            "content": [],
                        },
                        {"type": "text", "text": "No results."},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )
    body = provider._chat._build_request_body(request)
    rendered = json.dumps(body["messages"])
    assert "[Earlier tool record]" in rendered
    assert "sample" in rendered and "web_search_tool_result" in rendered
    assert "tool_calls" not in body["messages"][0]


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_malformed_carrier_fails_preflight_before_inference(wire):
    from free_claude_code.application.errors import InvalidRequestError

    provider = OpenRouterProvider(
        make_provider_config(api_key="synthetic", base_url="https://example.org/v1"),
        admission=immediate_admission(),
    )
    with pytest.raises(InvalidRequestError, match="replay"):
        if wire == "messages":
            provider.preflight_messages(
                MessagesRequest.model_validate(
                    {
                        "model": "m",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "redacted_thinking",
                                        "data": "fcc:history:v2:unsupported",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
        else:
            provider.preflight_responses(
                OpenAIResponsesRequest(
                    model="m",
                    input=[
                        {
                            "type": "reasoning",
                            "encrypted_content": "fcc:history:v2:unsupported",
                        }
                    ],
                )
            )


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_destination_without_native_reasoning_keeps_readable_context(wire):
    if wire == "responses":
        body = build_responses_chat_request(
            OpenAIResponsesRequest(
                model="m",
                input=[
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": "Preserve this thought."}
                        ],
                    },
                    {"role": "assistant", "content": "17"},
                ],
            ),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        ).body
    else:
        request = MessagesRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Preserve this thought."},
                            {"type": "text", "text": "17"},
                        ],
                    }
                ],
            }
        )
        body = {
            "messages": AnthropicToOpenAIConverter.convert_messages(
                request.messages, reasoning_replay=ReasoningReplayMode.DISABLED
            )
        }
    rendered = json.dumps(body)
    assert "[Earlier reasoning]" in rendered
    assert "Preserve this thought." in rendered
