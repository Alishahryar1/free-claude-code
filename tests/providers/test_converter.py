import json

import pytest

from free_claude_code.core.anthropic import (
    AnthropicToOpenAIConverter,
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_chat import (
    is_synthetic_chat_tool_turn_boundary,
)

# --- Mock Classes ---


class MockMessage:
    def __init__(self, role, content, reasoning_content=None):
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content


class MockBlock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._data = kwargs

    def model_dump(self, **kwargs):
        return dict(self._data)

    def get(self, key, default=None):
        return self._data.get(key, default)


class MockTool:
    def __init__(self, name, description, input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


# --- System Prompt Tests ---


def test_convert_system_prompt_str():
    system = "You are a helpful assistant."
    result = AnthropicToOpenAIConverter.convert_system_prompt(system)
    assert result == {"role": "system", "content": system}


def test_convert_system_prompt_list_text():
    system = [
        MockBlock(type="text", text="Part 1"),
        MockBlock(type="text", text="Part 2"),
    ]
    result = AnthropicToOpenAIConverter.convert_system_prompt(system)
    assert result == {"role": "system", "content": "Part 1\n\nPart 2"}


def test_convert_system_prompt_none():
    assert AnthropicToOpenAIConverter.convert_system_prompt(None) is None


def test_openai_build_uses_only_top_level_system_role() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "system": "Conversation-wide instructions",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "system", "content": "Instructions from this point"},
                {"role": "system", "content": "A second reminder"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "system", "content": "A final reminder"},
            ],
        }
    )

    body = build_base_request_body(request)

    assert body["messages"] == [
        {"role": "system", "content": "Conversation-wide instructions"},
        {
            "role": "user",
            "content": (
                "First question\n\nInstructions from this point\n\nA second reminder"
            ),
        },
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question\n\nA final reminder"},
    ]


def test_openai_build_demotes_inline_system_text_blocks_without_repositioning() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {"role": "user", "content": "Before"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "First instruction"},
                        {"type": "text", "text": "Second instruction"},
                    ],
                },
            ],
        }
    )

    body = build_base_request_body(request)

    assert body["messages"] == [
        {
            "role": "user",
            "content": "Before\n\nFirst instruction\n\nSecond instruction",
        }
    ]


def test_inline_system_message_preserves_existing_openai_cache_prefix() -> None:
    prefix_request = MessagesRequest.model_validate(
        {
            "model": "model",
            "system": "Conversation-wide instructions",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
        }
    )
    continued_request = MessagesRequest.model_validate(
        {
            "model": "model",
            "system": "Conversation-wide instructions",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {
                    "role": "system",
                    "content": (
                        "<system-reminder>Instructions from this point"
                        "</system-reminder>"
                    ),
                },
            ],
        }
    )

    prefix = build_base_request_body(prefix_request)["messages"]
    continued = build_base_request_body(continued_request)["messages"]

    assert continued[: len(prefix)] == prefix
    assert continued[len(prefix) :] == [
        {
            "role": "user",
            "content": (
                "Second question\n\n<system-reminder>Instructions from this point"
                "</system-reminder>"
            ),
        }
    ]


def test_inline_system_message_coalesces_with_multimodal_user_content() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {"role": "user", "content": "Existing context."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        },
                        {"type": "text", "text": "Inspect this image."},
                    ],
                },
                {"role": "system", "content": "Focus on correctness."},
            ],
        }
    )

    body = build_base_request_body(request)

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Existing context."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
                {
                    "type": "text",
                    "text": "Inspect this image.\n\nFocus on correctness.",
                },
            ],
        }
    ]


def test_inline_system_message_follows_completed_tool_result() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {"role": "user", "content": "Use the tool"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
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
                            "content": "done",
                        }
                    ],
                },
                {"role": "system", "content": "New instructions"},
            ],
        }
    )

    body = build_base_request_body(request)

    assert [message["role"] for message in body["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert body["messages"][3] == {"role": "assistant", "content": " "}
    assert body["messages"][-1]["content"] == "New instructions"


def test_openai_build_rejects_non_text_inline_system_blocks() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/a.png",
                            },
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(
        OpenAIConversionError,
        match="inline Anthropic system message content block 'image' without data loss",
    ):
        build_base_request_body(request)


def test_openai_build_rejects_empty_inline_system_content() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "system", "content": []}],
        }
    )

    with pytest.raises(OpenAIConversionError, match="contain text"):
        build_base_request_body(request)


# --- Tool Conversion Tests ---


def test_convert_tools():
    tools = [
        MockTool(
            "get_weather",
            "Get weather",
            {"type": "object", "properties": {"loc": {"type": "string"}}},
        ),
        MockTool("calculator", None, {"type": "object"}),
    ]
    result = AnthropicToOpenAIConverter.convert_tools(tools)
    assert len(result) == 2

    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["description"] == "Get weather"
    assert result[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"loc": {"type": "string"}},
    }

    assert result[1]["function"]["name"] == "calculator"
    assert result[1]["function"]["description"] == ""  # Check default empty string


def test_convert_tool_without_input_schema_uses_empty_object_schema():
    tools = [MockTool("web_search", None)]

    result = AnthropicToOpenAIConverter.convert_tools(tools)

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        (
            {"type": "tool", "name": "echo_smoke"},
            {"type": "function", "function": {"name": "echo_smoke"}},
        ),
        ({"type": "any"}, "required"),
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        (
            {"type": "function", "function": {"name": "already_openai"}},
            {"type": "function", "function": {"name": "already_openai"}},
        ),
    ],
)
def test_convert_tool_choice(tool_choice, expected):
    result = AnthropicToOpenAIConverter.convert_tool_choice(tool_choice)
    assert result == expected


def test_openai_build_materializes_auto_tool_choice_when_tools_are_declared():
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "Use the echo tool."}],
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo a value.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                }
            ],
        }
    )

    body = build_base_request_body(request)

    assert body["tool_choice"] == "auto"


def test_openai_build_omits_tool_choice_when_tools_are_absent():
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = build_base_request_body(request)

    assert "tools" not in body
    assert "tool_choice" not in body


# --- Message Conversion Tests: User ---


def test_convert_user_message_str():
    messages = [MockMessage("user", "Hello world")]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {"role": "user", "content": "Hello world"}


def test_convert_user_message_list_text():
    content = [
        MockBlock(type="text", text="Hello"),
        MockBlock(type="text", text="World"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {"role": "user", "content": "Hello\nWorld"}


def test_convert_user_message_tool_result_str():
    content = [
        MockBlock(type="tool_result", tool_use_id="tool_123", content="Result data")
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {
        "role": "tool",
        "tool_call_id": "tool_123",
        "content": "Result data",
    }


def test_convert_user_message_tool_result_list():
    # Tool result content as a list of text blocks
    tool_content = [
        {"type": "text", "text": "Line 1"},
        {"type": "text", "text": "Line 2"},
    ]
    content = [
        MockBlock(type="tool_result", tool_use_id="tool_456", content=tool_content)
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "tool_456"
    assert result[0]["content"] == "Line 1\nLine 2"


def test_convert_user_message_mixed_text_and_tool_result():
    # Note: Anthropic/OpenAI mapping usually separates these, but the converter handles lists
    # User text usually comes before tool results in a turn, or after.
    # The converter splits them into separate messages if they are different roles?
    # Let's check logic: _convert_user_message returns a list of dicts.
    content = [
        MockBlock(type="text", text="Here is the result:"),
        MockBlock(type="tool_result", tool_use_id="tool_789", content="42"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    # Order is preserved: user text first, then tool result.
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "Here is the result:"}
    assert result[1] == {"role": "tool", "tool_call_id": "tool_789", "content": "42"}


# --- Message Conversion Tests: Assistant ---


def test_convert_assistant_message_text_only():
    messages = [MockMessage("assistant", "I am ready.")]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {"role": "assistant", "content": "I am ready."}


def test_convert_assistant_message_blocks_text():
    content = [MockBlock(type="text", text="Part A")]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0] == {"role": "assistant", "content": "Part A"}


def test_convert_assistant_message_thinking():
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 1
    # Expecting <think> tags
    expected_content = (
        "<think>\nI need to calculate this.\n</think>\n\nThe answer is 4."
    )
    assert result[0]["content"] == expected_content
    assert "reasoning_content" not in result[0]


def test_convert_assistant_message_thinking_replays_reasoning_content():
    """Top-level reasoning replay avoids duplicating thinking into content."""
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert len(result) == 1
    assert result[0]["reasoning_content"] == "I need to calculate this."
    assert result[0]["content"] == "The answer is 4."
    assert "<think>" not in result[0]["content"]


def test_convert_assistant_message_thinking_replays_reasoning():
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]

    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning": "I need to calculate this.",
        }
    ]


def test_convert_assistant_top_level_reasoning_content_is_preserved():
    messages = [
        MockMessage(
            "assistant",
            "The answer is 4.",
            reasoning_content="I need to calculate this.",
        )
    ]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "I need to calculate this.",
        }
    ]


def test_convert_assistant_empty_top_level_reasoning_content_is_preserved():
    messages = [MockMessage("assistant", "The answer is 4.", reasoning_content="")]

    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "",
        }
    ]


def test_convert_assistant_thinking_tool_use_replays_top_level_reasoning():
    content = [
        MockBlock(type="thinking", thinking="I should call the tool."),
        MockBlock(
            type="tool_use",
            id="call_reasoning",
            name="search",
            input={"query": "python"},
        ),
    ]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_reasoning",
                    content="result",
                )
            ],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert len(result) == 2
    assert result[0]["content"] == ""
    assert result[0]["reasoning_content"] == "I should call the tool."
    assert "<think>" not in result[0]["content"]
    assert result[0]["tool_calls"][0]["id"] == "call_reasoning"


def test_convert_assistant_tool_use_replays_ollama_reasoning_field():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="thinking", thinking="Call the tool."),
                MockBlock(type="tool_use", id="call_1", name="Read", input={}),
            ],
        ),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_1", content="done")],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING
    )

    assert result[0]["reasoning"] == "Call the tool."
    assert "reasoning_content" not in result[0]
    assert result[0]["tool_calls"][0]["id"] == "call_1"


def test_convert_assistant_empty_thinking_replays_empty_reasoning_content():
    content = [
        MockBlock(type="thinking", thinking=""),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]

    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "",
        }
    ]


def test_convert_assistant_tool_use_replays_empty_reasoning_content():
    content = [
        MockBlock(type="thinking", thinking=""),
        MockBlock(type="tool_use", id="call_empty", name="Read", input={}),
    ]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_empty",
                    content="result",
                )
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert result[0]["content"] == ""
    assert result[0]["reasoning_content"] == ""
    assert result[0]["tool_calls"][0]["id"] == "call_empty"


def _convert_tool_turn(
    assistant_content: list[MockBlock],
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.REASONING_CONTENT,
) -> list[dict]:
    tool_ids = [
        block.get("id")
        for block in assistant_content
        if block.get("type") == "tool_use"
    ]
    return AnthropicToOpenAIConverter.convert_messages(
        [
            MockMessage("assistant", assistant_content),
            MockMessage(
                "user",
                [
                    MockBlock(
                        type="tool_result",
                        tool_use_id=tool_id,
                        content="result",
                    )
                    for tool_id in tool_ids
                ],
            ),
        ],
        reasoning_replay=reasoning_replay,
    )


def test_convert_assistant_tool_use_without_thinking_replays_empty_reasoning_content():
    result = _convert_tool_turn(
        [MockBlock(type="tool_use", id="call_missing", name="Read", input={})]
    )

    assert result[0]["content"] == ""
    assert result[0]["reasoning_content"] == ""
    assert result[0]["tool_calls"][0]["id"] == "call_missing"


def test_convert_parallel_tool_use_without_thinking_replays_one_empty_reasoning_field():
    result = _convert_tool_turn(
        [
            MockBlock(type="tool_use", id="call_1", name="Read", input={}),
            MockBlock(type="tool_use", id="call_2", name="Glob", input={}),
        ]
    )

    assert result[0]["reasoning_content"] == ""
    assert [tool_call["id"] for tool_call in result[0]["tool_calls"]] == [
        "call_1",
        "call_2",
    ]


def test_convert_redacted_thinking_tool_use_replays_empty_reasoning_without_data():
    result = _convert_tool_turn(
        [
            MockBlock(type="redacted_thinking", data="opaque-secret"),
            MockBlock(type="tool_use", id="call_redacted", name="Read", input={}),
        ]
    )

    assert result[0]["reasoning_content"] == ""
    assert result[0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque-secret"}
    ]
    assert "opaque-secret" not in result[0]["reasoning_content"]


def test_convert_text_only_assistant_without_thinking_omits_reasoning_content():
    result = AnthropicToOpenAIConverter.convert_messages(
        [MockMessage("assistant", "Visible answer")],
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    )

    assert result == [{"role": "assistant", "content": "Visible answer"}]


@pytest.mark.parametrize(
    "reasoning_replay",
    [
        ReasoningReplayMode.THINK_TAGS,
        ReasoningReplayMode.REASONING,
        ReasoningReplayMode.DISABLED,
    ],
)
def test_convert_tool_use_without_thinking_does_not_change_other_replay_modes(
    reasoning_replay: ReasoningReplayMode,
) -> None:
    result = _convert_tool_turn(
        [MockBlock(type="tool_use", id="call_other", name="Read", input={})],
        reasoning_replay,
    )

    assert "reasoning_content" not in result[0]


def test_convert_assistant_message_thinking_becomes_context_when_disabled():
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages,
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    assert len(result) == 1
    assert "reasoning_content" not in result[0]
    assert "<think>" not in result[0]["content"]
    assert (
        result[0]["content"]
        == "[Earlier reasoning]\nI need to calculate this.\n\nThe answer is 4."
    )


def test_convert_assistant_top_level_reasoning_becomes_context_when_disabled():
    messages = [
        MockMessage(
            "assistant",
            "The answer is 4.",
            reasoning_content="I need to calculate this.",
        )
    ]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.DISABLED
    )

    assert result == [
        {
            "role": "assistant",
            "content": "[Earlier reasoning]\nI need to calculate this.\n\nThe answer is 4.",
        }
    ]


def test_convert_assistant_message_tool_use():
    content = [
        MockBlock(type="text", text="I will call the tool."),
        MockBlock(
            type="tool_use", id="call_1", name="search", input={"query": "python"}
        ),
    ]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_1", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 2
    msg = result[0]
    assert msg["role"] == "assistant"
    assert "I will call the tool." in msg["content"]
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"]) == {"query": "python"}


def test_convert_assistant_tool_use_preserves_extra_content():
    content = [
        MockBlock(
            type="tool_use",
            id="call_1",
            name="search",
            input={"query": "python"},
            extra_content={"google": {"thought_signature": "sig"}},
        ),
    ]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_1", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result[0]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "sig"}
    }


def test_convert_assistant_message_empty_content():
    # Verify that empty content becomes a single space (NIM requirement)
    # if no tool calls are present.
    content = []
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0]["content"] == " "


def test_convert_assistant_message_tool_use_no_text():
    # If tool usage exists, content can be empty string?
    # Logic: if not content_str and not tool_calls: content_str = " "
    # So if tool_calls exist, content_str can be empty string?
    # Actually code says: if not content_str and not tool_calls.
    # So if tool_calls is present, content_str remains "" (empty).

    content = [MockBlock(type="tool_use", id="call_2", name="test", input={})]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_2", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert (
        result[0]["content"] == ""
    )  # Should be empty string, not space, because tools exist
    assert len(result[0]["tool_calls"]) == 1


def test_convert_mixed_blocks_and_types_and_roles():
    # comprehensive flow
    messages = [
        MockMessage("user", "Start"),
        MockMessage(
            "assistant",
            [
                MockBlock(type="thinking", thinking="Thinking..."),
                MockBlock(type="text", text="Here is a tool."),
            ],
        ),
        MockMessage(
            "assistant", [MockBlock(type="tool_use", id="t1", name="f", input={})]
        ),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="t1", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 4
    assert result[0]["role"] == "user"
    assert "<think>" in result[1]["content"]
    assert result[2]["tool_calls"][0]["id"] == "t1"


# --- Edge Cases ---


def test_get_block_attr_defaults():
    # Test helper directly
    from free_claude_code.core.anthropic import get_block_attr

    assert get_block_attr({}, "missing", "default") == "default"
    assert get_block_attr(object(), "missing", "default") == "default"


def test_input_not_dict():
    # Tool input might not be a dict (e.g. malformed or string)
    content = [MockBlock(type="tool_use", id="call_x", name="f", input="some_string")]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_x", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    # The converter calls json.dumps(tool_input) if dict, else str(tool_input)
    # So it should be "some_string"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == "some_string"


# --- Parametrized Edge Case Tests ---


@pytest.mark.parametrize(
    "system_input,expected",
    [
        ("You are helpful.", {"role": "system", "content": "You are helpful."}),
        (
            [MockBlock(type="text", text="A"), MockBlock(type="text", text="B")],
            {"role": "system", "content": "A\n\nB"},
        ),
        (None, None),
        ("", {"role": "system", "content": ""}),
        ([], None),
    ],
    ids=["string", "list_text", "none", "empty_string", "empty_list"],
)
def test_convert_system_prompt_parametrized(system_input, expected):
    """Parametrized system prompt conversion covering edge cases."""
    result = AnthropicToOpenAIConverter.convert_system_prompt(system_input)
    assert result == expected


@pytest.mark.parametrize(
    "content,expected_content",
    [
        ("Hello world", "Hello world"),
        ("", ""),
        ([MockBlock(type="text", text="A"), MockBlock(type="text", text="B")], "A\nB"),
        ([MockBlock(type="text", text="")], ""),
    ],
    ids=["simple_string", "empty_string", "list_blocks", "empty_text_block"],
)
def test_convert_user_message_parametrized(content, expected_content):
    """Parametrized user message conversion."""
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) >= 1
    assert result[0]["content"] == expected_content


def test_convert_assistant_message_unknown_block_type():
    """Unknown block types should be silently skipped."""
    content = [
        MockBlock(type="unknown_type", data="something"),
        MockBlock(type="text", text="visible"),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert "visible" in result[0]["content"]


def test_convert_tool_use_none_input():
    """Tool use with None input should not crash."""
    content = [MockBlock(type="tool_use", id="call_n", name="test", input=None)]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_n", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 2
    assert "tool_calls" in result[0]


def test_convert_assistant_interleaved_order_preserved():
    """Interleaved thinking, text, tool_use should preserve thinking+text order in content.

    Bug: Current implementation collects all thinking, then all text, then tool_calls.
    Original order [thinking, text, thinking, tool_use] becomes [all thinking, all text, tool_calls],
    losing the interleaving. Content string should reflect original block order for thinking+text.
    Tool calls stay at end (API constraint).
    """
    content = [
        MockBlock(type="thinking", thinking="First thought."),
        MockBlock(type="text", text="Here is the answer."),
        MockBlock(type="thinking", thinking="Second thought."),
        MockBlock(type="tool_use", id="call_1", name="search", input={"q": "x"}),
    ]
    messages = [
        MockMessage("assistant", content),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_1", content="result")],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 2
    msg = result[0]
    # Expected: thinking1, text, thinking2 in that order within content; tool_calls at end
    expected_content = "<think>\nFirst thought.\n</think>\n\nHere is the answer.\n\n<think>\nSecond thought.\n</think>"
    assert msg["content"] == expected_content, (
        f"Interleaved order lost. Got: {msg['content']!r}"
    )
    assert len(msg["tool_calls"]) == 1


def test_convert_user_message_text_before_tool_result_order():
    """User message with text then tool_result should preserve order: user text first, then tool.

    Bug: Current implementation emits tool_result immediately, then user text at end.
    Anthropic order is typically: user says something, then provides tool results.
    """
    content = [
        MockBlock(type="text", text="Please use this result:"),
        MockBlock(type="tool_result", tool_use_id="t1", content="42"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 2
    # Expected: user text first, then tool result
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Please use this result:"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == "t1"


def test_convert_multiple_tool_results():
    """Multiple tool results in a single user message."""
    content = [
        MockBlock(type="tool_result", tool_use_id="t1", content="Result 1"),
        MockBlock(type="tool_result", tool_use_id="t2", content="Result 2"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 2
    assert result[0]["tool_call_id"] == "t1"
    assert result[1]["tool_call_id"] == "t2"


def test_convert_user_message_tool_result_dict_as_json():
    content = [
        MockBlock(
            type="tool_result",
            tool_use_id="t_dict",
            content={"ok": True, "count": 2},
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0]["role"] == "tool"
    assert result[0]["content"] == '{"ok": true, "count": 2}'


def test_assistant_redacted_thinking_omitted_from_openai_chat():
    """Opaque redacted_thinking is not materialized as content or reasoning_content."""
    content = [
        MockBlock(type="redacted_thinking", data="secret-opaque"),
        MockBlock(type="text", text="Visible."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )
    assert result[0]["content"] == "Visible."
    assert "secret-opaque" not in result[0]["content"]
    assert "reasoning_content" not in result[0]


@pytest.mark.parametrize(
    "source,expected_url",
    [
        (
            {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            "data:image/png;base64,AAAA",
        ),
        (
            {"type": "url", "url": "https://example.com/image.png"},
            "https://example.com/image.png",
        ),
    ],
)
def test_convert_user_message_image_sources(source, expected_url):
    messages = [MockMessage("user", [MockBlock(type="image", source=source)])]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result == [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": expected_url}}],
        }
    ]


@pytest.mark.parametrize(
    "data",
    (
        "data:IMAGE/PNG;BASE64,aGVs\r\nbG8=",
        "  aG\tVs\r\nbG8=  ",
    ),
)
def test_convert_user_message_canonicalizes_equivalent_base64(data):
    messages = [
        MockMessage(
            "user",
            [
                MockBlock(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": data,
                    },
                )
            ],
        )
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result[0]["content"][0]["image_url"]["url"] == (
        "data:image/png;base64,aGVsbG8="
    )


def test_convert_image_tool_result_to_schema_valid_chat_history():
    image_url = "https://images.example.test/tool.png"
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_image", name="Read", input={})],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_image",
                    content=[
                        {
                            "type": "image",
                            "source": {"type": "url", "url": image_url},
                        }
                    ],
                )
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[1] == {
        "role": "tool",
        "tool_call_id": "call_image",
        "content": "[Image-bearing tool output follows in user content.]",
    }
    assert is_synthetic_chat_tool_turn_boundary(result[2])
    assert result[3]["content"] == [
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_image":',
        },
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    assert image_url not in result[1]["content"]


def test_convert_mixed_image_tool_result_preserves_part_order_once():
    image_url = "data:image/png;base64,aGVsbG8="
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_mixed", name="Read", input={})],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_mixed",
                    content=[
                        {"type": "text", "text": "before"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            },
                        },
                        {"type": "text", "text": "after"},
                    ],
                )
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result[3]["content"] == [
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_mixed":',
        },
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "after"},
    ]
    assert (
        sum(
            part.get("text") == "before"
            for message in result
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )
        == 1
    )


def test_convert_parallel_image_tool_results_closes_all_tools_before_images():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_a", name="ReadA", input={}),
                MockBlock(type="tool_use", id="call_b", name="ReadB", input={}),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_b",
                    content=[
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/b.png"},
                        }
                    ],
                ),
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_a",
                    content=[
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/a.png"},
                        }
                    ],
                ),
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
    ]
    assert [message["tool_call_id"] for message in result[1:3]] == [
        "call_a",
        "call_b",
    ]
    assert result[4]["content"] == [
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_a":',
        },
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_b":',
        },
        {"type": "image_url", "image_url": {"url": "https://x/b.png"}},
    ]


def test_convert_image_tool_result_precedes_following_user_text():
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_image", name="Read", input={})],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_image",
                    content=[
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/a.png"},
                        }
                    ],
                ),
                MockBlock(type="text", text="Now describe it."),
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result[3]["content"][-1] == {
        "type": "text",
        "text": "Now describe it.",
    }


@pytest.mark.parametrize(
    "source",
    (
        {"type": "base64", "media_type": "image/png", "data": "SECRET_BAD"},
        {"type": "file", "file_id": "file_1"},
    ),
)
def test_convert_image_tool_result_rejects_unportable_source_safely(source):
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_image", name="Read", input={})],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_image",
                    content=[{"type": "image", "source": source}],
                )
            ],
        ),
    ]

    with pytest.raises(OpenAIConversionError) as exc_info:
        AnthropicToOpenAIConverter.convert_messages(messages)

    assert "SECRET_BAD" not in str(exc_info.value)


def test_convert_user_message_preserves_interleaved_image_text_order():
    messages = [
        MockMessage(
            "user",
            [
                MockBlock(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "RklSU1Q=",
                    },
                ),
                MockBlock(type="text", text="Compare the first image with this one."),
                MockBlock(
                    type="image",
                    source={"type": "url", "url": "https://example.com/second.jpg"},
                ),
                MockBlock(type="text", text="Describe the differences."),
            ],
        )
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,RklSU1Q="},
                },
                {
                    "type": "text",
                    "text": "Compare the first image with this one.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/second.jpg"},
                },
                {"type": "text", "text": "Describe the differences."},
            ],
        }
    ]


def test_convert_user_image_before_tool_result_preserves_message_order():
    messages = [
        MockMessage(
            "user",
            [
                MockBlock(type="text", text="Inspect this."),
                MockBlock(
                    type="image",
                    source={"type": "url", "url": "https://example.com/image.png"},
                ),
                MockBlock(type="tool_result", tool_use_id="tool_1", content="done"),
            ],
        )
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tool_1", "content": "done"},
    ]


@pytest.mark.parametrize(
    "source,error",
    [
        (
            {"type": "base64", "media_type": "", "data": "AAAA"},
            "media type",
        ),
        (
            {"type": "base64", "media_type": "image/png", "data": ""},
            "data must be non-empty",
        ),
        ({"type": "url", "url": ""}, "URL must be a non-empty"),
        ({"type": "file", "file_id": "file_1"}, "cannot cross"),
        ({}, "cannot cross"),
    ],
)
def test_convert_user_message_rejects_unconvertible_image_sources(source, error):
    messages = [MockMessage("user", [MockBlock(type="image", source=source)])]

    with pytest.raises(OpenAIConversionError, match=error):
        AnthropicToOpenAIConverter.convert_messages(messages)


def test_convert_assistant_text_after_tool_use_requires_matching_tool_result():
    """Dangling post-tool assistant text cannot be replayed as valid OpenAI chat."""
    content = [
        MockBlock(type="tool_use", id="call_z", name="Read", input={}),
        MockBlock(type="text", text="After tool"),
    ]
    messages = [MockMessage("assistant", content)]
    with pytest.raises(OpenAIConversionError, match="missing tool_result"):
        AnthropicToOpenAIConverter.convert_messages(messages)


def test_convert_assistant_text_after_tool_use_inserts_after_tool_results():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_z", name="Read", input={}),
                MockBlock(type="text", text="Post-tool commentary"),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_z",
                    content="file contents",
                )
            ],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0]["role"] == "assistant" and "tool_calls" in result[0]
    assert result[1]["role"] == "tool" and result[1]["tool_call_id"] == "call_z"
    assert result[2] == {"role": "assistant", "content": "Post-tool commentary"}


def _completed_single_tool_turn(*, user_text=None):
    result_blocks = [
        MockBlock(
            type="tool_result",
            tool_use_id="call_z",
            content="file contents",
        )
    ]
    if user_text is not None:
        result_blocks.append(MockBlock(type="text", text=user_text))
    return [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_z", name="Read", input={})],
        ),
        MockMessage("user", result_blocks),
    ]


def test_unrelated_user_text_before_tool_result_is_buffered_after_closed_tool_turn():
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_z", name="Read", input={})],
        ),
        MockMessage("user", "Please also summarize it."),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_z",
                    content="file contents",
                )
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[0]["tool_calls"][0]["id"] == "call_z"
    assert result[1]["tool_call_id"] == "call_z"
    assert result[2] == {"role": "assistant", "content": " "}
    assert result[3]["content"] == "Please also summarize it."


def test_user_after_completed_tool_result_gets_neutral_assistant_boundary():
    messages = [
        MockMessage("user", "Read the file."),
        *_completed_single_tool_turn(),
        MockMessage("user", "Please summarize it."),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[3] == {"role": "assistant", "content": " "}
    assert is_synthetic_chat_tool_turn_boundary(result[3])
    assert json.loads(json.dumps(result[3])) == {
        "role": "assistant",
        "content": " ",
    }
    assert result[4]["content"] == "Please summarize it."


def test_user_text_beside_tool_result_follows_neutral_assistant_boundary():
    messages = _completed_single_tool_turn(user_text="Please summarize it.")

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[2] == {"role": "assistant", "content": " "}
    assert result[3]["content"] == "Please summarize it."


def test_multiple_tool_results_get_one_assistant_boundary_before_user():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_a", name="ReadA", input={}),
                MockBlock(type="tool_use", id="call_b", name="ReadB", input={}),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_b",
                    content="result b",
                ),
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_a",
                    content="result a",
                ),
            ],
        ),
        MockMessage("user", "Compare both results."),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
    ]
    assert [message["tool_call_id"] for message in result[1:3]] == [
        "call_a",
        "call_b",
    ]
    assert result[3] == {"role": "assistant", "content": " "}
    assert result[4]["content"] == "Compare both results."


def test_terminal_tool_result_does_not_get_assistant_boundary():
    messages = _completed_single_tool_turn()

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == ["assistant", "tool"]


def test_existing_assistant_completion_prevents_extra_tool_turn_boundary():
    messages = [
        *_completed_single_tool_turn(),
        MockMessage("assistant", "The file contains useful information."),
        MockMessage("user", "Please summarize it."),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[2]["content"] == "The file contains useful information."


def test_unrelated_assistant_text_before_tool_result_is_buffered_until_after_tool_result():
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_z", name="Read", input={})],
        ),
        MockMessage("assistant", "Waiting for the result."),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_z",
                    content="file contents",
                )
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert result[0]["tool_calls"][0]["id"] == "call_z"
    assert result[1]["tool_call_id"] == "call_z"
    assert result[2]["content"] == "Waiting for the result."


def test_user_text_in_tool_result_message_is_replayed_after_tool_sequence():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_z", name="Read", input={}),
                MockBlock(type="text", text="Post-tool commentary"),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(type="text", text="Use this result too."),
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_z",
                    content="file contents",
                ),
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[1]["tool_call_id"] == "call_z"
    assert result[2]["content"] == "Post-tool commentary"
    assert result[3]["content"] == "Use this result too."


def test_nested_pending_tool_use_waits_for_its_own_tool_result_before_deferred_text():
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_a", name="ReadA", input={})],
        ),
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_b", name="ReadB", input={}),
                MockBlock(type="text", text="Post-call-b commentary"),
            ],
        ),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_a", content="result a")],
        ),
        MockMessage(
            "user",
            [MockBlock(type="tool_result", tool_use_id="call_b", content="result b")],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result[0]["tool_calls"][0]["id"] == "call_a"
    assert result[1]["tool_call_id"] == "call_a"
    assert result[2]["tool_calls"][0]["id"] == "call_b"
    assert result[3]["tool_call_id"] == "call_b"
    assert result[4]["content"] == "Post-call-b commentary"


def test_nested_pending_uses_early_nested_tool_result_after_outer_result():
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_a", name="ReadA", input={})],
        ),
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_b", name="ReadB", input={}),
                MockBlock(type="text", text="Post-call-b commentary"),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(type="tool_result", tool_use_id="call_b", content="result b"),
                MockBlock(type="tool_result", tool_use_id="call_a", content="result a"),
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result[0]["tool_calls"][0]["id"] == "call_a"
    assert result[1]["tool_call_id"] == "call_a"
    assert result[2]["tool_calls"][0]["id"] == "call_b"
    assert result[3]["tool_call_id"] == "call_b"
    assert result[4]["content"] == "Post-call-b commentary"


def test_multi_tool_turn_waits_for_all_results_before_deferred_text():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_a", name="ReadA", input={}),
                MockBlock(type="tool_use", id="call_b", name="ReadB", input={}),
                MockBlock(type="text", text="Both tools are done."),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(type="tool_result", tool_use_id="call_b", content="result b"),
                MockBlock(type="tool_result", tool_use_id="call_a", content="result a"),
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [message["tool_call_id"] for message in result[1:3]] == [
        "call_a",
        "call_b",
    ]
    assert result[3]["content"] == "Both tools are done."


def test_nested_pending_buffers_user_text_until_all_prior_tool_sequences_complete():
    messages = [
        MockMessage(
            "assistant",
            [MockBlock(type="tool_use", id="call_a", name="ReadA", input={})],
        ),
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_b", name="ReadB", input={}),
                MockBlock(type="text", text="Post-call-b commentary"),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(type="text", text="Use both results."),
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_a",
                    content="result a",
                ),
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_b",
                    content="result b",
                ),
            ],
        ),
    ]

    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert result[1]["tool_call_id"] == "call_a"
    assert result[3]["tool_call_id"] == "call_b"
    assert result[4]["content"] == "Post-call-b commentary"
    assert result[5]["content"] == "Use both results."


def test_openai_build_accepts_declared_native_top_level_hints() -> None:
    """OpenAI conversion ignores known non-OpenAI hints (e.g. context_management) without 400."""
    req = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "h"}],
            "context_management": {"edits": []},
            "output_config": {"foo": 1},
            "mcp_servers": [{"type": "url", "url": "https://x.com"}],
        }
    )
    body = build_base_request_body(req, default_max_tokens=100)
    assert "context_management" not in body
    assert "output_config" not in body
    assert "mcp_servers" not in body
    assert body["model"] == "m"


def test_openai_build_converts_validated_anthropic_image_block() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "vision-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/webp",
                                "data": "AAAA",
                            },
                        },
                        {"type": "text", "text": "What is shown?"},
                    ],
                }
            ],
        }
    )

    body = build_base_request_body(request)

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/webp;base64,AAAA"},
                },
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]


def test_openai_build_rejects_unknown_top_level_extras() -> None:
    """Truly unknown keys must still be rejected (not dropped silently)."""
    req = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "h"}],
            "experimental_client_only_passthrough": True,
        }
    )
    with pytest.raises(OpenAIConversionError, match="top-level request fields"):
        build_base_request_body(req)


@pytest.mark.parametrize(
    "content",
    [
        [MockBlock(type="server_tool_use", id="1", name="web_search", input={})],
        [MockBlock(type="web_search_tool_result", tool_use_id="1", content=[])],
        [
            MockBlock(
                type="web_fetch_tool_result",
                tool_use_id="1",
                content={"type": "web_fetch_result", "url": "https://a.com/x"},
            )
        ],
    ],
)
def test_convert_hosted_tool_history_preserves_results_and_rejects_active_calls(
    content,
) -> None:
    messages = [MockMessage("assistant", content)]
    if content[0].type == "server_tool_use":
        with pytest.raises(OpenAIConversionError, match="active server tool"):
            AnthropicToOpenAIConverter.convert_messages(messages)
    else:
        result = AnthropicToOpenAIConverter.convert_messages(messages)
        assert "[Earlier tool record]" in result[0]["content"]
