import json
import pytest
from providers.common.message_converter import AnthropicToOpenAIConverter

# --- Mock Classes ---


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockBlock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._data = kwargs

    def get(self, key, default=None):
        return self._data.get(key, default)


class MockTool:
    def __init__(self, name, description, input_schema):
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


def test_convert_assistant_message_thinking_include_reasoning_for_openrouter():
    """When include_reasoning_for_openrouter=True, reasoning_content is added."""
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, include_reasoning_for_openrouter=True
    )

    assert len(result) == 1
    assert result[0]["reasoning_content"] == "I need to calculate this."
    assert "<think>" in result[0]["content"]


def test_convert_assistant_message_tool_use():
    content = [
        MockBlock(type="text", text="I will call the tool."),
        MockBlock(
            type="tool_use", id="call_1", name="search", input={"query": "python"}
        ),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert "I will call the tool." in msg["content"]
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"]) == {"query": "python"}


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
    messages = [MockMessage("assistant", content)]
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
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert "<think>" in result[1]["content"]
    assert result[2]["tool_calls"][0]["id"] == "t1"


# --- Edge Cases ---


def test_get_block_attr_defaults():
    # Test helper directly
    from providers.common.message_converter import get_block_attr

    assert get_block_attr({}, "missing", "default") == "default"
    assert get_block_attr(object(), "missing", "default") == "default"


def test_input_not_dict():
    # Tool input might not be a dict (e.g. malformed or string)
    content = [MockBlock(type="tool_use", id="call_x", name="f", input="some_string")]
    messages = [MockMessage("assistant", content)]
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
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
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
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 1
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
