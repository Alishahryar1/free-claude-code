from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_chat import (
    close_chat_tool_result_turns,
    is_synthetic_chat_tool_turn_boundary,
)


def test_close_chat_tool_result_turns_inserts_identifiable_assistant_boundary() -> None:
    messages: list[JsonObject] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        {"role": "user", "content": "continue"},
    ]

    closed = close_chat_tool_result_turns(messages)

    assert [message["role"] for message in closed] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert closed[2]["content"] == " "
    assert is_synthetic_chat_tool_turn_boundary(closed[2]) is True
    assert is_synthetic_chat_tool_turn_boundary(closed[0]) is False


def test_close_chat_tool_result_turns_leaves_other_transitions_unchanged() -> None:
    messages: list[JsonObject] = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "done"},
    ]

    assert close_chat_tool_result_turns(messages) == messages
