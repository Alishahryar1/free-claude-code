from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_chat import (
    close_chat_tool_result_turns,
    estimate_chat_input_tokens,
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


def test_estimate_chat_input_tokens_counts_message_content_and_tools() -> None:
    body = {
        "messages": [{"role": "system", "content": "system " * 100}],
        "tools": [{"name": "read", "description": "word " * 100}],
    }

    estimate = estimate_chat_input_tokens(body)

    # Message content plus tool schemas both count; the estimate must exceed
    # either alone so a large tool catalog cannot slip past a capacity check.
    assert estimate > estimate_chat_input_tokens({"messages": body["messages"]})
    assert estimate > estimate_chat_input_tokens({"tools": body["tools"]})
    assert estimate >= 4 * len(body["messages"])


def test_estimate_chat_input_tokens_counts_tool_calls_and_reasoning() -> None:
    plain = {"messages": [{"role": "assistant", "content": "ok"}]}
    with_calls = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "thinking " * 50,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "' + "x" * 100 + '"}',
                        },
                    }
                ],
            }
        ]
    }

    assert estimate_chat_input_tokens(with_calls) > estimate_chat_input_tokens(plain)


def test_estimate_chat_input_tokens_counts_data_url_images_by_payload() -> None:
    small = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 30_000},
                    }
                ],
            }
        ]
    }
    large = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 300_000},
                    }
                ],
            }
        ]
    }

    assert estimate_chat_input_tokens(small) >= 85
    assert estimate_chat_input_tokens(large) > estimate_chat_input_tokens(small)


def test_estimate_chat_input_tokens_tolerates_missing_and_malformed_bodies() -> None:
    assert estimate_chat_input_tokens({}) == 0
    assert estimate_chat_input_tokens({"messages": "not a list"}) == 0
    # Malformed entries contribute no content tokens, but the per-message
    # overhead still applies, matching the Anthropic estimator.
    assert estimate_chat_input_tokens({"messages": ["not a mapping", None]}) == 8
    assert (
        estimate_chat_input_tokens(
            {"messages": [{"role": "user", "content": [{"type": "unknown"}]}]}
        )
        >= 1
    )
