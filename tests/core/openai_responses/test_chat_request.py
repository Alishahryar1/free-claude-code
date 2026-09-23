import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.openai_responses.chat_request import (
    build_responses_chat_request,
)
from free_claude_code.core.openai_responses.errors import ResponsesConversionError
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest


def _request(**overrides: object) -> OpenAIResponsesRequest:
    payload: dict[str, object] = {
        "model": "provider/model",
        "input": "Hello",
    }
    payload.update(overrides)
    return OpenAIResponsesRequest.model_validate(payload)


def test_build_responses_chat_request_preserves_rich_supported_semantics() -> None:
    translated = build_responses_chat_request(
        _request(
            instructions="System rules",
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Developer rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                            "detail": "low",
                        },
                    ],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Use the tool."}],
                    "encrypted_content": "opaque-replay",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "namespace": "mcp.shell",
                    "name": "echo value",
                    "arguments": '{"value":"FCC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
                {"type": "message", "role": "user", "content": "Continue"},
            ],
            tools=[
                {"type": "web_search", "external_web_access": True},
                {
                    "type": "namespace",
                    "name": "mcp.shell",
                    "tools": [
                        {
                            "type": "function",
                            "name": "echo value",
                            "description": "Echo a value",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                            "strict": True,
                        }
                    ],
                },
                {
                    "type": "custom",
                    "name": "apply patch",
                    "description": "Apply a patch",
                    "format": {"type": "text"},
                },
            ],
            tool_choice={
                "type": "function",
                "namespace": "mcp.shell",
                "name": "echo value",
            },
            parallel_tool_calls=False,
            max_output_tokens=128,
            temperature=0.2,
            top_p=0.9,
            metadata={"trace": "abc"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                    },
                    "strict": True,
                }
            },
        ),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        structured_reasoning_details=True,
    )

    body = translated.body
    namespace_name = "mcp_shell__echo_value"
    namespace_alias = translated.tool_names.encode(namespace_name)
    custom_alias = translated.tool_names.encode("apply patch")
    assert namespace_alias == namespace_name
    assert custom_alias != "apply patch"

    assert body == {
        "model": "provider/model",
        "messages": [
            {
                "role": "system",
                "content": "System rules\n\nDeveloper rules",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AA==",
                            "detail": "low",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "[Earlier reasoning summary]\nUse the tool.",
                "reasoning_content": "",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "opaque-replay"}
                ],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": namespace_name,
                            "arguments": '{"value":"FCC"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "FCC"},
            {"role": "assistant", "content": " "},
            {"role": "user", "content": "Continue"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": namespace_name,
                    "description": "Echo a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply patch",
                    "strict": False,
                    "description": (
                        "Apply a patch\n\nCustom tool input format: unconstrained text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Free-form input for the custom tool.",
                            }
                        },
                        "required": ["input"],
                    },
                },
            },
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": namespace_name},
        },
        "parallel_tool_calls": False,
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "metadata": {"trace": "abc"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                },
                "strict": True,
            },
        },
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ReasoningReplayMode.REASONING_CONTENT, {"reasoning_content": "Think"}),
        (ReasoningReplayMode.REASONING, {"reasoning": "Think"}),
        (ReasoningReplayMode.THINK_TAGS, {"content": "<think>\nThink\n</think>"}),
        (
            ReasoningReplayMode.DISABLED,
            {"content": "[Earlier reasoning]\nThink\n\nAnswer"},
        ),
    ],
)
def test_build_responses_chat_request_uses_provider_reasoning_replay(
    mode: ReasoningReplayMode,
    expected: dict[str, str],
) -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Think"}],
                },
                {"type": "message", "role": "assistant", "content": "Answer"},
            ]
        ),
        reasoning_replay=mode,
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assistant = messages[0]
    assert isinstance(assistant, dict)
    if mode is ReasoningReplayMode.THINK_TAGS:
        assert assistant["content"] == "<think>\nThink\n</think>\n\nAnswer"
    else:
        for key, value in expected.items():
            assert assistant[key] == value
        if mode is ReasoningReplayMode.DISABLED:
            assert assistant == {
                "role": "assistant",
                "content": "[Earlier reasoning]\nThink\n\nAnswer",
            }


def test_build_responses_chat_request_quarantines_one_malformed_call_pair() -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {"role": "user", "content": "Hello"},
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "echo",
                    "arguments": "{",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_bad",
                    "output": "stale",
                },
                {
                    "type": "function_call",
                    "call_id": "call_good",
                    "name": "echo",
                    "arguments": '{"value":"ok"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_good",
                    "output": "ok",
                },
            ]
        ),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    )

    assert translated.body["messages"] == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {
                    "id": "call_good",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"value":"ok"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_good", "content": "ok"},
    ]


@pytest.mark.parametrize(
    "image",
    (
        {"type": "input_image", "file_id": "file_1"},
        {"type": "input_image", "image_url": {"detail": "low"}},
        {"type": "input_image", "image_url": ""},
    ),
)
def test_build_responses_chat_request_rejects_unportable_direct_image(
    image: dict[str, object],
) -> None:
    with pytest.raises(ResponsesConversionError, match="input_image"):
        build_responses_chat_request(
            _request(input=[image]),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )


def test_build_responses_chat_request_preserves_chat_shaped_image_detail() -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "input_image",
                    "image_url": {
                        "url": "https://images.example.test/direct.png",
                        "detail": "low",
                    },
                    "detail": "high",
                }
            ]
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    assert translated.body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://images.example.test/direct.png",
                        "detail": "high",
                    },
                }
            ],
        }
    ]


def test_build_responses_chat_request_relocates_rich_function_output() -> None:
    image_url = "data:image/png;base64,AA=="
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_image",
                    "name": "read_image",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_image",
                    "output": [
                        {"type": "input_text", "text": "before"},
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_file", "file_id": "file_1"},
                        {"type": "input_text", "text": "after"},
                    ],
                },
            ]
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_image",
        "content": "[Image-bearing tool output follows in user content.]",
    }
    assert messages[3]["content"] == [
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_image":',
        },
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": image_url}},
        {
            "type": "text",
            "text": '{"type":"input_file","file_id":"file_1"}',
        },
        {"type": "text", "text": "after"},
    ]
    assert image_url not in messages[1]["content"]


def test_build_responses_chat_request_closes_parallel_tools_before_rich_outputs() -> (
    None
):
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "read_a",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "read_b",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "https://images.example.test/a.png",
                        }
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "https://images.example.test/b.png",
                        }
                    ],
                },
            ]
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
    ]
    assert [message["tool_call_id"] for message in messages[1:3]] == [
        "call_a",
        "call_b",
    ]
    assert messages[4]["content"] == [
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_a":',
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://images.example.test/a.png"},
        },
        {
            "type": "text",
            "text": 'Image-bearing output for tool call "call_b":',
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://images.example.test/b.png"},
        },
    ]


def test_build_responses_chat_request_flushes_rich_output_before_later_input() -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_image",
                    "name": "read_image",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_image",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "https://images.example.test/a.png",
                        }
                    ],
                },
                {"role": "user", "content": "Describe the image."},
            ]
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assert messages[-2]["content"][0]["text"].startswith(
        "Image-bearing output for tool call"
    )
    assert messages[-1] == {"role": "user", "content": "Describe the image."}


def test_build_responses_chat_request_relocates_computer_screenshot() -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "computer_1",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "https://images.example.test/screen.png",
                    },
                    "acknowledged_safety_checks": [{"id": "check_1"}],
                }
            ]
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    assert translated.body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Computer screenshot for call "computer_1":',
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://images.example.test/screen.png"},
                },
            ],
        }
    ]


def test_build_responses_chat_request_keeps_reasoning_before_computer_output() -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Think"}],
                },
                {
                    "type": "computer_call_output",
                    "call_id": "computer_1",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "https://images.example.test/screen.png",
                    },
                },
            ]
        ),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["assistant", "user"]
    assert messages[0]["reasoning_content"] == "Think"


@pytest.mark.parametrize(
    "output",
    (
        {"type": "computer_screenshot", "file_id": "file_1"},
        {"type": "computer_screenshot", "image_url": ""},
        {"type": "not_a_screenshot", "image_url": "https://x/image.png"},
    ),
)
def test_build_responses_chat_request_rejects_unportable_computer_screenshot(
    output: dict[str, object],
) -> None:
    with pytest.raises(ResponsesConversionError, match="computer_call_output"):
        build_responses_chat_request(
            _request(
                input=[
                    {
                        "type": "computer_call_output",
                        "call_id": "computer_1",
                        "output": output,
                    }
                ]
            ),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )


def test_build_responses_chat_request_skips_unsupported_optional_items_and_choice() -> (
    None
):
    translated = build_responses_chat_request(
        _request(
            input=[
                {"type": "computer_screenshot", "image_url": "ignored"},
                {"role": "user", "content": "Hello"},
            ],
            tools=[{"type": "web_search_preview"}],
            tool_choice={"type": "web_search_preview"},
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    assert translated.body == {
        "model": "provider/model",
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_build_responses_chat_request_rejects_no_usable_input() -> None:
    with pytest.raises(ResponsesConversionError, match="usable input"):
        build_responses_chat_request(
            _request(input=[{"type": "computer_screenshot", "image_url": "x"}]),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )


def test_build_responses_chat_request_rejects_message_with_only_file_id_image() -> None:
    with pytest.raises(ResponsesConversionError, match="input_image"):
        build_responses_chat_request(
            _request(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "file_id": "file_not_representable_in_chat",
                            }
                        ],
                    }
                ]
            ),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )


def test_build_responses_chat_request_rejects_colliding_tool_wire_names() -> None:
    with pytest.raises(ResponsesConversionError, match="collide"):
        build_responses_chat_request(
            _request(
                tools=[
                    {
                        "type": "function",
                        "name": "mcp_shell__echo_value",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "namespace",
                        "name": "mcp.shell",
                        "tools": [
                            {
                                "type": "function",
                                "name": "echo value",
                                "parameters": {"type": "object"},
                            }
                        ],
                    },
                ],
                tool_choice={
                    "type": "function",
                    "namespace": "mcp.shell",
                    "name": "echo value",
                },
            ),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )
