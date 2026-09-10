import json

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses.errors import ResponsesConversionError
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy

_KEEP_ALL_THINKING_EDIT = {
    "type": "clear_thinking_20251015",
    "keep": "all",
}


def test_build_responses_provider_request_preserves_multiturn_protocol() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "max_tokens": 4096,
            "system": "System instructions",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "summary"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "Calling a tool"},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {"q": "value"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": {"answer": 42},
                        },
                        {"type": "text", "text": "Continue"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            },
                        },
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up a value",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup"},
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
    )

    assert body["model"] == "gpt-test"
    assert body["instructions"] == "System instructions"
    assert body["max_output_tokens"] == 4096
    assert body["stream"] is True
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["input"][0] == {
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "summary"}],
    }
    assert body["input"][1] == {
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "opaque",
    }
    assert body["input"][2]["role"] == "assistant"
    assert body["input"][3] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": json.dumps(
            {"q": "value"}, ensure_ascii=False, separators=(",", ":")
        ),
    }
    assert body["input"][4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"answer": 42}',
    }
    assert body["input"][5]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,aGVsbG8=",
    }


def test_responses_provider_request_canonicalizes_direct_anthropic_image() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "data:IMAGE/PNG;BASE64,aGVs\r\nbG8=",
                            },
                        }
                    ],
                }
            ],
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aGVsbG8=",
                }
            ],
        }
    ]


def test_responses_provider_request_preserves_rich_tool_output_order() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_image",
                            "content": [
                                {"type": "text", "text": "before"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://images.example.test/tool.png",
                                    },
                                },
                                {"type": "document", "title": "reference"},
                                {"type": "text", "text": "after"},
                            ],
                        }
                    ],
                }
            ],
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_image",
            "output": [
                {"type": "input_text", "text": "before"},
                {
                    "type": "input_image",
                    "image_url": "https://images.example.test/tool.png",
                },
                {
                    "type": "input_text",
                    "text": '{"type": "document", "title": "reference"}',
                },
                {"type": "input_text", "text": "after"},
            ],
        }
    ]


@pytest.mark.parametrize(
    "source",
    (
        {"type": "base64", "media_type": "image/png", "data": "SECRET_BAD"},
        {"type": "file", "file_id": "file_1"},
    ),
)
def test_responses_provider_request_rejects_unportable_tool_image(source) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_image",
                            "content": [{"type": "image", "source": source}],
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(ResponsesConversionError) as exc_info:
        build_responses_provider_request(
            request,
            reasoning=ReasoningPolicy.provider_default(),
        )

    assert "SECRET_BAD" not in str(exc_info.value)


def test_build_responses_provider_request_accepts_claude_client_controls() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "context_management": {"edits": [_KEEP_ALL_THINKING_EDIT]},
            "output_config": {"effort": "high"},
        }
    )
    snapshot = request.model_dump()

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )

    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "context_management" not in body
    assert "output_config" not in body
    assert request.model_dump() == snapshot


def test_build_responses_provider_request_materializes_auto_tool_choice() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
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

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.off(),
    )

    assert body["tool_choice"] == "auto"


def test_build_responses_provider_request_omits_choice_without_tools() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.off(),
    )

    assert "tools" not in body
    assert "tool_choice" not in body


def test_build_responses_provider_request_uses_resolved_reasoning_policy() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": {"effort": "low"},
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MAX),
    )

    assert body["reasoning"] == {"effort": "max", "summary": "auto"}


@pytest.mark.parametrize(
    "context_management",
    [
        None,
        {},
        {"edits": []},
        {"edits": [_KEEP_ALL_THINKING_EDIT]},
        {"edits": [_KEEP_ALL_THINKING_EDIT, _KEEP_ALL_THINKING_EDIT]},
    ],
)
def test_build_responses_provider_request_accepts_noop_context_management(
    context_management: dict[str, object] | None,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "context_management": context_management,
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "context_management" not in body


@pytest.mark.parametrize(
    "context_management",
    [
        {"edits": [{"type": "clear_thinking_20251015"}]},
        {
            "edits": [
                {
                    "type": "clear_thinking_20251015",
                    "keep": {"type": "thinking_turns", "value": 2},
                }
            ]
        },
        {"edits": [{"type": "clear_tool_uses_20250919"}]},
        {"edits": [{"type": "unknown_edit", "keep": "all"}]},
        {
            "edits": [
                {
                    **_KEEP_ALL_THINKING_EDIT,
                    "extra": True,
                }
            ]
        },
        {"edits": [], "extra": True},
        {"edits": None},
        {"edits": {}},
        {"edits": "not-a-list"},
    ],
)
def test_build_responses_provider_request_rejects_active_or_malformed_context(
    context_management: dict[str, object],
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "context_management": context_management,
        }
    )

    with pytest.raises(ResponsesConversionError, match="context_management"):
        build_responses_provider_request(
            request,
            reasoning=ReasoningPolicy.provider_default(),
        )


@pytest.mark.parametrize(
    ("effort", "reasoning", "expected"),
    [
        (
            "high",
            ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            {"effort": "high", "summary": "auto"},
        ),
        ("none", ReasoningPolicy.off(), {"effort": "none"}),
        ("future", ReasoningPolicy.provider_default(), None),
    ],
)
def test_build_responses_provider_request_accepts_application_owned_effort(
    effort: str,
    reasoning: ReasoningPolicy,
    expected: dict[str, str] | None,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": {"effort": effort},
        }
    )

    body = build_responses_provider_request(request, reasoning=reasoning)

    assert body.get("reasoning") == expected
    assert "output_config" not in body


@pytest.mark.parametrize(
    ("output_config", "unsupported_path"),
    [
        ({"format": {"type": "json_schema"}}, "output_config.format"),
        (
            {"effort": "high", "format": {"type": "json_schema"}},
            "output_config.format",
        ),
        ({"future_control": True}, "output_config.future_control"),
    ],
)
def test_build_responses_provider_request_rejects_unconsumed_output_config(
    output_config: dict[str, object],
    unsupported_path: str,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": output_config,
        }
    )

    with pytest.raises(ResponsesConversionError, match=unsupported_path):
        build_responses_provider_request(
            request,
            reasoning=ReasoningPolicy.provider_default(),
        )


def test_responses_provider_request_uses_one_portable_tool_alias() -> None:
    original = "mcp__responses_provider__" + "x" * 70
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": original,
                            "input": {"q": "value"},
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
            ],
            "tools": [
                {
                    "name": original,
                    "description": "Long tool",
                    "input_schema": {"type": "object"},
                },
                {"name": "safe_tool", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "tool", "name": original},
        }
    )
    snapshot = request.model_dump()

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    alias = body["tools"][0]["name"]
    assert alias != original
    assert len(alias) <= 64
    assert body["tools"][1]["name"] == "safe_tool"
    assert body["tool_choice"] == {"type": "function", "name": alias}
    function_call = next(
        item for item in body["input"] if item["type"] == "function_call"
    )
    assert function_call["name"] == alias
    assert function_call["call_id"] == "call_1"
    assert function_call["arguments"] == '{"q":"value"}'
    assert request.model_dump() == snapshot


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_sequences", ["stop"]),
        ("top_k", 4),
        ("mcp_servers", [{"name": "server"}]),
        ("extra_body", {"unknown": True}),
    ],
)
def test_build_responses_provider_request_rejects_lossy_fields(
    field: str, value: object
) -> None:
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }

    with pytest.raises(ResponsesConversionError, match=field):
        build_responses_provider_request(
            MessagesRequest.model_validate(payload),
            reasoning=ReasoningPolicy.provider_default(),
        )


def test_build_responses_provider_request_rejects_provider_managed_tools() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "web_search",
                    "type": "web_search_20250305",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )

    with pytest.raises(ResponsesConversionError, match="web_search_20250305"):
        build_responses_provider_request(
            request,
            reasoning=ReasoningPolicy.provider_default(),
        )


def test_build_responses_provider_request_rejects_unknown_request_fields() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "future_field": True,
        }
    )

    with pytest.raises(ResponsesConversionError, match="future_field"):
        build_responses_provider_request(
            request,
            reasoning=ReasoningPolicy.provider_default(),
        )


def test_build_responses_provider_request_sanitizes_tool_patterns() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "save a file"}],
            "tools": [
                {
                    "name": "Artifact",
                    "description": "Save an artifact",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "pattern": (
                                    r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"'
                                    r"\\./[\]]{1,200}"
                                ),
                            },
                            "slug": {
                                "type": "string",
                                "pattern": r"^[a-z-]+$",
                            },
                        },
                    },
                }
            ],
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    parameters = body["tools"][0]["parameters"]
    assert parameters["properties"]["file_name"] == {"type": "string"}
    assert parameters["properties"]["slug"] == {
        "type": "string",
        "pattern": r"^[a-z-]+$",
    }
