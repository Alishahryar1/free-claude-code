import json

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses.errors import ResponsesConversionError
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from tests.wire_payload import object_array, string_payload

_KEEP_ALL_THINKING_EDIT = {
    "type": "clear_thinking_20251015",
    "keep": "all",
}


def test_raw_signed_thinking_keeps_readable_context_outside_opaque_item() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Earlier summary",
                            "signature": "old-cipher",
                        },
                        {"type": "text", "text": "Answer"},
                    ],
                }
            ],
        }
    )
    original = request.model_dump()
    body: JsonObject = build_responses_provider_request(
        request, reasoning=ReasoningPolicy.on()
    )
    opaque, context = object_array(body["input"])[:2]
    assert opaque["encrypted_content"] == "old-cipher"
    assert opaque["summary"] == []
    assert "content" not in opaque
    assert context["role"] == "assistant"
    assert "Earlier summary" in json.dumps(context)
    assert json.dumps(body).count("Earlier summary") == 1
    assert request.model_dump() == original


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

    body: JsonObject = build_responses_provider_request(
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
    assert string_payload(object_array(body["input"])[0]["id"]).startswith("rs_")
    assert string_payload(object_array(body["input"])[1]["id"]).startswith("rs_")
    assert object_array(body["input"])[0]["id"] != object_array(body["input"])[1]["id"]
    assert object_array(body["input"])[0] == {
        "id": object_array(body["input"])[0]["id"],
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "summary"}],
    }
    assert object_array(body["input"])[1] == {
        "id": object_array(body["input"])[1]["id"],
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "opaque",
    }
    assert object_array(body["input"])[2]["role"] == "assistant"
    assert object_array(body["input"])[3] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": json.dumps(
            {"q": "value"}, ensure_ascii=False, separators=(",", ":")
        ),
    }
    assert object_array(body["input"])[4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"answer": 42}',
    }
    assert object_array(object_array(body["input"])[5]["content"])[1] == {
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

    body: JsonObject = build_responses_provider_request(
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
                                {"type": "future_media", "text": "opaque"},
                                {"type": "text", "text": "after"},
                            ],
                        }
                    ],
                }
            ],
        }
    )

    body: JsonObject = build_responses_provider_request(
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
def test_responses_provider_request_rejects_unportable_tool_image(
    source: JsonObject,
) -> None:
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

    body: JsonObject = build_responses_provider_request(
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

    body: JsonObject = build_responses_provider_request(
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

    body: JsonObject = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.off(),
    )

    assert "tools" not in body
    assert "tool_choice" not in body


@pytest.mark.parametrize("choice", ["auto", "any", "none"])
def test_build_responses_provider_request_resolves_explicit_choice_without_tools(
    choice: str,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": choice},
        }
    )

    if choice == "any":
        with pytest.raises(ResponsesConversionError, match="Required tool choice"):
            build_responses_provider_request(request, reasoning=ReasoningPolicy.off())
        return
    body: JsonObject = build_responses_provider_request(
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

    body: JsonObject = build_responses_provider_request(
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

    body: JsonObject = build_responses_provider_request(
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

    body: JsonObject = build_responses_provider_request(request, reasoning=reasoning)

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

    body: JsonObject = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    alias = string_payload(object_array(body["tools"])[0]["name"])
    assert alias != original
    assert len(alias) <= 64
    assert object_array(body["tools"])[1]["name"] == "safe_tool"
    assert body["tool_choice"] == {"type": "function", "name": alias}
    function_call = next(
        item for item in object_array(body["input"]) if item["type"] == "function_call"
    )
    assert function_call["name"] == alias
    assert function_call["call_id"] == "call_1"
    assert function_call["arguments"] == '{"q":"value"}'
    assert request.model_dump() == snapshot


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_config", {"future_control": True}),
        ("future_field", True),
        ("top_k", 4),
        ("mcp_servers", [{"name": "server"}]),
        ("extra_body", {"unknown": True}),
    ],
)
def test_build_responses_provider_request_omits_unsupported_options(
    field: str, value: object
) -> None:
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }

    body = build_responses_provider_request(
        MessagesRequest.model_validate(payload),
        reasoning=ReasoningPolicy.provider_default(),
    )
    assert field not in body


def test_build_responses_provider_request_rejects_stop_sequences() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "stop_sequences": ["stop"],
        }
    )
    with pytest.raises(ResponsesConversionError, match="stop_sequences"):
        build_responses_provider_request(request, reasoning=ReasoningPolicy.off())


def test_build_responses_provider_request_omits_optional_provider_managed_tools() -> (
    None
):
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

    assert "tools" not in build_responses_provider_request(
        request, reasoning=ReasoningPolicy.off()
    )
