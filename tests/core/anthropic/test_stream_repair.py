"""Neutral Anthropic continuation and tool-repair helpers."""

from free_claude_code.core.anthropic.streaming import (
    ToolSchema,
    accept_tool_json_repair,
    continuation_suffix,
    make_response_recovery_body,
    make_text_recovery_body,
    make_tool_repair_body,
)


def test_continuation_suffix_trims_overlap() -> None:
    assert continuation_suffix("hello wor", "world") == "ld"
    assert continuation_suffix("alpha", "alpha beta") == " beta"
    assert continuation_suffix("", "fresh") == "fresh"


def test_tool_json_repair_requires_append_only_schema_valid_json() -> None:
    schemas = {
        "Echo": ToolSchema(
            name="Echo",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        )
    }

    accepted = accept_tool_json_repair(
        '{"message":',
        '"ok"}',
        tool_name="Echo",
        schemas=schemas,
    )
    assert accepted is not None
    assert accepted.suffix == '"ok"}'
    assert accepted.parsed_input == {"message": "ok"}

    assert (
        accept_tool_json_repair(
            '{"message":',
            "1}",
            tool_name="Echo",
            schemas=schemas,
        )
        is None
    )


def test_recovery_bodies_do_not_own_transport_flags() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function"}],
        "tool_choice": "auto",
    }

    text_body = make_text_recovery_body(body, "partial")
    response_body = make_response_recovery_body(body, "partial")
    tool_body = make_tool_repair_body(
        body,
        tool_name="Echo",
        prefix='{"message":',
        input_schema={"type": "object"},
    )

    assert "stream" not in text_body
    assert "stream" not in response_body
    assert "stream" not in tool_body
    assert "tools" not in text_body
    assert response_body["tools"] == body["tools"]
    assert response_body["tool_choice"] == body["tool_choice"]
    assert "tools" not in tool_body
