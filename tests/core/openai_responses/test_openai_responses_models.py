"""Wire-compatibility tests for core-owned OpenAI Responses models."""

import pytest
from pydantic import ValidationError

from free_claude_code.core.openai_responses import OpenAIResponsesRequest


def test_responses_request_preserves_defaults_and_unknown_extensions() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "provider/model",
            "input": "hello",
            "provider_extension": {"enabled": True},
        }
    )

    assert request.stream is True
    assert request.model_extra == {"provider_extension": {"enabled": True}}
    assert request.model_dump(mode="json", exclude_none=True) == {
        "model": "provider/model",
        "input": "hello",
        "stream": True,
        "provider_extension": {"enabled": True},
    }


@pytest.mark.parametrize(("stream", "expected"), [(False, False), (None, None)])
def test_responses_request_preserves_explicit_stream_values(
    stream: bool | None,
    expected: bool | None,
) -> None:
    request = OpenAIResponsesRequest.model_validate(
        {"model": "provider/model", "input": "hello", "stream": stream}
    )

    assert request.stream is expected


def test_responses_request_keeps_permissive_nested_protocol_shapes() -> None:
    payload = {
        "model": "provider/model",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
        ],
        "tools": [
            {
                "type": "namespace",
                "name": "mcp__tools",
                "tools": [{"type": "custom", "name": "apply_patch"}],
            }
        ],
        "tool_choice": {"type": "custom", "name": "apply_patch"},
        "metadata": {"trace": ["a", 1]},
        "reasoning": {"effort": "high", "provider_hint": {"mode": "extended"}},
    }

    request = OpenAIResponsesRequest.model_validate(payload)

    dumped = request.model_dump(mode="json", exclude_none=True)
    for field_name, value in payload.items():
        assert dumped[field_name] == value


def test_responses_request_preserves_existing_pydantic_coercion() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "provider/model",
            "stream": "true",
            "temperature": "0.2",
            "max_output_tokens": "32",
        }
    )

    assert request.stream is True
    assert request.temperature == 0.2
    assert request.max_output_tokens == 32


def test_responses_request_still_requires_model() -> None:
    with pytest.raises(ValidationError):
        OpenAIResponsesRequest.model_validate({"input": "hello"})


def test_responses_request_keeps_openapi_component_name() -> None:
    assert OpenAIResponsesRequest.model_json_schema()["title"] == (
        "OpenAIResponsesRequest"
    )
