import pytest

from free_claude_code.providers.model_listing import (
    extract_openai_model_infos,
    extract_tool_capable_model_infos,
)


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        ({"mandatory": True}, "required"),
        ({"supported_efforts": ["none", "high"]}, "optional"),
        ({"mandatory": True, "supported_efforts": ["none"]}, "unknown"),
        ({"mandatory": "true"}, "unknown"),
        ({"mandatory": False}, "unknown"),
        (None, "unknown"),
    ],
)
def test_explicit_gateway_reasoning_capability(reasoning, expected):
    infos = extract_tool_capable_model_infos(
        {
            "data": [
                {
                    "id": "route",
                    "supported_parameters": ["tools"],
                    "reasoning": reasoning,
                }
            ]
        },
        provider_name="GATEWAY",
    )
    assert next(iter(infos)).reasoning_capability.value == expected


@pytest.mark.parametrize(
    ("payload", "options", "expected"),
    [
        ({"reasoning": False}, {"thinking_boolean_path": ("reasoning",)}, "none"),
        ({"reasoning": True}, {"thinking_boolean_path": ("reasoning",)}, "unknown"),
        (
            {"tags": ["non-reasoning"]},
            {"tags_field": "tags", "non_thinking_tag": "non-reasoning"},
            "none",
        ),
        (
            {"tags": ["non-reasoning", "reasoning"]},
            {"tags_field": "tags", "non_thinking_tag": "non-reasoning"},
            "unknown",
        ),
        (
            {"supported_parameters": ["tools"]},
            {"thinking_sequence_path": ("supported_parameters",)},
            "unknown",
        ),
    ],
)
def test_intrinsic_capability_is_distinct_from_available_controls(
    payload, options, expected
):
    infos = extract_openai_model_infos(
        {"data": [{"id": "route", **payload}]}, provider_name="TEST", **options
    )
    assert next(iter(infos)).reasoning_capability.value == expected
