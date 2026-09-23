"""Tests for neutral OpenAI-compatible model-list filtering."""

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    RequiredPathValues,
    extract_openai_model_infos,
    extract_tool_capable_model_infos,
)


def test_required_path_values_match_nested_json_scalars() -> None:
    model_infos = extract_openai_model_infos(
        {
            "data": [
                {
                    "id": "included",
                    "metadata": {
                        "kind": "chat",
                        "enabled": True,
                    },
                },
                {
                    "id": "excluded",
                    "metadata": {
                        "kind": "image",
                        "enabled": True,
                    },
                },
            ]
        },
        provider_name="TEST",
        required_path_values=(
            (("metadata", "kind"), ("chat", "code")),
            (("metadata", "enabled"), (True,)),
        ),
    )

    assert model_infos == frozenset({ProviderModelInfo("included")})


@pytest.mark.parametrize(
    ("metadata", "required_path_values", "path"),
    [
        ({}, ((("kind",), ("chat",)),), "kind"),
        ({"kind": True}, ((("kind",), ("chat",)),), "kind"),
        ({"enabled": 1}, ((("enabled",), (True,)),), "enabled"),
    ],
)
def test_required_path_values_reject_missing_or_wrong_scalar_types(
    metadata: dict[str, object],
    required_path_values: RequiredPathValues,
    path: str,
) -> None:
    with pytest.raises(ModelListResponseError, match=path):
        extract_openai_model_infos(
            {"data": [{"id": "model", **metadata}]},
            provider_name="TEST",
            required_path_values=required_path_values,
        )


def test_required_path_values_reject_an_empty_included_set() -> None:
    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        extract_openai_model_infos(
            {"data": [{"id": "image", "type": "image"}]},
            provider_name="TEST",
            required_path_values=((("type",), ("chat",)),),
        )


def test_duplicate_model_ids_preserve_first_validated_capability() -> None:
    model_infos = extract_openai_model_infos(
        {
            "data": [
                {"id": "overlap", "features": ["reasoning"]},
                {"id": "overlap", "features": ["non-reasoning"]},
            ]
        },
        provider_name="TEST",
        tags_field="features",
        non_thinking_tag="non-reasoning",
    )

    assert model_infos == frozenset(
        {ProviderModelInfo("overlap", supports_thinking=True)}
    )


def test_optional_model_capabilities_are_normalized_and_copied_to_aliases() -> None:
    model_infos = extract_openai_model_infos(
        {
            "data": [
                {
                    "id": "vision-reasoning",
                    "aliases": ["latest"],
                    "architecture": {"input_modalities": ["text", "image", "audio"]},
                    "supported_parameters": ["tools", "reasoning"],
                    "limits": {"context": 131072, "output": 16384},
                },
                {
                    "id": "text-only",
                    "aliases": [],
                    "architecture": {"input_modalities": ["text"]},
                    "supported_parameters": ["tools"],
                },
            ]
        },
        provider_name="TEST",
        aliases_field="aliases",
        input_modalities_path=("architecture", "input_modalities"),
        thinking_sequence_path=("supported_parameters",),
        context_window_tokens_path=("limits", "context"),
        max_output_tokens_path=("limits", "output"),
    )

    vision = ProviderModelInfo(
        "vision-reasoning",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=131072,
        max_output_tokens=16384,
    )
    assert model_infos == frozenset(
        {
            vision,
            ProviderModelInfo(
                "latest",
                supports_thinking=True,
                input_modalities=vision.input_modalities,
                context_window_tokens=131072,
                max_output_tokens=16384,
            ),
            ProviderModelInfo(
                "text-only",
                supports_thinking=False,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )


@pytest.mark.parametrize(
    "invalid_value",
    [True, False, 0, -1, 1.5, "4096", None, [], {}],
)
def test_optional_token_limits_require_exact_positive_integers(
    invalid_value: object,
) -> None:
    [info] = extract_openai_model_infos(
        {
            "data": [
                {
                    "id": "model",
                    "limits": {"context": invalid_value, "output": invalid_value},
                }
            ]
        },
        provider_name="TEST",
        context_window_tokens_path=("limits", "context"),
        max_output_tokens_path=("limits", "output"),
    )

    assert info.context_window_tokens is None
    assert info.max_output_tokens is None


def test_optional_token_limits_degrade_independently() -> None:
    [info] = extract_openai_model_infos(
        {
            "data": [
                {
                    "id": "model",
                    "limits": {"context": "bad", "output": 8192},
                }
            ]
        },
        provider_name="TEST",
        context_window_tokens_path=("limits", "context"),
        max_output_tokens_path=("limits", "output"),
    )

    assert info.context_window_tokens is None
    assert info.max_output_tokens == 8192


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("modalities", ["image"], (True, None)),
        ("modalities", "text", (True, None)),
        ("reasoning", "true", (None, frozenset({ModelInputModality.TEXT}))),
        ("reasoning", ["reasoning", 7], (None, frozenset({ModelInputModality.TEXT}))),
    ],
)
def test_malformed_optional_capability_degrades_only_that_field(
    field: str,
    value: object,
    expected: tuple[bool | None, frozenset[ModelInputModality] | None],
) -> None:
    item: dict[str, object] = {
        "id": "model",
        "architecture": {"input_modalities": ["text"]},
        "supported_parameters": ["reasoning"],
    }
    if field == "modalities":
        item["architecture"] = {"input_modalities": value}
    else:
        item["supported_parameters"] = value

    [info] = extract_openai_model_infos(
        {"data": [item]},
        provider_name="TEST",
        input_modalities_path=("architecture", "input_modalities"),
        thinking_sequence_path=("supported_parameters",),
    )

    assert (info.supports_thinking, info.input_modalities) == expected


def test_malformed_optional_boolean_and_tags_are_unknown_not_catalog_failures() -> None:
    boolean_info = next(
        iter(
            extract_openai_model_infos(
                {"data": [{"id": "model", "capabilities": {"reasoning": "yes"}}]},
                provider_name="TEST",
                thinking_boolean_path=("capabilities", "reasoning"),
            )
        )
    )
    tags_info = next(
        iter(
            extract_openai_model_infos(
                {"data": [{"id": "model", "tags": ["reasoning", 7]}]},
                provider_name="TEST",
                tags_field="tags",
            )
        )
    )

    assert boolean_info.supports_thinking is None
    assert tags_info.supports_thinking is None


def test_tool_capable_parser_retains_exact_input_modalities() -> None:
    infos = extract_tool_capable_model_infos(
        {
            "data": [
                {
                    "id": "vision",
                    "supported_parameters": ["tools", "reasoning"],
                    "architecture": {"input_modalities": ["text", "image", "audio"]},
                    "context_length": 262144,
                    "top_provider": {"max_completion_tokens": 32768},
                },
                {
                    "id": "unknown-media",
                    "supported_parameters": ["tool_choice"],
                    "architecture": {"input_modalities": "text"},
                },
            ]
        },
        provider_name="TEST",
    )

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "vision",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=262144,
                max_output_tokens=32768,
            ),
            ProviderModelInfo("unknown-media", supports_thinking=False),
        }
    )
