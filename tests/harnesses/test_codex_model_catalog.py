from collections.abc import Mapping
from typing import Any, cast

from free_claude_code.harnesses.codex_model_catalog import (
    build_codex_model_catalog,
)
from free_claude_code.harnesses.model_catalog import client_models_from_response


def _models_payload(*model_ids: str) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": model_id,
                "provider_model_ref": (
                    model_id.removeprefix("claude-3-freecc-no-thinking/")
                ),
                "display_name": model_id,
            }
            for model_id in model_ids
        ]
    }


def _catalog_models(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    models = catalog["models"]
    assert isinstance(models, list)
    catalog_models: list[Mapping[str, Any]] = []
    for model in models:
        assert isinstance(model, Mapping)
        catalog_models.append(cast(Mapping[str, Any], model))
    return catalog_models


def _slugs(catalog: Mapping[str, Any]) -> list[str]:
    slugs: list[str] = []
    for model in _catalog_models(catalog):
        slug = model["slug"]
        assert isinstance(slug, str)
        slugs.append(slug)
    return slugs


def test_codex_catalog_uses_direct_configured_and_cached_model_slugs() -> None:
    catalog = build_codex_model_catalog(
        client_models_from_response(
            _models_payload(
                "nvidia_nim/nvidia/nemotron-3-super",
                "open_router/meta-llama/llama-3.3-70b",
            )
        )
    )

    assert _slugs(catalog) == [
        "nvidia_nim/nvidia/nemotron-3-super",
        "open_router/meta-llama/llama-3.3-70b",
    ]
    model = _catalog_models(catalog)[0]
    assert {
        "slug",
        "display_name",
        "description",
        "default_reasoning_level",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "priority",
        "additional_speed_tiers",
        "service_tiers",
    } <= set(model)


def test_codex_catalog_ignores_rows_without_direct_provider_identity() -> None:
    catalog = build_codex_model_catalog(
        client_models_from_response(
            {
                "data": [
                    {"id": "claude-opus-4-20250514"},
                    {
                        "id": "nvidia_nim/provider-model",
                        "provider_model_ref": "nvidia_nim/provider-model",
                        "display_name": "NIM",
                    },
                ]
            }
        )
    )

    assert _slugs(catalog) == ["nvidia_nim/provider-model"]


def test_codex_catalog_deduplicates_direct_wire_ids() -> None:
    catalog = build_codex_model_catalog(
        client_models_from_response(
            _models_payload(
                "nvidia_nim/provider-model",
                "nvidia_nim/provider-model",
            )
        )
    )

    assert _slugs(catalog) == ["nvidia_nim/provider-model"]


def test_codex_catalog_preserves_no_thinking_only_entries_for_routing() -> None:
    catalog = build_codex_model_catalog(
        client_models_from_response(
            _models_payload("claude-3-freecc-no-thinking/open_router/plain-model")
        )
    )

    assert _slugs(catalog) == ["claude-3-freecc-no-thinking/open_router/plain-model"]


def test_codex_catalog_ordering_and_priorities_are_deterministic() -> None:
    catalog = build_codex_model_catalog(
        client_models_from_response(
            _models_payload(
                "anthropic/gemini/models/gemini-test",
                "nvidia_nim/nvidia/test",
                "anthropic/gemini/models/gemini-test",
                "open_router/provider/test",
            )
        )
    )

    models = _catalog_models(catalog)
    assert _slugs(catalog) == [
        "anthropic/gemini/models/gemini-test",
        "nvidia_nim/nvidia/test",
        "open_router/provider/test",
    ]
    assert [model["priority"] for model in models] == [0, 1, 2]


def test_codex_catalog_accepts_direct_provider_slugs_without_a_provider_registry() -> (
    None
):
    catalog = build_codex_model_catalog(
        client_models_from_response(
            _models_payload(
                "nvidia_nim/provider-model",
                "future_provider/provider-model",
            )
        )
    )

    assert _slugs(catalog) == [
        "nvidia_nim/provider-model",
        "future_provider/provider-model",
    ]


def test_codex_catalog_projects_known_capabilities_and_preserves_unknown_defaults() -> (
    None
):
    catalog = build_codex_model_catalog(
        client_models_from_response(
            {
                "data": [
                    {
                        "id": "provider/vision-reasoning",
                        "provider_model_ref": "provider/vision-reasoning",
                        "supportsReasoning": True,
                        "inputModalities": ["text", "image"],
                        "contextWindow": 131072,
                        "maxCompletionTokens": 8192,
                    },
                    {
                        "id": "claude-3-freecc-no-thinking/provider/text-only",
                        "provider_model_ref": "provider/text-only",
                        "supportsReasoning": False,
                        "inputModalities": ["text"],
                        "maxCompletionTokens": 4096,
                    },
                    {
                        "id": "provider/unknown",
                        "provider_model_ref": "provider/unknown",
                    },
                ]
            }
        )
    )

    vision, text_only, unknown = _catalog_models(catalog)
    assert vision["input_modalities"] == ["text", "image"]
    assert vision["default_reasoning_level"] == "medium"
    assert vision["supported_reasoning_levels"]
    assert vision["supports_reasoning_summaries"] is True
    assert vision["default_reasoning_summary"] == "none"
    assert vision["context_window"] == 131072
    assert vision["max_context_window"] == 131072

    assert text_only["input_modalities"] == ["text"]
    assert "default_reasoning_level" not in text_only
    assert text_only["supported_reasoning_levels"] == []
    assert text_only["supports_reasoning_summaries"] is False
    assert "default_reasoning_summary" not in text_only
    assert text_only["context_window"] == 200000
    assert text_only["max_context_window"] == 200000

    assert unknown["input_modalities"] == ["text"]
    assert unknown["default_reasoning_level"] == "medium"
    assert unknown["supported_reasoning_levels"]
    assert unknown["supports_reasoning_summaries"] is True
    assert unknown["default_reasoning_summary"] == "none"
    assert unknown["context_window"] == 200000
    assert unknown["max_context_window"] == 200000
