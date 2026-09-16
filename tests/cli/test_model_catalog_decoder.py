import pytest

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.cli.launchers.catalog_http import catalog_models_from_response
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality


def test_client_models_project_nested_direct_refs_in_source_order() -> None:
    assert catalog_models_from_response(
        {
            "data": [
                {
                    "id": "nvidia_nim/nvidia/nemotron-3-super",
                    "provider_model_ref": "nvidia_nim/nvidia/nemotron-3-super",
                    "display_name": "Display 0",
                },
                {
                    "id": "open_router/meta-llama/llama-3.3-70b",
                    "provider_model_ref": "open_router/meta-llama/llama-3.3-70b",
                    "display_name": "Display 1",
                },
            ]
        }
    ) == (
        CatalogModel(
            wire_slug="nvidia_nim/nvidia/nemotron-3-super",
            provider_model_ref="nvidia_nim/nvidia/nemotron-3-super",
            display_name="Display 0",
            supports_reasoning=None,
        ),
        CatalogModel(
            wire_slug="open_router/meta-llama/llama-3.3-70b",
            provider_model_ref="open_router/meta-llama/llama-3.3-70b",
            display_name="Display 1",
            supports_reasoning=None,
        ),
    )


def test_client_models_keep_no_thinking_direct_route() -> None:
    models = catalog_models_from_response(
        {
            "data": [
                {
                    "id": "claude-3-freecc-no-thinking/nvidia_nim/provider-model",
                    "provider_model_ref": "nvidia_nim/provider-model",
                    "display_name": "Display 0",
                    "supportsReasoning": False,
                }
            ]
        }
    )

    assert models == (
        CatalogModel(
            wire_slug="claude-3-freecc-no-thinking/nvidia_nim/provider-model",
            provider_model_ref="nvidia_nim/provider-model",
            display_name="Display 0",
            supports_reasoning=False,
        ),
    )


def test_client_models_keep_no_thinking_only_route() -> None:
    assert catalog_models_from_response(
        {
            "data": [
                {
                    "id": "claude-3-freecc-no-thinking/open_router/plain-model",
                    "provider_model_ref": "open_router/plain-model",
                    "display_name": "Display 0",
                    "supportsReasoning": False,
                }
            ]
        }
    ) == (
        CatalogModel(
            wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
            provider_model_ref="open_router/plain-model",
            display_name="Display 0",
            supports_reasoning=False,
        ),
    )


def test_client_models_parse_capabilities_without_deriving_reasoning_from_slug() -> (
    None
):
    models = catalog_models_from_response(
        {
            "data": [
                {
                    "id": "provider/reasoning",
                    "provider_model_ref": "provider/reasoning",
                    "supportsReasoning": False,
                    "inputModalities": ["text"],
                    "contextWindow": 131072,
                    "maxCompletionTokens": 8192,
                },
                {
                    "id": "claude-3-freecc-no-thinking/provider/unknown",
                    "provider_model_ref": "provider/unknown",
                    "supportsReasoning": "not-a-bool",
                    "inputModalities": ["text", "image"],
                    "contextWindow": 0,
                    "maxCompletionTokens": "8192",
                },
                {
                    "id": "provider/malformed-media",
                    "provider_model_ref": "provider/malformed-media",
                    "supportsReasoning": True,
                    "inputModalities": ["text", 7],
                },
            ]
        }
    )

    assert [
        (
            model.supports_reasoning,
            model.input_modalities,
            model.context_window_tokens,
            model.max_output_tokens,
        )
        for model in models
    ] == [
        (False, frozenset({ModelInputModality.TEXT}), 131072, 8192),
        (
            None,
            frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
            None,
            None,
        ),
        (True, None, None, None),
    ]


def test_client_models_ignore_compatibility_unknown_and_malformed_entries() -> None:
    payload: JsonObject = {
        "data": [
            {"id": "claude-opus-4-20250514"},
            {"id": "unknown/model", "provider_model_ref": 123},
            {"id": "   ", "provider_model_ref": "open_router/model"},
            {"id": "open_router/model", "provider_model_ref": "open_router/"},
            {"id": 123},
            "not-an-object",
        ]
    }

    assert catalog_models_from_response(payload) == ()
    assert catalog_models_from_response({"data": "not-a-list"}) == ()


def test_client_models_deduplicate_wire_slugs_deterministically() -> None:
    models = catalog_models_from_response(
        {
            "data": [
                {
                    "id": "gemini/models/gemini-test",
                    "provider_model_ref": "gemini/models/gemini-test",
                    "display_name": "Display 0",
                },
                {
                    "id": "gemini/models/gemini-test",
                    "provider_model_ref": "gemini/models/gemini-test",
                    "display_name": "Display 1",
                },
                {
                    "id": "open_router/provider/test",
                    "provider_model_ref": "open_router/provider/test",
                    "display_name": "Display 2",
                },
            ]
        }
    )

    assert [model.wire_slug for model in models] == [
        "gemini/models/gemini-test",
        "open_router/provider/test",
    ]
    assert models[0].display_name == "Display 0"


@pytest.mark.parametrize("default", [None, "", 1, "provider/missing"])
def test_catalog_rejects_missing_or_unroutable_default(default) -> None:
    from free_claude_code.cli.launchers.catalog_http import model_catalog_from_response

    payload = {
        "data": [{"id": "provider/model", "provider_model_ref": "provider/model"}]
    }
    if default is not None:
        payload["default_model_id"] = default
    with pytest.raises(ValueError, match="default"):
        model_catalog_from_response(payload)


def test_http_catalog_preserves_nonblank_identity_and_selects_default_by_id() -> None:
    from free_claude_code.cli.launchers.catalog_http import model_catalog_from_response

    payload = {
        "default_model_id": "provider/padded ",
        "data": [
            {"id": "provider/padded", "provider_model_ref": "provider/padded"},
            {
                "id": "provider/padded ",
                "provider_model_ref": "provider/padded ",
                "display_name": " Padded ",
            },
        ],
    }
    catalog = model_catalog_from_response(payload)
    assert [model.wire_slug for model in catalog.models] == [
        "provider/padded",
        "provider/padded ",
    ]
    assert catalog.default_model_id == catalog.models[1].wire_slug
    assert catalog.models[1].display_name == " Padded "
