import json

import pytest
from fastapi.applications import FastAPI
from fastapi.testclient import TestClient

from free_claude_code.api.dependencies import app_services
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.api_keys import normalize_api_keys
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.settings import Settings
from free_claude_code.core.model_capabilities import ModelInputModality
from tests.api.support import create_test_app, provider_manager_for_app
from tests.wire_payload import (
    array_payload,
    decode_object,
    object_array,
    object_payload,
    response_object,
    response_payload,
    string_payload,
)


def _settings(
    *,
    model: str = "deepseek/deepseek-chat",
    model_fable: str | None = None,
    model_opus: str | None = "open_router/anthropic/claude-opus",
    model_haiku: str | None = "deepseek/deepseek-chat",
    model_fallbacks: tuple[ModelFallback, ...] | None = None,
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=None,
        model_haiku=model_haiku,
        model_fallbacks=model_fallbacks,
        proxy_auth_enabled=False,
        proxy_auth_token="freecc",
        deepseek_api_keys=normalize_api_keys("deepseek-key"),
        open_router_api_keys=normalize_api_keys("open-router-key"),
        wafer_api_keys=normalize_api_keys("wafer-key"),
    )


def _cache_models(app: FastAPI, provider_id: str, *model_ids: str) -> None:
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {ProviderModelInfo(model_id) for model_id in model_ids},
    )


@pytest.mark.parametrize("view", ["claude", "claude-desktop"])
@pytest.mark.parametrize("overrides", [False, True])
def test_claude_shortcuts_are_exactly_four_current_versions_and_route_by_tier(
    view: str, overrides: bool
):
    expected = {
        "claude-fable-5-2": "Claude Fable 5.2",
        "claude-opus-5-5": "Claude Opus 5.5",
        "claude-sonnet-5-5": "Claude Sonnet 5.5",
        "claude-haiku-5-5": "Claude Haiku 5.5",
    }
    settings = _settings().model_copy(
        update={
            f"model_{tier}": f"deepseek/{tier}-target" if overrides else None
            for tier in ("fable", "opus", "sonnet", "haiku")
        }
    )
    response = TestClient(create_test_app(settings)).get(f"/v1/models?view={view}")
    assert response.status_code == 200
    shortcuts = {
        row["id"]: object_payload(row)["display_name"]
        for row in object_array(response_object(response)["data"])
        if "/" not in string_payload(object_payload(row)["id"])
    }
    assert list(shortcuts.items()) == list(expected.items())
    router = ModelRouter(settings)
    for model_id in shortcuts:
        tier = string_payload(model_id).split("-")[1]
        target = f"deepseek/{tier}-target" if overrides else settings.model
        assert (
            router.resolve(string_payload(model_id)).primary.provider_model_ref
            == target
        )


@pytest.mark.parametrize("view", ["messages", "responses"])
def test_context_fallback_round_trip_preserves_reported_capabilities(view: str):
    from free_claude_code.application.model_catalog import read_model_catalog
    from free_claude_code.cli.launchers.catalog_http import model_catalog_from_response
    from free_claude_code.cli.launchers.opencode_config import build_opencode_config

    app = create_test_app(_settings(model_opus=None))
    manager = provider_manager_for_app(app)
    manager.cache_model_infos(
        "deepseek", {ProviderModelInfo("known", context_window_tokens=8000)}
    )
    before = read_model_catalog(manager)
    payload = response_object(TestClient(app).get(f"/v1/models?view={view}"))
    rows = {object_payload(row)["id"]: row for row in array_payload(payload["data"])}
    assert object_payload(rows["deepseek/deepseek-chat"])["contextWindow"] == 200000
    assert (
        object_payload(rows["deepseek/deepseek-chat"])["contextWindowSource"]
        == "default"
    )
    assert object_payload(rows["deepseek/known"])["contextWindow"] == 8000
    assert object_payload(rows["deepseek/known"])["contextWindowSource"] == "provider"
    decoded = model_catalog_from_response(payload)
    assert decoded == before
    config = build_opencode_config(
        decoded.models,
        default_model_id=decoded.default_model_id,
        proxy_root_url="http://localhost:8082",
    )
    models = object_payload(
        object_payload(decode_object(json.dumps(config.file))["providers"])[
            "free-claude-code"
        ]
    )["models"]
    assert object_payload(object_payload(models)["deepseek/deepseek-chat"])[
        "limit"
    ] == {"context": 200000}
    assert object_payload(object_payload(models)["deepseek/known"])["limit"] == {
        "context": 8000,
        "output": 2000,
    }


@pytest.mark.parametrize("view", ["messages", "responses"])
def test_sorted_catalog_retains_an_explicit_no_thinking_server_default(view: str):
    settings = _settings(
        model="open_router/Zulu",
        model_opus=None,
        model_haiku=None,
        model_fallbacks=(
            ModelFallback(model="wafer/z-fallback"),
            ModelFallback(model="deepseek/a-fallback"),
        ),
    )
    app = create_test_app(settings)
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("Zulu", supports_thinking=False),
            ProviderModelInfo("apple"),
        },
    )
    payload = response_object(TestClient(app).get(f"/v1/models?view={view}"))
    assert [
        object_payload(row)["provider_model_ref"]
        for row in array_payload(payload["data"])
    ] == [
        "deepseek/a-fallback",
        "open_router/apple",
        "open_router/Zulu",
        "wafer/z-fallback",
    ]
    assert payload["default_model_id"] == "claude-3-freecc-no-thinking/open_router/Zulu"
    assert payload["first_id"] == "deepseek/a-fallback"
    assert payload["last_id"] == "wafer/z-fallback"
    assert settings.model_fallbacks == (
        ModelFallback(model="wafer/z-fallback"),
        ModelFallback(model="deepseek/a-fallback"),
    )


def test_models_list_includes_configured_refs_cached_provider_models_and_aliases():
    app = create_test_app(_settings())
    _cache_models(app, "deepseek", "deepseek-chat")
    _cache_models(
        app,
        "open_router",
        "meta/llama-3.3",
        "anthropic/claude-opus",
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    data = response_object(response)
    ids = [object_payload(item)["id"] for item in array_payload(data["data"])]
    assert ids[0] == "claude-fable-5-2"
    assert [model_id for model_id in ids if "/" in string_payload(model_id)] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
        "anthropic/open_router/meta/llama-3.3",
        "claude-3-freecc-no-thinking/open_router/meta/llama-3.3",
    ]
    assert ids.count("anthropic/deepseek/deepseek-chat") == 1
    assert ids.count("anthropic/open_router/anthropic/claude-opus") == 1
    display_names = {
        object_payload(item)["id"]: object_payload(item)["display_name"]
        for item in array_payload(data["data"])
    }
    assert (
        display_names["anthropic/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3"
    )
    assert (
        display_names["claude-3-freecc-no-thinking/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3 (no thinking)"
    )
    assert "claude-sonnet-5-5" in ids
    assert "claude-fable-5-2" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]
    assert data["has_more"] is False
    assert all(
        "provider_model_ref" not in object_payload(item)
        and "apiBackend" not in object_payload(item)
        for item in array_payload(data["data"])
    )


def test_claude_models_use_custom_provider_display_name_without_changing_ids():
    provider_id = "custom_f526dba2aa0a472f9c506ba236056d57"
    model_ref = f"{provider_id}/deepseek-v4.1-flash"
    definition = CustomProviderDefinition(
        provider_id=provider_id,
        display_name="Phoenix",
        base_url="https://gateway.example/v1",
    )
    settings = _settings(model=model_ref, model_opus=None, model_haiku=None).model_copy(
        update={"custom_providers": (definition,)}
    )
    app = create_test_app(settings)
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {
            ProviderModelInfo("deepseek-v4.1-flash", supports_thinking=True),
            ProviderModelInfo("deepseek-plain", supports_thinking=False),
        },
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    rows = {
        object_payload(row)["id"]: row
        for row in array_payload(response_object(response)["data"])
    }
    assert object_payload(rows[f"anthropic/{model_ref}"])["display_name"] == (
        "Phoenix/deepseek-v4.1-flash"
    )
    assert object_payload(rows[f"claude-3-freecc-no-thinking/{model_ref}"])[
        "display_name"
    ] == ("Phoenix/deepseek-v4.1-flash (no thinking)")
    plain_ref = f"{provider_id}/deepseek-plain"
    assert object_payload(rows[f"claude-3-freecc-no-thinking/{plain_ref}"])[
        "display_name"
    ] == ("Phoenix/deepseek-plain (no thinking)")


def test_models_list_uses_thinking_metadata_for_cached_models():
    app = create_test_app(_settings(model_opus=None))
    manager = provider_manager_for_app(app)
    _cache_models(app, "deepseek", "deepseek-chat")
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [
        object_payload(item)["id"]
        for item in array_payload(response_object(response)["data"])
    ]
    assert "anthropic/open_router/reasoning-model" in ids
    assert "claude-3-freecc-no-thinking/open_router/reasoning-model" in ids
    assert "anthropic/open_router/plain-model" not in ids
    assert "claude-3-freecc-no-thinking/open_router/plain-model" in ids


def test_models_list_uses_cached_metadata_for_configured_refs():
    app = create_test_app(
        _settings(
            model="open_router/plain-model",
            model_opus=None,
            model_haiku=None,
        )
    )
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {ProviderModelInfo("plain-model", supports_thinking=False)},
    )

    response = TestClient(app).get("/v1/models")

    ids = [
        object_payload(item)["id"]
        for item in array_payload(response_object(response)["data"])
    ]
    assert "anthropic/open_router/plain-model" not in ids
    assert ids[-1] == "claude-3-freecc-no-thinking/open_router/plain-model"


def test_models_list_includes_cached_wafer_models():
    app = create_test_app(
        _settings(
            model="wafer/DeepSeek-V4-Pro",
            model_opus=None,
            model_haiku=None,
        )
    )
    _cache_models(app, "wafer", "DeepSeek-V4-Pro", "MiniMax-M2.7")

    response = TestClient(app).get("/v1/models")

    ids = [
        object_payload(item)["id"]
        for item in array_payload(response_object(response)["data"])
    ]
    assert "anthropic/wafer/DeepSeek-V4-Pro" in ids
    assert "claude-3-freecc-no-thinking/wafer/DeepSeek-V4-Pro" in ids
    assert "anthropic/wafer/MiniMax-M2.7" in ids
    assert "claude-3-freecc-no-thinking/wafer/MiniMax-M2.7" in ids


def test_models_list_includes_fallback_refs_without_cached_discovery():
    app = create_test_app(
        _settings(
            model_opus=None,
            model_haiku=None,
            model_fallbacks=(ModelFallback(model="wafer/DeepSeek-V4-Pro"),),
        )
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [
        object_payload(item)["id"]
        for item in array_payload(response_object(response)["data"])
    ]
    assert "anthropic/wafer/DeepSeek-V4-Pro" in ids
    assert "claude-3-freecc-no-thinking/wafer/DeepSeek-V4-Pro" in ids


def test_models_list_works_with_empty_discovery_catalog():
    app = create_test_app(_settings())

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [
        object_payload(item)["id"]
        for item in array_payload(response_object(response)["data"])
    ]
    assert ids[0] == "claude-fable-5-2"
    assert [model_id for model_id in ids if "/" in string_payload(model_id)] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
    ]
    assert "claude-sonnet-5-5" in ids


def test_direct_model_views_exclude_claude_aliases_and_duplicate_variants():
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    manager = provider_manager_for_app(app)
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    messages = response_object(TestClient(app).get("/v1/models?view=messages"))
    responses = response_object(TestClient(app).get("/v1/models?view=responses"))

    assert [object_payload(row)["id"] for row in array_payload(messages["data"])] == [
        "deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/open_router/plain-model",
        "open_router/reasoning-model",
    ]
    assert "claude-sonnet-5-5" not in {
        object_payload(row)["id"] for row in array_payload(messages["data"])
    }
    assert all(
        "apiBackend" not in object_payload(row)
        for row in array_payload(messages["data"])
    )
    assert array_payload(responses["data"])[0] == {
        "object": "model",
        "created": 0,
        "owned_by": "free-claude-code",
        "created_at": "1970-01-01T00:00:00Z",
        "display_name": "deepseek/deepseek-chat",
        "id": "deepseek/deepseek-chat",
        "type": "model",
        "provider_model_ref": "deepseek/deepseek-chat",
        "apiBackend": "responses",
        "maxRetries": 0,
        "contextWindow": 200000,
        "contextWindowSource": "default",
        "supportsReasoningEffort": True,
        "reasoningEfforts": [
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ],
        "inferenceIdleTimeoutSecs": 2_147_483,
    }
    plain = array_payload(responses["data"])[1]
    assert object_payload(plain)["supportsReasoningEffort"] is False
    assert "reasoningEfforts" not in object_payload(plain)


def test_direct_model_views_serialize_known_capabilities_and_omit_unknowns():
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo(
                "vision-reasoning",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131072,
                max_output_tokens=8192,
            ),
            ProviderModelInfo(
                "text-only",
                supports_thinking=False,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=65536,
            ),
            ProviderModelInfo("unknown"),
        },
    )
    client = TestClient(app)

    messages = {
        object_payload(row)["provider_model_ref"]: row
        for row in array_payload(
            response_object(client.get("/v1/models?view=messages"))["data"]
        )
    }
    responses = {
        object_payload(row)["provider_model_ref"]: row
        for row in array_payload(
            response_object(client.get("/v1/models?view=responses"))["data"]
        )
    }

    assert (
        object_payload(messages["open_router/vision-reasoning"])["supportsReasoning"]
        is True
    )
    assert object_payload(messages["open_router/vision-reasoning"])[
        "inputModalities"
    ] == [
        "text",
        "image",
    ]
    assert (
        object_payload(messages["open_router/vision-reasoning"])["contextWindow"]
        == 131072
    )
    assert (
        object_payload(messages["open_router/vision-reasoning"])["maxCompletionTokens"]
        == 8192
    )
    assert (
        object_payload(messages["open_router/text-only"])["supportsReasoning"] is False
    )
    assert object_payload(messages["open_router/text-only"])["inputModalities"] == [
        "text"
    ]
    assert object_payload(messages["open_router/text-only"])["contextWindow"] == 65536
    assert "maxCompletionTokens" not in object_payload(
        messages["open_router/text-only"]
    )
    assert "supportsReasoning" not in object_payload(messages["open_router/unknown"])
    assert "inputModalities" not in object_payload(messages["open_router/unknown"])
    assert object_payload(messages["open_router/unknown"])["contextWindow"] == 200000
    assert (
        object_payload(messages["open_router/unknown"])["contextWindowSource"]
        == "default"
    )
    assert "maxCompletionTokens" not in object_payload(messages["open_router/unknown"])
    assert (
        object_payload(responses["open_router/text-only"])["supportsReasoningEffort"]
        is False
    )
    assert "reasoningEfforts" not in object_payload(responses["open_router/text-only"])
    assert (
        object_payload(responses["open_router/vision-reasoning"])["contextWindow"]
        == 131072
    )
    assert (
        object_payload(responses["open_router/vision-reasoning"])["maxCompletionTokens"]
        == 8192
    )


def test_claude_model_view_does_not_expose_capability_fields():
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo(
                "vision-reasoning",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131072,
                max_output_tokens=8192,
            )
        },
    )

    rows = response_object(TestClient(app).get("/v1/models"))["data"]

    assert all(
        "supportsReasoning" not in object_payload(row)
        and "inputModalities" not in object_payload(row)
        and "contextWindow" not in object_payload(row)
        and "maxCompletionTokens" not in object_payload(row)
        for row in array_payload(rows)
    )


def test_muse_model_catalog_marks_every_routable_model_visible():
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    manager = provider_manager_for_app(app)
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )
    client = TestClient(app)

    responses = client.get("/v1/models?view=responses")
    muse = client.get("/muse-code/models?view=claude")

    assert responses.status_code == 200
    assert muse.status_code == 200
    assert all(
        "metadata" not in object_payload(row)
        for row in array_payload(response_object(responses)["data"])
    )
    for row in array_payload(response_object(muse)["data"]):
        metadata = object_payload(object_payload(row)["metadata"])["muse-code"]
        assert object_payload(metadata)["is_hidden"] is False
        assert object_payload(metadata)["name"] == object_payload(row)["display_name"]
        assert object_payload(metadata)["limit"] == {"context": 200000, "output": 81920}
        original = {
            key: value
            for key, value in object_payload(row).items()
            if key != "metadata"
        }
        assert original in array_payload(response_object(responses)["data"])
    assert [
        object_payload(row)["id"]
        for row in array_payload(response_object(muse)["data"])
    ] == [
        "deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/open_router/plain-model",
        "open_router/reasoning-model",
    ]


def test_muse_catalog_uses_native_metadata_for_known_limits_and_reasoning():
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo(
                "reasoning-model",
                supports_thinking=True,
                context_window_tokens=131072,
                max_output_tokens=8192,
            ),
            ProviderModelInfo(
                "plain-model",
                supports_thinking=False,
                context_window_tokens=65536,
            ),
        },
    )
    rows = {
        object_payload(row)["provider_model_ref"]: object_payload(
            object_payload(row)["metadata"]
        )["muse-code"]
        for row in array_payload(
            response_object(TestClient(app).get("/muse-code/models"))["data"]
        )
    }

    assert rows["open_router/reasoning-model"] == {
        "name": "open_router/reasoning-model",
        "is_hidden": False,
        "reasoning": True,
        "limit": {"context": 131072, "output": 8192},
    }
    assert rows["open_router/plain-model"] == {
        "name": "open_router/plain-model (no thinking)",
        "is_hidden": False,
        "reasoning": False,
        "limit": {"context": 65536, "output": 65536},
    }
    assert rows["deepseek/deepseek-chat"] == {
        "name": "deepseek/deepseek-chat",
        "is_hidden": False,
        "limit": {"context": 200000, "output": 81920},
    }


@pytest.mark.parametrize(
    ("context_tokens", "output_tokens", "expected_context", "expected_output"),
    [
        (None, None, 200000, 81920),
        (131072, 8192, 131072, 8192),
        (65536, None, 65536, 65536),
        (None, 4096, 200000, 4096),
        (65536, 49152, 65536, 49152),
    ],
)
def test_muse_limits_are_complete_without_changing_reported_capabilities(
    context_tokens: int | None,
    output_tokens: int | None,
    expected_context: int,
    expected_output: int,
):
    from free_claude_code.application.model_catalog import read_model_catalog

    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    manager = provider_manager_for_app(app)
    manager.cache_model_infos(
        "deepseek",
        {
            ProviderModelInfo(
                "deepseek-chat",
                context_window_tokens=context_tokens,
                max_output_tokens=output_tokens,
            )
        },
    )
    before = read_model_catalog(manager)
    client = TestClient(app)
    response_row = array_payload(
        response_object(client.get("/v1/models?view=responses"))["data"]
    )[0]
    muse_row = array_payload(response_object(client.get("/muse-code/models"))["data"])[
        0
    ]

    assert object_payload(
        object_payload(object_payload(muse_row).pop("metadata"))["muse-code"]
    )["limit"] == {
        "context": expected_context,
        "output": expected_output,
    }
    assert muse_row == response_row
    assert ("maxCompletionTokens" in object_payload(response_row)) is (
        output_tokens is not None
    )
    assert read_model_catalog(manager) == before
    assert before.models[0].max_output_tokens == output_tokens


def test_unknown_model_view_is_rejected():
    response = TestClient(create_test_app(_settings())).get("/v1/models?view=other")

    assert response.status_code == 422


@pytest.mark.parametrize("view", ["messages", "responses"])
def test_real_json_catalog_agrees_with_application_identity_and_known_metadata(
    view: str,
):
    from free_claude_code.application.model_catalog import read_model_catalog
    from free_claude_code.cli.launchers.catalog_http import model_catalog_from_response

    app = create_test_app(
        _settings(model="open_router/Zulu", model_opus=None, model_haiku=None)
    )
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo(
                "Zulu", supports_thinking=False, context_window_tokens=32000
            ),
            ProviderModelInfo(
                "vendor/vision ",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                max_output_tokens=4096,
            ),
        },
    )
    expected = read_model_catalog(app_services(app).requests)
    payload = response_payload(TestClient(app).get(f"/v1/models?view={view}"))
    assert model_catalog_from_response(object_payload(payload)) == expected
