import math

from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelPricing,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.model_capabilities import ModelInputModality
from tests.api.support import create_test_app, provider_manager_for_app


def _settings(
    *,
    model: str = "deepseek/deepseek-chat",
    model_fable: str | None = None,
    model_opus: str | None = "open_router/anthropic/claude-opus",
    model_haiku: str | None = "deepseek/deepseek-chat",
    model_fallbacks: tuple[str, ...] | None = None,
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
        deepseek_api_key="deepseek-key",
        open_router_api_key="open-router-key",
        wafer_api_key="wafer-key",
    )


def _cache_models(app, provider_id: str, *model_ids: str) -> None:
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {ProviderModelInfo(model_id) for model_id in model_ids},
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
    data = response.json()
    ids = [item["id"] for item in data["data"]]
    assert ids[:6] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
        "anthropic/open_router/meta/llama-3.3",
        "claude-3-freecc-no-thinking/open_router/meta/llama-3.3",
    ]
    assert ids.count("anthropic/deepseek/deepseek-chat") == 1
    assert ids.count("anthropic/open_router/anthropic/claude-opus") == 1
    display_names = {item["id"]: item["display_name"] for item in data["data"]}
    assert (
        display_names["anthropic/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3"
    )
    assert (
        display_names["claude-3-freecc-no-thinking/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3 (no thinking)"
    )
    assert "claude-sonnet-4-20250514" in ids
    assert "claude-fable-5" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]
    assert data["has_more"] is False
    assert all(
        "provider_model_ref" not in item and "apiBackend" not in item
        for item in data["data"]
    )


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
    ids = [item["id"] for item in response.json()["data"]]
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

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/plain-model" not in ids
    assert ids[0] == "claude-3-freecc-no-thinking/open_router/plain-model"


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

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/wafer/DeepSeek-V4-Pro" in ids
    assert "claude-3-freecc-no-thinking/wafer/DeepSeek-V4-Pro" in ids
    assert "anthropic/wafer/MiniMax-M2.7" in ids
    assert "claude-3-freecc-no-thinking/wafer/MiniMax-M2.7" in ids


def test_models_list_includes_fallback_refs_without_cached_discovery():
    app = create_test_app(
        _settings(
            model_opus=None,
            model_haiku=None,
            model_fallbacks=("wafer/DeepSeek-V4-Pro",),
        )
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/wafer/DeepSeek-V4-Pro" in ids
    assert "claude-3-freecc-no-thinking/wafer/DeepSeek-V4-Pro" in ids


def test_models_list_works_with_empty_discovery_catalog():
    app = create_test_app(_settings())

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids[:4] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
    ]
    assert "claude-sonnet-4-20250514" in ids


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

    messages = TestClient(app).get("/v1/models?view=messages").json()
    responses = TestClient(app).get("/v1/models?view=responses").json()

    assert [row["id"] for row in messages["data"]] == [
        "deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/open_router/plain-model",
        "open_router/reasoning-model",
    ]
    assert "claude-sonnet-4-20250514" not in {row["id"] for row in messages["data"]}
    assert all("apiBackend" not in row for row in messages["data"])
    assert responses["data"][0] == {
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
        "inferenceIdleTimeoutSecs": 660,
    }
    plain = responses["data"][1]
    assert plain["supportsReasoningEffort"] is False
    assert "reasoningEfforts" not in plain


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
        row["provider_model_ref"]: row
        for row in client.get("/v1/models?view=messages").json()["data"]
    }
    responses = {
        row["provider_model_ref"]: row
        for row in client.get("/v1/models?view=responses").json()["data"]
    }

    assert messages["open_router/vision-reasoning"]["supportsReasoning"] is True
    assert messages["open_router/vision-reasoning"]["inputModalities"] == [
        "text",
        "image",
    ]
    assert messages["open_router/vision-reasoning"]["contextWindow"] == 131072
    assert messages["open_router/vision-reasoning"]["maxCompletionTokens"] == 8192
    assert messages["open_router/text-only"]["supportsReasoning"] is False
    assert messages["open_router/text-only"]["inputModalities"] == ["text"]
    assert messages["open_router/text-only"]["contextWindow"] == 65536
    assert "maxCompletionTokens" not in messages["open_router/text-only"]
    assert "supportsReasoning" not in messages["open_router/unknown"]
    assert "inputModalities" not in messages["open_router/unknown"]
    assert "contextWindow" not in messages["open_router/unknown"]
    assert "maxCompletionTokens" not in messages["open_router/unknown"]
    assert responses["open_router/text-only"]["supportsReasoningEffort"] is False
    assert "reasoningEfforts" not in responses["open_router/text-only"]
    assert responses["open_router/vision-reasoning"]["contextWindow"] == 131072
    assert responses["open_router/vision-reasoning"]["maxCompletionTokens"] == 8192


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

    rows = TestClient(app).get("/v1/models").json()["data"]

    assert all(
        "supportsReasoning" not in row
        and "inputModalities" not in row
        and "contextWindow" not in row
        and "maxCompletionTokens" not in row
        for row in rows
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
    assert all("metadata" not in row for row in responses.json()["data"])
    for row in muse.json()["data"]:
        metadata = row["metadata"]["muse-code"]
        assert metadata["is_hidden"] is False
        assert metadata["name"] == row["display_name"]
        assert "limit" not in metadata
        original = {key: value for key, value in row.items() if key != "metadata"}
        assert original in responses.json()["data"]
    assert [row["id"] for row in muse.json()["data"]] == [
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
        row["provider_model_ref"]: row["metadata"]["muse-code"]
        for row in TestClient(app).get("/muse-code/models").json()["data"]
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
        "limit": {"context": 65536},
    }
    assert rows["deepseek/deepseek-chat"] == {
        "name": "deepseek/deepseek-chat",
        "is_hidden": False,
    }


def test_responses_model_view_rounds_timeout_up_before_margin():
    app = create_test_app(
        _settings().model_copy(update={"provider_progress_timeout": 1.1})
    )

    response = TestClient(app).get("/v1/models?view=responses")

    assert response.status_code == 200
    assert {row["inferenceIdleTimeoutSecs"] for row in response.json()["data"]} == {62}


def test_responses_model_view_timeout_fits_unsigned_range():
    largest_accepted_timeout = math.nextafter(float(1 << 64), 0.0)
    app = create_test_app(
        _settings().model_copy(
            update={"provider_progress_timeout": largest_accepted_timeout}
        )
    )

    response = TestClient(app).get("/v1/models?view=responses")

    assert response.status_code == 200
    assert all(
        row["inferenceIdleTimeoutSecs"] <= (1 << 64) - 1
        for row in response.json()["data"]
    )


def test_unknown_model_view_is_rejected():
    response = TestClient(create_test_app(_settings())).get("/v1/models?view=other")

    assert response.status_code == 422


def test_models_list_surfaces_cached_model_metadata():
    """Optional pricing/capability metadata appears when cached."""
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    provider_manager_for_app(app).cache_model_infos(
        "deepseek",
        {
            ProviderModelInfo(
                "deepseek-chat",
                supports_thinking=True,
                context_window_tokens=128000,
                max_output_tokens=8192,
                capabilities=("chat", "thinking", "tools"),
                pricing=ProviderModelPricing(input=0.0001, output=0.0002),
            )
        },
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    by_id = {item["id"]: item for item in response.json()["data"]}
    rich = by_id["anthropic/deepseek/deepseek-chat"]
    assert rich["capabilities"] == ["chat", "thinking", "tools"]
    assert rich["pricing"] == {"input": 0.0001, "output": 0.0002}
    # The no-thinking route forces reasoning off, so it must never
    # advertise thinking even though the upstream model supports it.
    no_thinking = by_id["claude-3-freecc-no-thinking/deepseek/deepseek-chat"]
    assert no_thinking["capabilities"] == ["chat", "tools"]
    assert no_thinking["pricing"] == {"input": 0.0001, "output": 0.0002}


def test_models_list_omits_metadata_when_upstream_does_not_provide_it():
    """Models without metadata omit the optional fields entirely."""
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    _cache_models(app, "deepseek", "deepseek-chat")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    by_id = {item["id"]: item for item in response.json()["data"]}
    plain = by_id["anthropic/deepseek/deepseek-chat"]
    assert "capabilities" not in plain
    assert "pricing" not in plain


def test_direct_model_views_surface_optional_metadata():
    """messages/responses views expose the same optional metadata."""
    app = create_test_app(_settings(model_opus=None, model_haiku=None))
    provider_manager_for_app(app).cache_model_infos(
        "deepseek",
        {
            ProviderModelInfo(
                "deepseek-chat",
                supports_thinking=True,
                capabilities=("chat", "thinking", "tools"),
                pricing=ProviderModelPricing(input=0.0001, output=0.0002),
            ),
            ProviderModelInfo(
                "plain-chat",
                supports_thinking=False,
                capabilities=("chat", "thinking"),
            ),
            ProviderModelInfo("unknown-model"),
        },
    )
    client = TestClient(app)

    for view in ("messages", "responses"):
        rows = {
            item["id"]: item
            for item in client.get(f"/v1/models?view={view}").json()["data"]
        }
        rich = rows["deepseek/deepseek-chat"]
        assert rich["capabilities"] == ["chat", "thinking", "tools"]
        assert rich["pricing"] == {"input": 0.0001, "output": 0.0002}
        # supports_thinking=False routes through the no-thinking gateway id,
        # which must not advertise thinking.
        plain = rows["claude-3-freecc-no-thinking/deepseek/plain-chat"]
        assert plain["capabilities"] == ["chat"]
        assert "pricing" not in plain
        unknown = rows["deepseek/unknown-model"]
        assert "capabilities" not in unknown
        assert "pricing" not in unknown
