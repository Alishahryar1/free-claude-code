"""Authenticated local catalog acquisition for terminal launchers."""

import json
from collections.abc import Mapping
from typing import Literal
from urllib.request import Request

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.cli.local_http import open_local_request
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality

CATALOG_TIMEOUT_SECONDS = 35.0


def fetch_proxy_model_catalog(
    proxy_root_url: str,
    auth_token: str,
    view: Literal["messages", "responses"] = "responses",
) -> ModelCatalog:
    """Fetch the authenticated FCC-local `/v1/models` response directly."""

    url = f"{proxy_root_url.rstrip('/')}/v1/models?view={view}"
    request = Request(
        url,
        headers={"Authorization": f"Bearer {auth_token}"},
        method="GET",
    )
    with open_local_request(request, timeout=CATALOG_TIMEOUT_SECONDS) as response:
        payload: JsonValue = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("model list response was not a JSON object")
    return model_catalog_from_response(payload)


def model_catalog_from_response(payload: Mapping[str, JsonValue]) -> ModelCatalog:
    models = catalog_models_from_response(payload)
    if not models:
        raise ValueError("model catalog contains no routable models")
    default_model_id = _nonempty_string(payload.get("default_model_id"))
    if default_model_id is None:
        raise ValueError(
            "model catalog is missing its default; restart the FCC server after updating"
        )
    if not any(model.wire_slug == default_model_id for model in models):
        raise ValueError("model catalog default is not a routable model")
    return ModelCatalog(models, default_model_id)


def catalog_models_from_response(
    models_response: Mapping[str, JsonValue],
) -> tuple[CatalogModel, ...]:
    """Project an FCC `/v1/models` response into direct client model records."""

    models: list[CatalogModel] = []
    seen_slugs: set[str] = set()

    for candidate in _catalog_candidates(models_response):
        if candidate.wire_slug in seen_slugs:
            continue
        seen_slugs.add(candidate.wire_slug)
        models.append(candidate)

    return tuple(models)


def _catalog_candidates(
    models_response: Mapping[str, JsonValue],
) -> list[CatalogModel]:
    data = models_response.get("data")
    if not isinstance(data, list):
        return []

    candidates: list[CatalogModel] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = _nonempty_string(item.get("id"))
        if model_id is None:
            continue
        provider_model_ref = _provider_model_ref(item.get("provider_model_ref"))
        if provider_model_ref is None:
            continue
        candidates.append(
            CatalogModel(
                wire_slug=model_id,
                provider_model_ref=provider_model_ref,
                display_name=_nonempty_string(item.get("display_name")) or model_id,
                supports_reasoning=_optional_boolean(item.get("supportsReasoning")),
                input_modalities=_input_modalities(item.get("inputModalities")),
                context_window_tokens=_optional_positive_int(item.get("contextWindow")),
                max_output_tokens=_optional_positive_int(
                    item.get("maxCompletionTokens")
                ),
            )
        )
    return candidates


def _nonempty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.strip() else None


def _optional_boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _input_modalities(value: object) -> frozenset[ModelInputModality] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        modalities = frozenset(ModelInputModality(item) for item in value)
    except TypeError, ValueError:
        return None
    if ModelInputModality.TEXT not in modalities:
        return None
    return modalities


def _provider_model_ref(value: object) -> str | None:
    ref = _nonempty_string(value)
    if ref is None:
        return None
    provider_id, separator, model_id = ref.partition("/")
    if not separator or not provider_id or not model_id:
        return None
    return ref
