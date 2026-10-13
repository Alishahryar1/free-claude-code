"""Authenticated cursor discovery shared by Messages endpoint owners."""

from collections.abc import Mapping
from typing import cast

import httpx

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    extract_openai_model_infos,
)


def messages_model_info(item: JsonObject, provider_name: str) -> ProviderModelInfo:
    infos = extract_openai_model_infos(
        {"data": [item]},
        provider_name=provider_name,
        thinking_boolean_path=("capabilities", "thinking", "supported"),
        fixed_input_modalities=frozenset({ModelInputModality.TEXT}),
        input_modality_boolean_paths=(
            (ModelInputModality.IMAGE, ("capabilities", "image_input", "supported")),
        ),
        context_window_tokens_path=("max_input_tokens",),
        max_output_tokens_path=("max_tokens",),
    )
    return next(iter(infos))


async def list_messages_models(
    client: httpx.AsyncClient,
    admission: ProviderAdmissionController,
    *,
    base_url: str,
    headers: Mapping[str, str],
) -> tuple[JsonObject, ...]:
    params = {"limit": "1000"}
    seen: set[str] = set()
    records: dict[str, JsonObject] = {}
    for _ in range(100):

        async def fetch(params: dict[str, str] = params) -> object:
            response = await client.get(
                base_url.rstrip("/") + "/models", headers=headers, params=params
            )
            response.raise_for_status()
            return response.json()

        page = await admission.start_execution().run_call(
            fetch, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("has_more"), bool)
            or not isinstance(page.get("data"), list)
        ):
            raise ModelListResponseError("Invalid Messages model-list page")
        ids: set[str] = set()
        for item in page["data"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not item["id"].strip()
            ):
                raise ModelListResponseError("Invalid Messages model identifier")
            ids.add(item["id"])
            records[item["id"]] = cast(JsonObject, item)
        if not page["has_more"]:
            return tuple(records.values())
        cursor = page.get("last_id")
        if not isinstance(cursor, str) or cursor not in ids or cursor in seen:
            raise ModelListResponseError("Messages model-list cursor did not advance")
        seen.add(cursor)
        params = {"limit": "1000", "after_id": cursor}
    raise ModelListResponseError("Messages model-list exceeded 100 pages")
