"""Vertex publisher-model response parsing."""

from collections.abc import Mapping, Sequence
from typing import Any

from free_claude_code.providers.model_listing import ModelListResponseError


def extract_vertex_model_page(payload: Any) -> tuple[frozenset[str], str | None]:
    """Translate one Google publisher-model page to OpenAI-compatible model IDs."""
    if not isinstance(payload, Mapping):
        raise _malformed("expected an object")
    models = payload.get("publisherModels")
    if not _is_sequence(models):
        raise _malformed("expected top-level publisherModels array")

    model_ids: set[str] = set()
    for item in models:
        if not isinstance(item, Mapping):
            raise _malformed("expected every publisherModels item to be an object")
        name = item.get("name")
        if not isinstance(name, str):
            raise _malformed("expected every publisher model to include name")
        model_ids.add(_openai_model_id(name))

    next_page_token = payload.get("nextPageToken")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise _malformed("expected nextPageToken to be a string")
    return frozenset(model_ids), next_page_token or None


def _openai_model_id(resource_name: str) -> str:
    parts = resource_name.split("/", 3)
    if (
        len(parts) != 4
        or parts[0] != "publishers"
        or not parts[1].strip()
        or parts[2] != "models"
        or not parts[3].strip()
    ):
        raise _malformed("expected publisher model resource names")
    return f"{parts[1]}/{parts[3]}"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _malformed(reason: str) -> ModelListResponseError:
    return ModelListResponseError(f"VERTEX model-list response is malformed: {reason}")
