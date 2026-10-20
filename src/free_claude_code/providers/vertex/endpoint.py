"""Vertex AI service and OpenAI-compatible endpoint construction."""

import re
from urllib.parse import quote

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.provider_catalog import VERTEX_AI_API_ROOT

_LOCATION_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def vertex_service_endpoint(location: str) -> str:
    """Return Google's global or regional Vertex AI service endpoint."""
    normalized = _validated_location(location)
    if normalized == "global":
        return VERTEX_AI_API_ROOT
    return f"https://{normalized}-aiplatform.googleapis.com"


def vertex_openai_base_url(project_id: str, location: str) -> str:
    """Return the project-scoped Vertex OpenAI-compatible API base URL."""
    project = project_id.strip()
    if not project:
        raise ApplicationUnavailableError(
            "VERTEX_PROJECT_ID is not set. Add it in the Admin UI."
        )
    normalized_location = _validated_location(location)
    service_endpoint = vertex_service_endpoint(normalized_location)
    return (
        f"{service_endpoint}/v1/projects/{quote(project, safe='')}/locations/"
        f"{normalized_location}/endpoints/openapi"
    )


def vertex_publisher_models_url(location: str) -> str:
    """Return Google's native publisher-model listing endpoint."""
    return f"{vertex_service_endpoint(location)}/v1beta1/publishers/google/models"


def _validated_location(location: str) -> str:
    normalized = location.strip().lower()
    if not normalized:
        raise ApplicationUnavailableError(
            "VERTEX_LOCATION is not set. Use global or a Google Cloud region."
        )
    if _LOCATION_PATTERN.fullmatch(normalized) is None:
        raise ApplicationUnavailableError(
            "VERTEX_LOCATION must be global or a lowercase Google Cloud region."
        )
    return normalized
