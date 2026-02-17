"""LM Studio provider implementation."""

from typing import Any

from providers.openai_compat import OpenAICompatibleProvider
from providers.base import ProviderConfig

from .request import build_request_body


LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio provider using OpenAI-compatible local API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LMSTUDIO",
            base_url=config.base_url or LMSTUDIO_DEFAULT_BASE_URL,
            api_key=config.api_key or "lm-studio",
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
