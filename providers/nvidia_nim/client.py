"""NVIDIA NIM provider implementation."""

from typing import Any

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaNimProvider(OpenAICompatibleProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_BASE_URL,
            api_key=config.api_key,
            nim_settings=nim_settings,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        assert self._nim_settings is not None
        return build_request_body(request, self._nim_settings)
