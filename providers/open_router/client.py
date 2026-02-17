"""OpenRouter provider implementation."""

from typing import Any, Iterator

from providers.openai_compat import OpenAICompatibleProvider
from providers.base import ProviderConfig
from providers.common import SSEBuilder

from .request import build_request_body


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENROUTER",
            base_url=config.base_url or OPENROUTER_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)

    def _handle_extra_reasoning(self, delta: Any, sse: SSEBuilder) -> Iterator[str]:
        """Handle reasoning_details for StepFun models."""
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details and isinstance(reasoning_details, list):
            for item in reasoning_details:
                text = item.get("text", "") if isinstance(item, dict) else ""
                if text:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(text)
