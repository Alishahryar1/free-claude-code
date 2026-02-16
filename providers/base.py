"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field

from config.nim import NimSettings


class ProviderConfig(BaseModel):
    """Configuration for a provider.

    Base fields apply to all providers. Provider-specific parameters
    (e.g. NIM temperature, top_p) are passed via nim_settings.
    """

    api_key: str
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None
    rate_window: int = 60
    nim_settings: NimSettings = Field(default_factory=NimSettings)
    http_read_timeout: float = 300.0
    http_write_timeout: float = 10.0
    http_connect_timeout: float = 2.0


class BaseProvider(ABC):
    """Base class for all providers. Extend this to add your own."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        if False:
            yield ""
