"""Typed capabilities consumed by application use cases."""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Protocol

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy

from .model_metadata import ProviderModelInfo
from .responses_execution import ResponsesBinding


class ProviderPort(Protocol):
    """Minimal provider capability required to execute one request."""

    def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]: ...

    def bind_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[ResponsesBinding]: ...


ProviderResolver = Callable[[str], Awaitable[ProviderPort]]
ModelInfoLookup = Callable[[str, str], ProviderModelInfo | None]


class RequestRuntimeLease(Protocol):
    """One provider generation retained for a complete API response."""

    @property
    def generation_id(self) -> int: ...

    @property
    def settings(self) -> Settings: ...

    def is_provider_cached(self, provider_id: str) -> bool: ...

    async def wait_for_token_estimation(self) -> None: ...

    async def resolve_provider(self, provider_id: str) -> ProviderPort: ...

    def model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None: ...

    async def release(self) -> None: ...


class ModelCatalogPort(Protocol):
    """A coherent read-only projection of model configuration and metadata."""

    def current_settings(self) -> Settings: ...

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None: ...

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]: ...


@dataclass(frozen=True, slots=True)
class ModelCatalogSnapshot:
    settings: Settings
    model_infos: tuple[ProviderModelInfo, ...]

    def current_settings(self) -> Settings:
        return self.settings

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return self.model_infos

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        prefixed = f"{provider_id}/{model_id}"
        return next(
            (
                replace(info, model_id=model_id)
                for info in self.model_infos
                if info.model_id == prefixed
            ),
            None,
        )


class RequestRuntimePort(ModelCatalogPort, Protocol):
    """Provider generation and model metadata required by application requests."""

    async def acquire(self) -> RequestRuntimeLease: ...

    async def wait_for_catalog(self) -> ModelCatalogSnapshot: ...

    def catalog_status(self) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class StopResult:
    """Implementation-neutral result retaining the existing ``/stop`` variants."""

    cancelled_count: int | None = None
    source: str | None = None


class TaskController(Protocol):
    """Stop managed work without exposing messaging or CLI resources."""

    async def stop_all(self) -> StopResult | None: ...
