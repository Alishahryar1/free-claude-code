"""OpenCode rich-catalog parsing and provider-scoped snapshot ownership."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import httpx

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import classify_provider_failure
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    optional_input_modalities,
    optional_positive_int,
)

OPENCODE_CATALOG_URL = "https://models.opencode.ai/api.json"
_DEFAULT_PACKAGE = "@ai-sdk/openai-compatible"
_RESPONSES_PACKAGE = "@ai-sdk/openai"
_VISIBLE_STATUSES = frozenset({"active", "beta"})
_HIDDEN_STATUSES = frozenset({"alpha", "deprecated"})


class OpenCodeUpstreamTransport(StrEnum):
    """Standard OpenAI endpoint selected by one catalog route."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


@dataclass(frozen=True, slots=True)
class OpenCodeModelRoute:
    """One public selector resolved to its upstream model and transport."""

    selector_id: str
    upstream_model_id: str
    transport: OpenCodeUpstreamTransport
    supports_thinking: bool | None
    input_modalities: frozenset[ModelInputModality] | None
    context_window_tokens: int | None
    max_output_tokens: int | None

    @property
    def model_info(self) -> ProviderModelInfo:
        return ProviderModelInfo(
            model_id=self.selector_id,
            supports_thinking=self.supports_thinking,
            input_modalities=self.input_modalities,
            context_window_tokens=self.context_window_tokens,
            max_output_tokens=self.max_output_tokens,
            reasoning_capability=ReasoningCapability.NONE
            if self.supports_thinking is False
            else ReasoningCapability.UNKNOWN,
        )


@dataclass(frozen=True, slots=True)
class OpenCodeCatalogSnapshot:
    """One fully validated immutable provider-specific catalog view."""

    routes: Mapping[str, OpenCodeModelRoute]
    model_infos: frozenset[ProviderModelInfo]

    def route(self, selector_id: str) -> OpenCodeModelRoute | None:
        return self.routes.get(selector_id)


def parse_open_code_catalog(
    payload: object,
    *,
    provider_key: str,
    provider_name: str,
) -> OpenCodeCatalogSnapshot:
    """Parse one provider section without publishing a partial snapshot."""
    root = _mapping(payload, provider_name, "top-level object")
    provider = _mapping(
        root.get(provider_key),
        provider_name,
        f"provider section {provider_key!r}",
    )
    provider_package = _optional_package(
        provider.get("npm"),
        provider_name=provider_name,
        location="provider npm",
    )
    models = _mapping(
        provider.get("models"),
        provider_name,
        "models object",
    )

    routes: dict[str, OpenCodeModelRoute] = {}
    for raw_selector_id, raw_model in models.items():
        selector_id = _required_string(
            raw_selector_id,
            provider_name=provider_name,
            location="model selector",
        )
        if selector_id in routes:
            raise _catalog_error(
                provider_name,
                f"duplicate normalized model selector {selector_id!r}",
            )
        model = _mapping(
            raw_model,
            provider_name,
            f"model {selector_id!r}",
        )
        status = _model_status(model, provider_name, selector_id)
        if status in _HIDDEN_STATUSES:
            continue

        upstream_model_id = _required_string(
            model.get("id"),
            provider_name=provider_name,
            location=f"model {selector_id!r} id",
        )
        model_provider = model.get("provider")
        model_package = None
        if model_provider is not None:
            provider_override = _mapping(
                model_provider,
                provider_name,
                f"model {selector_id!r} provider",
            )
            model_package = _optional_package(
                provider_override.get("npm"),
                provider_name=provider_name,
                location=f"model {selector_id!r} provider npm",
            )
        supports_thinking = _optional_boolean(model.get("reasoning"))
        raw_modalities = model.get("modalities")
        input_modalities = (
            optional_input_modalities(raw_modalities.get("input"))
            if isinstance(raw_modalities, Mapping)
            else None
        )
        limit = model.get("limit")
        context_window_tokens = (
            optional_positive_int(limit.get("context"))
            if isinstance(limit, Mapping)
            else None
        )
        max_output_tokens = (
            optional_positive_int(limit.get("output"))
            if isinstance(limit, Mapping)
            else None
        )
        effective_package = model_package or provider_package or _DEFAULT_PACKAGE
        route = OpenCodeModelRoute(
            selector_id=selector_id,
            upstream_model_id=upstream_model_id,
            transport=(
                OpenCodeUpstreamTransport.RESPONSES
                if effective_package == _RESPONSES_PACKAGE
                else OpenCodeUpstreamTransport.CHAT_COMPLETIONS
            ),
            supports_thinking=supports_thinking,
            input_modalities=input_modalities,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
        )
        routes[selector_id] = route

    if not routes:
        raise _catalog_error(provider_name, "no active or beta models")
    model_infos = frozenset(route.model_info for route in routes.values())
    return OpenCodeCatalogSnapshot(
        routes=MappingProxyType(routes),
        model_infos=model_infos,
    )


class OpenCodeCatalog:
    """Own one OpenCode provider's catalog client and last-good snapshot."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_key: str,
        provider_name: str,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider_key = provider_key
        self._provider_name = provider_name
        self._admission = admission
        self._client = client or httpx.AsyncClient(
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )
        self._snapshot: OpenCodeCatalogSnapshot | None = None
        self._lock = asyncio.Lock()
        self._read_timeout = config.http_read_timeout

    @property
    def current_snapshot(self) -> OpenCodeCatalogSnapshot | None:
        return self._snapshot

    async def snapshot(
        self, *, request_id: str | None = None
    ) -> OpenCodeCatalogSnapshot:
        """Return last-good data or coalesce one cold catalog load."""
        if self._snapshot is not None:
            return self._snapshot
        async with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            try:
                return await self._load_and_publish(request_id=request_id)
            except Exception as exc:
                raise classify_provider_failure(
                    exc,
                    provider_name=self._provider_name,
                    read_timeout_s=self._read_timeout,
                    request_id=request_id,
                ) from exc

    async def refresh(self) -> OpenCodeCatalogSnapshot:
        """Load and atomically publish one fully validated fresh snapshot."""
        async with self._lock:
            return await self._load_and_publish()

    async def cleanup(self) -> None:
        await self._client.aclose()

    async def _load_and_publish(
        self, *, request_id: str | None = None
    ) -> OpenCodeCatalogSnapshot:
        async def request() -> OpenCodeCatalogSnapshot:
            response = await self._client.get(
                OPENCODE_CATALOG_URL,
                headers={"Accept": "application/json", "User-Agent": "opencode"},
            )
            try:
                response.raise_for_status()
                try:
                    payload: object = response.json()
                except ValueError as exc:
                    raise _catalog_error(
                        self._provider_name,
                        "invalid JSON",
                    ) from exc
                return parse_open_code_catalog(
                    payload,
                    provider_key=self._provider_key,
                    provider_name=self._provider_name,
                )
            finally:
                await response.aclose()

        execution = self._admission.start_execution(request_id=request_id)
        snapshot = await execution.run_call(
            request,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
        )
        self._snapshot = snapshot
        return snapshot


def _mapping(
    value: object,
    provider_name: str,
    location: str,
) -> Mapping[object, object]:
    if isinstance(value, Mapping):
        return value
    raise _catalog_error(provider_name, f"expected {location}")


def _required_string(
    value: object,
    *,
    provider_name: str,
    location: str,
) -> str:
    if isinstance(value, str) and (normalized := value.strip()):
        return normalized
    raise _catalog_error(provider_name, f"expected non-empty {location}")


def _optional_package(
    value: object,
    *,
    provider_name: str,
    location: str,
) -> str | None:
    if value is None:
        return None
    return _required_string(
        value,
        provider_name=provider_name,
        location=location,
    )


def _optional_boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _model_status(
    model: Mapping[object, object],
    provider_name: str,
    selector_id: str,
) -> str:
    value = model.get("status")
    if value is None:
        return "active"
    status = _required_string(
        value,
        provider_name=provider_name,
        location=f"model {selector_id!r} status",
    )
    if status not in _VISIBLE_STATUSES | _HIDDEN_STATUSES:
        raise _catalog_error(
            provider_name,
            f"unknown model {selector_id!r} status {status!r}",
        )
    return status


def _catalog_error(provider_name: str, detail: str) -> ModelListResponseError:
    return ModelListResponseError(
        f"{provider_name} rich catalog is malformed: {detail}"
    )
