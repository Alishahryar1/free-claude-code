"""Command Code model catalog parsing and provider-scoped snapshot ownership."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import httpx

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_reporting import finalize_provider_failure
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    optional_positive_int,
)

_MESSAGES_ENDPOINT = "/messages"
_CHAT_ENDPOINT = "/chat/completions"


class CommandCodeTransport(StrEnum):
    """Upstream wire format selected by one catalog route."""

    MESSAGES = "messages"
    CHAT_COMPLETIONS = "chat_completions"


@dataclass(frozen=True, slots=True)
class CommandCodeModelRoute:
    """One advertised model resolved to the transport that serves it."""

    model_id: str
    transport: CommandCodeTransport
    model_info: ProviderModelInfo


@dataclass(frozen=True, slots=True)
class CommandCodeCatalogSnapshot:
    """One fully validated immutable catalog view."""

    routes: Mapping[str, CommandCodeModelRoute]
    model_infos: frozenset[ProviderModelInfo]

    def route(self, model_id: str) -> CommandCodeModelRoute | None:
        return self.routes.get(model_id)


def parse_command_code_catalog(
    payload: object, *, provider_name: str
) -> CommandCodeCatalogSnapshot:
    """Parse one ``GET /models`` response without publishing a partial snapshot.

    Claude models advertise only ``/messages``; every other model advertises
    ``/chat/completions`` (and usually ``/responses``, which is not used here).
    A model that advertises neither supported endpoint is skipped.
    """
    if not isinstance(payload, Mapping):
        raise _catalog_error(provider_name, "expected top-level object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise _catalog_error(provider_name, "expected data array")

    routes: dict[str, CommandCodeModelRoute] = {}
    for position, raw_model in enumerate(data):
        if not isinstance(raw_model, Mapping):
            raise _catalog_error(provider_name, f"expected model object at {position}")
        raw_id = raw_model.get("id")
        if not isinstance(raw_id, str) or not (model_id := raw_id.strip()):
            raise _catalog_error(provider_name, f"expected model id at {position}")
        if model_id in routes:
            raise _catalog_error(provider_name, f"duplicate model id {model_id!r}")
        endpoints = raw_model.get("supported_endpoints")
        if not isinstance(endpoints, list) or not all(
            isinstance(endpoint, str) for endpoint in endpoints
        ):
            raise _catalog_error(
                provider_name, f"expected supported_endpoints of model {model_id!r}"
            )
        if _MESSAGES_ENDPOINT in endpoints:
            transport = CommandCodeTransport.MESSAGES
        elif _CHAT_ENDPOINT in endpoints:
            transport = CommandCodeTransport.CHAT_COMPLETIONS
        else:
            continue
        routes[model_id] = CommandCodeModelRoute(
            model_id=model_id,
            transport=transport,
            model_info=ProviderModelInfo(
                model_id=model_id,
                context_window_tokens=optional_positive_int(
                    raw_model.get("context_length")
                ),
            ),
        )

    if not routes:
        raise _catalog_error(provider_name, "no model served by a supported endpoint")
    return CommandCodeCatalogSnapshot(
        routes=MappingProxyType(routes),
        model_infos=frozenset(route.model_info for route in routes.values()),
    )


class CommandCodeCatalog:
    """Own the authenticated catalog request and the last-good snapshot."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient,
    ) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._provider_name = provider_name
        self._admission = admission
        self._client = client
        self._read_timeout = config.http_read_timeout
        self._snapshot: CommandCodeCatalogSnapshot | None = None
        self._lock = asyncio.Lock()

    async def snapshot(
        self, *, request_id: str | None = None
    ) -> CommandCodeCatalogSnapshot:
        """Return last-good data or coalesce one cold catalog load."""
        if self._snapshot is not None:
            return self._snapshot
        async with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            try:
                return await self._load_and_publish(request_id=request_id)
            except Exception as exc:
                raise finalize_provider_failure(
                    exc,
                    provider_name=self._provider_name,
                    read_timeout_s=self._read_timeout,
                    request_id=request_id,
                ) from exc

    async def refresh(self) -> CommandCodeCatalogSnapshot:
        """Load and atomically publish one fully validated fresh snapshot."""
        async with self._lock:
            return await self._load_and_publish()

    async def _load_and_publish(
        self, *, request_id: str | None = None
    ) -> CommandCodeCatalogSnapshot:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async def request() -> CommandCodeCatalogSnapshot:
            response = await self._client.get(
                self._base_url + "/models", headers=headers
            )
            try:
                response.raise_for_status()
                try:
                    payload: object = response.json()
                except ValueError as exc:
                    raise _catalog_error(self._provider_name, "invalid JSON") from exc
                return parse_command_code_catalog(
                    payload, provider_name=self._provider_name
                )
            finally:
                await response.aclose()

        execution = self._admission.start_execution(request_id=request_id, model=None)
        snapshot = await execution.run_call(
            request, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )
        self._snapshot = snapshot
        return snapshot


def _catalog_error(provider_name: str, detail: str) -> ModelListResponseError:
    return ModelListResponseError(f"{provider_name} catalog is malformed: {detail}")
