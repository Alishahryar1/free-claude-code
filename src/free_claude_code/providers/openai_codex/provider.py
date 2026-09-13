"""ChatGPT Codex provider backed by shared SDK Responses execution."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx2
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.model_listing import (
    optional_input_modalities,
    optional_positive_int,
    reasoning_capability_from_options,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport

from .auth import OpenAIAuthManager
from .endpoint import CodexEndpointContext
from .login import OPENAI_CODEX_ORIGINATOR

try:
    FCC_VERSION = version("free-claude-code")
except PackageNotFoundError:
    FCC_VERSION = "dev"


class OpenAICodexProvider(BaseProvider):
    """Own SDK resources and supply the Codex backend's request requirements."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: OpenAIAuthManager,
        admission: ProviderAdmissionController,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._admission = admission
        self._client_headers = {
            "User-Agent": f"{OPENAI_CODEX_ORIGINATOR}/{FCC_VERSION}",
            "originator": OPENAI_CODEX_ORIGINATOR,
            "version": FCC_VERSION,
        }
        self._pool = (
            transport
            if transport is not None
            else httpx2.AsyncHTTPTransport(proxy=config.proxy)
        )
        timeout = httpx2.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            write=config.http_write_timeout,
        )
        self._client = AsyncOpenAI(
            api_key=_endpoint_required,
            base_url=config.base_url,
            max_retries=0,
            timeout=timeout,
            http_client=httpx2.AsyncClient(transport=self._pool, timeout=timeout),
        )
        self._responses = OpenAIResponsesTransport(
            client=self._client,
            admission=admission,
            provider_name="OpenAI",
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            endpoint_transport=self._pool,
            omitted_request_fields=frozenset({"max_output_tokens", "metadata"}),
        )

    def _endpoint(self, *, session_id: str | None = None) -> CodexEndpointContext:
        return CodexEndpointContext(
            self._auth,
            base_url=self._config.base_url,
            headers=self._client_headers,
            session_id=session_id,
        )

    async def cleanup(self) -> None:
        await self._client.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Discover models visible to the currently connected ChatGPT account."""
        return _model_infos(await self._list_models_payload())

    async def _list_models_payload(self) -> Any:
        """Admit each catalog GET while borrowing request-scoped SDK credentials."""
        execution = self._admission.start_execution()
        endpoint = RequestEndpoint(self._endpoint(), self._pool)
        try:
            while execution.can_attempt:
                scope: ProviderAttemptScope | None = None
                try:
                    client = await endpoint.openai_client(self._client)
                    attempt = await execution.open_attempt(
                        ProviderOperationKind.MODEL_DISCOVERY
                    )
                    scope = ProviderAttemptScope(
                        attempt,
                        provider_name="OpenAI",
                        request_id=execution.request_id,
                    )
                    payload = await client.get(
                        "models",
                        cast_to=object,
                        options={"params": {"client_version": FCC_VERSION}},
                    )
                    await attempt.accept()
                    execution.succeed()
                    return payload
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if scope is not None:
                        if await endpoint.retry_authentication(
                            error, scope.attempt, execution
                        ):
                            continue
                        if not scope.attempt.accepted:
                            decision = await scope.attempt.fail(error)
                            if decision.retry_allowed:
                                continue
                    execution.fail(error)
                    raise
                finally:
                    if scope is not None:
                        await scope.aclose(active_error=sys.exception())
            if execution.last_failure is not None:
                raise execution.last_failure
            raise RuntimeError("OpenAI model discovery ended without an outcome.")
        finally:
            try:
                await endpoint.aclose()
            finally:
                execution.abandon()

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        return self._responses.stream_messages(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            endpoint_context=self._endpoint(session_id=str(uuid.uuid4())),
            model_info=model_info,
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        return self._responses.stream_responses(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            endpoint_context=self._endpoint(session_id=str(uuid.uuid4())),
        )


async def _endpoint_required() -> str:
    raise RuntimeError(
        "Codex requests require request-scoped subscription credentials."
    )


def _model_infos(payload: Any) -> frozenset[ProviderModelInfo]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("OpenAI model-list response is missing the models array.")
    infos: set[ProviderModelInfo] = set()
    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("slug")
        visibility = model.get("visibility")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or visibility != "list"
        ):
            continue
        efforts = model.get(
            "supported_reasoning_levels",
            model.get("supported_reasoning_efforts"),
        )
        infos.add(
            ProviderModelInfo(
                model_id=model_id,
                supports_thinking=_supports_reasoning(efforts),
                reasoning_capability=reasoning_capability_from_options(
                    {
                        "supported_efforts": [level["effort"] for level in efforts]
                        if _supports_reasoning(efforts) is not None
                        else None,
                    }
                ),
                input_modalities=optional_input_modalities(
                    model.get("input_modalities")
                ),
                context_window_tokens=optional_positive_int(
                    model.get("context_window")
                ),
            )
        )
    if not infos:
        raise ValueError("OpenAI did not advertise any visible models.")
    return frozenset(infos)


def _supports_reasoning(levels: object) -> bool | None:
    if not isinstance(levels, list):
        return None
    for level in levels:
        if not isinstance(level, dict):
            return None
        effort = level.get("effort")
        if not isinstance(effort, str) or not effort.strip():
            return None
    return bool(levels)
