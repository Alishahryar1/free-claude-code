"""CodeBuddy provider backed by shared Chat Completions execution."""

import asyncio
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Any

import httpx2
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.failure_policy import provider_authentication_status
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.model_listing import optional_positive_int
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
)
from free_claude_code.providers.request_recovery import RequestRecovery

from .auth import CodeBuddyAuthManager
from .behavior import CodeBuddyChatBehavior
from .endpoint import CodeBuddyEndpointContext
from .site import CodeBuddySite, default_site

# Known-good model ids on the intl-cli site, used when the live catalog
# endpoint is unavailable. Keep in sync with the upstream console.
_STATIC_MODEL_IDS: tuple[str, ...] = (
    "auto",
    "deepseek-v4.1-flash",
    "kimi-k3",
    "glm-5.3",
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "gemini-3.1-pro",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gpt-5.3-codex",
    "minimax-m3",
    "hy3",
)

_CODEBUDDY_PROFILE = OpenAIChatProfile(
    OpenAIChatRequestPolicy(
        provider_name="CODEBUDDY",
        reasoning_replay=ReasoningReplayMode.DISABLED,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    NO_REASONING,
)


class CodeBuddyProvider(BaseProvider):
    """Own SDK resources and supply CodeBuddy's request requirements."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: CodeBuddyAuthManager,
        admission: ProviderAdmissionController,
        site: CodeBuddySite | None = None,
        client: AsyncOpenAI | None = None,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._site = site or _site_from_base_url(config.base_url)
        self._admission = admission
        self._behavior = CodeBuddyChatBehavior(_CODEBUDDY_PROFILE)
        self._owns_client = client is None
        if client is not None:
            self._client = client
            self._pool = None
        else:
            self._pool = httpx2.AsyncHTTPTransport(proxy=config.proxy)
            timeout = httpx2.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            )
            self._client = AsyncOpenAI(
                api_key=_endpoint_required,
                base_url=self._site.chat_base_url,
                max_retries=0,
                timeout=timeout,
                http_client=httpx2.AsyncClient(transport=self._pool, timeout=timeout),
            )
        self._endpoint_transport = endpoint_transport
        self._chat = OpenAIChatTransport(
            client=self._client,
            admission=admission,
            behavior=self._behavior,
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            log_api_error_tracebacks=config.log_api_error_tracebacks,
            endpoint_transport=endpoint_transport,
        )

    def _endpoint(self) -> CodeBuddyEndpointContext:
        return CodeBuddyEndpointContext(self._auth, site=self._site)

    async def cleanup(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Discover models visible to the connected CodeBuddy account.

        The live console catalog is authoritative; when it cannot be read
        (offline upstream, expired session), fall back to the static
        known-good id list so routing keeps working.
        """

        try:
            payload = await self._fetch_catalog_payload()
            infos = _catalog_infos(payload)
        except Exception as error:
            logger.warning(
                "CODEBUDDY model catalog unavailable, using static list: {}",
                error,
            )
            return frozenset(
                ProviderModelInfo(model_id=model_id) for model_id in _STATIC_MODEL_IDS
            )
        if not infos:
            return frozenset(
                ProviderModelInfo(model_id=model_id) for model_id in _STATIC_MODEL_IDS
            )
        return infos

    async def _fetch_catalog_payload(self) -> Any:
        """Admit the live catalog GET while borrowing request credentials.

        One 401/403 forces a token refresh and retries once through the
        shared endpoint recovery before the caller falls back to the static
        model list.
        """

        execution = self._admission.start_execution()
        endpoint = RequestEndpoint(self._endpoint())
        recovery = RequestRecovery(execution, endpoint=endpoint)
        try:
            while execution.can_attempt:
                scope: ProviderAttemptScope | None = None
                try:
                    resolved = await endpoint.resolve()
                    attempt = await execution.open_attempt(
                        ProviderOperationKind.MODEL_DISCOVERY
                    )
                    scope = ProviderAttemptScope(
                        attempt,
                        provider_name=self._behavior.profile.provider_name,
                        request_id=execution.request_id,
                    )
                    payload = await self._fetch_catalog_payload_once(resolved)
                    await attempt.accept()
                    execution.succeed()
                    return payload
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if scope is not None:
                        if await recovery.retry_authentication(
                            error, provider_authentication_status(error), scope.attempt
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
            raise RuntimeError("CodeBuddy model discovery ended without an outcome.")
        finally:
            execution.abandon()

    async def _fetch_catalog_payload_once(self, endpoint: HttpEndpoint) -> Any:
        headers = {**endpoint.headers, "Accept": "application/json"}
        transport = self._endpoint_transport or self._pool
        async with httpx2.AsyncClient(
            transport=transport,
            timeout=httpx2.Timeout(30.0),
        ) as client:
            response = await client.get(
                f"{self._site.api_base}/console/enterprises/personal/models",
                headers=headers,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("code") != 0:
            raise RuntimeError("CodeBuddy model catalog returned an error envelope.")
        return payload

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
        return self._chat.stream_messages(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            endpoint_context=self._endpoint(),
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
        return self._chat.stream_responses(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            endpoint_context=self._endpoint(),
        )


async def _endpoint_required() -> str:
    raise RuntimeError(
        "CodeBuddy requests require request-scoped connected-account credentials."
    )


def _site_from_base_url(base_url: str | None) -> CodeBuddySite:
    """Honor a configured base URL, falling back to the intl-cli preset."""

    preset = default_site()
    if not base_url:
        return preset
    api_base = base_url.rstrip("/")
    if api_base == preset.api_base:
        return preset
    return replace(preset, api_base=api_base, origin=api_base)


def _catalog_infos(payload: Any) -> frozenset[ProviderModelInfo]:
    """Map the console catalog envelope to provider model metadata."""

    if not isinstance(payload, dict):
        raise ValueError("CodeBuddy model catalog is not an object.")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("CodeBuddy model catalog is missing its data section.")
    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError("CodeBuddy model catalog is missing the models array.")
    cli_ids = _cli_model_ids(data.get("agents"))

    infos: set[ProviderModelInfo] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if model.get("disabled"):
            continue
        if cli_ids is not None and model_id not in cli_ids:
            continue
        modalities = (
            frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE})
            if model.get("supportsImages") is True
            else frozenset({ModelInputModality.TEXT})
        )
        infos.add(
            ProviderModelInfo(
                model_id=model_id,
                supports_thinking=(
                    True if model.get("supportsReasoning") is True else None
                ),
                input_modalities=modalities,
                context_window_tokens=optional_positive_int(
                    model.get("maxInputTokens")
                ),
                max_output_tokens=optional_positive_int(model.get("maxOutputTokens")),
            )
        )
    return frozenset(infos)


def _cli_model_ids(agents: Any) -> set[str] | None:
    """Return the CLI agent's advertised model ids, None when unrestricted."""

    if not isinstance(agents, list):
        return None
    for agent in agents:
        if isinstance(agent, dict) and agent.get("name") == "cli":
            models = agent.get("models")
            if isinstance(models, list) and models:
                return {model for model in models if isinstance(model, str)}
    return None
