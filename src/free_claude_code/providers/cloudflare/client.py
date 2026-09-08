"""Cloudflare Workers AI provider using OpenAI-compatible chat completions."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any
from urllib.parse import quote

import httpx

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import CLOUDFLARE_AI_REST_ROOT
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.http import maybe_await_aclose
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    extract_openai_model_infos,
)
from free_claude_code.providers.openai_chat import (
    ChatStreamOutput,
    ChatTemplateReasoning,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    validate_extra_body_does_not_override_canonical_fields,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="CLOUDFLARE",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    max_tokens_field="max_completion_tokens",
)
_PROFILE = OpenAIChatProfile(_REQUEST_POLICY, ChatTemplateReasoning())


def cloudflare_ai_base_url(api_root: str | None, account_id: str) -> str:
    """Return the account-scoped Cloudflare Workers AI OpenAI-compatible base URL."""

    return f"{_cloudflare_account_api_url(api_root, account_id)}/ai/v1"


def _cloudflare_model_search_url(api_root: str | None, account_id: str) -> str:
    """Return the Cloudflare account model-search endpoint URL."""

    return f"{_cloudflare_account_api_url(api_root, account_id)}/ai/models/search"


def _cloudflare_account_api_url(api_root: str | None, account_id: str) -> str:
    """Return the account-scoped Cloudflare API root URL."""

    stripped_account = account_id.strip()
    if not stripped_account:
        raise ApplicationUnavailableError(
            "CLOUDFLARE_ACCOUNT_ID is not set. Add it in the Admin UI."
        )
    root = (api_root or CLOUDFLARE_AI_REST_ROOT).rstrip("/")
    encoded_account = quote(stripped_account, safe="")
    return f"{root}/accounts/{encoded_account}"


class CloudflareProvider(OpenAIChatProvider):
    """Cloudflare Workers AI OpenAI-compatible chat provider."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        account_id: str,
        admission: ProviderAdmissionController,
    ):
        base_url = cloudflare_ai_base_url(config.base_url, account_id)
        self._model_search_url = _cloudflare_model_search_url(
            config.base_url, account_id
        )
        self._model_list_client = httpx.AsyncClient(
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )
        super().__init__(
            replace(config, base_url=base_url),
            profile=_PROFILE,
            admission=admission,
        )

    async def cleanup(self) -> None:
        """Release provider client resources."""
        await super().cleanup()
        await self._model_list_client.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return Cloudflare Workers AI metadata from account model search."""

        async def request() -> httpx.Response:
            response = await self._model_list_client.get(
                self._model_search_url,
                params={"format": "openrouter"},
                headers=self._model_list_headers(),
            )
            try:
                response.raise_for_status()
            except Exception:
                await maybe_await_aclose(response)
                raise
            return response

        execution = self._admission.start_execution()
        response = await execution.run_call(
            request,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
        )
        try:
            try:
                payload = response.json()
            except ValueError as exc:
                raise ModelListResponseError(
                    "CLOUDFLARE model-list response is malformed: invalid JSON"
                ) from exc
            return extract_openai_model_infos(payload, provider_name="CLOUDFLARE")
        finally:
            await maybe_await_aclose(response)

    def _handle_extra_reasoning(
        self, delta: Any, output: ChatStreamOutput, *, output_reasoning: bool
    ) -> Iterator[str]:
        """Map Cloudflare's ``reasoning`` delta field to Anthropic thinking."""
        reasoning = _cloudflare_reasoning(delta)
        if not output_reasoning or not reasoning:
            return
        yield from output.ensure_reasoning_block()
        yield output.emit_reasoning_delta(reasoning)

    def _model_list_headers(self) -> dict[str, str]:
        if not self._api_keys:
            raise AssertionError("Cloudflare requires a static API token")
        return {"Authorization": f"Bearer {self._api_keys[0]}"}


def _cloudflare_reasoning(delta: Any) -> str | None:
    reasoning = getattr(delta, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    model_extra = getattr(delta, "model_extra", None)
    if isinstance(model_extra, Mapping):
        reasoning = model_extra.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return reasoning

    return None
