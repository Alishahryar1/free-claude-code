"""OpenAI Platform provider using the shared Responses transport."""

import re
from collections.abc import AsyncIterator, Mapping
from typing import Literal

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.model_listing import extract_openai_model_infos
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport

type _SamplingSupport = Literal["never", "none_only"]
type _DefaultReasoning = Literal["none", "active", "unknown"]

# /models exposes IDs but no sampling capabilities. Unknown IDs pass through.
_MODEL_SAMPLING: dict[str, tuple[_SamplingSupport, _DefaultReasoning]] = {
    "gpt-5": ("never", "active"),
    "gpt-5-mini": ("never", "active"),
    "gpt-5-nano": ("never", "active"),
    "gpt-5.1": ("none_only", "none"),
    "gpt-5.2": ("none_only", "none"),
    "gpt-5.3-codex": ("never", "unknown"),
    "gpt-5.4": ("none_only", "none"),
    "gpt-6-astra": ("never", "unknown"),
    "gpt-6-sol": ("none_only", "active"),
    "gpt-6-luna": ("none_only", "active"),
}


def _adapt_openai_sampling(body: JsonObject) -> JsonObject:
    model = body.get("model")
    if not isinstance(model, str):
        return body
    base_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    profile = _MODEL_SAMPLING.get(base_model)
    if profile is None:
        return body
    support, default = profile
    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    reasoning_active = effort != "none" if effort is not None else default == "active"
    if support == "never" or reasoning_active:
        body.pop("temperature", None)
        body.pop("top_p", None)
    return body


class OpenAIAPIProvider(BaseProvider):
    """Own public API credentials and SDK resources, independent of ChatGPT OAuth."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAI API requires an API key")
        timeout = httpx2.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            write=config.http_write_timeout,
        )
        if transport is not None:
            http_client = httpx2.AsyncClient(transport=transport, timeout=timeout)
        elif config.proxy:
            http_client = DefaultAsyncHttpx2Client(proxy=config.proxy, timeout=timeout)
        else:
            http_client = None
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        self._admission = admission
        self._responses = OpenAIResponsesTransport(
            client=self._client,
            admission=admission,
            provider_name="OpenAI API",
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            request_body_adapter=_adapt_openai_sampling,
        )

    async def cleanup(self) -> None:
        await self._client.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        payload = await self._admission.start_execution().run_call(
            self._client.models.list,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
        )
        return extract_openai_model_infos(payload, provider_name="OPENAI_API")

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
        )
