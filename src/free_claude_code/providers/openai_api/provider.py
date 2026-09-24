"""OpenAI Platform provider using the shared Responses transport."""

import re
from collections.abc import AsyncIterator, Mapping

import httpx2
from openai import APIError, APIStatusError, AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesStreamFailure,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.model_listing import extract_openai_model_infos
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport


def _sampling_retry_body(
    error: Exception, body: JsonObject, sent_body: JsonObject
) -> JsonObject | None:
    """Omit only a sampling field explicitly rejected on this outbound attempt."""
    if not isinstance(error, APIError | ResponsesStreamFailure):
        return None
    if isinstance(error, APIStatusError) and error.status_code != 400:
        return None
    detail = error.body
    if not isinstance(detail, Mapping):
        return None
    param = detail.get("param")
    if not isinstance(param, str) or param not in {"temperature", "top_p"}:
        return None
    if param not in body or param not in sent_body or body[param] != sent_body[param]:
        return None
    code = detail.get("code")
    if code == "unsupported_value":
        # This code also covers ordinary invalid values. Only default-only
        # model incompatibility authorizes dropping the caller's setting.
        message = detail.get("message")
        if (
            not isinstance(message, str)
            or re.fullmatch(
                rf"Unsupported value: ['\"]{param}['\"] does not support .+ "
                r"with this model\. Only the default \([^)]+\) value is supported\.",
                message.strip(),
                re.IGNORECASE,
            )
            is None
        ):
            return None
    elif code != "unsupported_parameter":
        return None
    return {key: value for key, value in body.items() if key != param}


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
            request_correction=_sampling_retry_body,
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
