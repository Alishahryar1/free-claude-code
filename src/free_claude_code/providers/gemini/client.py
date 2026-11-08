"""Google AI Studio provider using native GenerateContent and model discovery."""

from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import uuid4

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.gemini.native_request import (
    apply_reasoning,
    chat_to_gemini,
    messages_to_gemini,
    native_gemini_body,
    responses_to_gemini,
)
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningCapability,
    ReasoningControl,
    ReasoningPolicy,
)
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.openai_chat import (
    AnthropicChatStreamOutput,
    NativeChatStreamOutput,
    ResponsesChatStreamOutput,
)
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)

from .transport import GeminiTransport


class GeminiProvider(BaseProvider):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        history_cache: HistoryModeCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config, history_cache=history_cache)
        self._admission = admission
        self._snapshot = HttpEndpoint(
            base_url=config.base_url.rstrip("/"), api_key=config.api_key, headers={}
        )
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            proxy=config.proxy,
            transport=transport,
            follow_redirects=False,
        )
        self._gemini = GeminiTransport(
            self._http,
            admission,
            self._history_cache,
            read_timeout=config.http_read_timeout,
        )

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        return self._snapshot

    async def cleanup(self) -> None:
        await self._http.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        records: dict[str, ProviderModelInfo] = {}
        page_token: str | None = None
        seen: set[str] = set()
        while True:

            async def fetch(page_token: str | None = page_token) -> dict[str, Any]:
                response = await self._http.get(
                    f"{self._snapshot.base_url}/models",
                    headers={"x-goog-api-key": self._snapshot.api_key or ""},
                    params={"pageToken": page_token} if page_token else {},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("models"), list
                ):
                    raise ValueError(
                        "Gemini model discovery returned an invalid catalog"
                    )
                return payload

            execution = self._admission.start_execution(model=None)
            payload = await execution.run_call(
                fetch, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
            )
            for model in payload["models"]:
                if not isinstance(model, dict) or not isinstance(
                    model.get("name"), str
                ):
                    raise ValueError("Gemini model discovery returned an invalid model")
                if "generateContent" not in model.get("supportedGenerationMethods", []):
                    continue
                records[model["name"]] = ProviderModelInfo(
                    model_id=model["name"],
                    supports_thinking=model.get("thinking")
                    if isinstance(model.get("thinking"), bool)
                    else None,
                    reasoning_capability=ReasoningCapability.NONE
                    if model.get("thinking") is False
                    else ReasoningCapability.UNKNOWN,
                    context_window_tokens=model.get("inputTokenLimit"),
                    max_output_tokens=model.get("outputTokenLimit"),
                )
            page_token = payload.get("nextPageToken")
            if not page_token:
                return frozenset(records.values())
            if not isinstance(page_token, str) or page_token in seen:
                raise ValueError("Gemini model discovery repeated a page token")
            seen.add(page_token)

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
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        prepared, wire_reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            # Native zero-budget disabling is supported by the 2.5 Flash family.
            can_disable=request.model.removeprefix("models/").startswith(
                "gemini-2.5-flash"
            ),
            normal_max_tokens=None,
        )
        try:
            body = messages_to_gemini(prepared)
            apply_reasoning(body, wire_reasoning, native=False)
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error
        correction = (
            ReasoningCorrection(
                (("generationConfig", "thinkingConfig", "thinkingBudget"),),
                ("generationConfig", "maxOutputTokens"),
                None,
            )
            if reasoning.control is ReasoningControl.PREFER_OFF
            and wire_reasoning.control is ReasoningControl.OFF
            else None
        )
        public_model = response_model or request.model
        message_id = f"msg_{uuid4().hex}"
        return self._gemini.stream(
            body,
            endpoint_context=self,
            request_id=request_id,
            response_model=public_model,
            include_thoughts=reasoning.output_enabled,
            reasoning_correction=correction,
            output_factory=lambda: AnthropicChatStreamOutput(
                message_id=message_id,
                model=public_model,
                input_tokens=input_tokens,
                log_raw_events=self._config.log_raw_sse_events,
            ),
            continuation=continuation,
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
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        try:
            body = responses_to_gemini(request)
            apply_reasoning(body, reasoning, native=False)
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error
        public_model = response_model or request.model
        adapter = ResponsesToolAdapter(request, ResponsesToolPolicy())
        return self._gemini.stream(
            body,
            endpoint_context=self,
            request_id=request_id,
            response_model=public_model,
            include_thoughts=reasoning.output_enabled,
            output_factory=lambda: ResponsesChatStreamOutput(
                adapter, input_tokens=input_tokens, response_model=public_model
            ),
            continuation=continuation,
        )

    def stream_chat(
        self,
        request: ChatRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        try:
            body = chat_to_gemini(request)
            apply_reasoning(body, reasoning, native=False)
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error
        public_model = response_model or request.model
        return self._gemini.stream(
            body,
            endpoint_context=self,
            request_id=request_id,
            response_model=public_model,
            include_thoughts=reasoning.output_enabled,
            output_factory=lambda: NativeChatStreamOutput(
                model=public_model,
                input_tokens=input_tokens,
                include_usage=request.include_usage,
            ),
            continuation=continuation,
        )

    def stream_gemini(
        self,
        request: GeminiRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        try:
            body = native_gemini_body(request)
            apply_reasoning(body, reasoning, native=True)
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error
        public_model = response_model or request.model

        return self._gemini.stream(
            body,
            endpoint_context=self,
            request_id=request_id,
            response_model=public_model,
            include_thoughts=reasoning.output_enabled,
            output_factory=lambda: None,
            continuation=continuation,
        )
