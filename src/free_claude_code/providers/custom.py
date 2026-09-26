"""Custom endpoint resource ownership using FCC's existing wire transports."""

from collections.abc import AsyncIterator, Mapping

import httpx
import httpx2

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.core.anthropic import MessagesRequest, ReasoningReplayMode
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    responses_reasoning_policy,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    extract_openai_model_infos,
    model_infos_from_ids,
)
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    ChatTemplateReasoning,
    NamedEffortReasoning,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    ReasoningEncoder,
    ReasoningObject,
    ThinkingObjectReasoning,
    create_chat_client,
)
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport


def _reasoning_encoder(name: str) -> ReasoningEncoder:
    efforts = tuple(
        (effort, "xhigh" if effort is ReasoningEffort.MAX else effort.value)
        for effort in ReasoningEffort
    )
    limited = tuple(
        (
            effort,
            "low"
            if effort is ReasoningEffort.MINIMAL
            else "high"
            if effort in {ReasoningEffort.XHIGH, ReasoningEffort.MAX}
            else effort.value,
        )
        for effort in ReasoningEffort
    )
    return {
        "provider_default": NO_REASONING,
        "openai_effort": NamedEffortReasoning(
            efforts, disabled_value="none", enabled_value="medium"
        ),
        "limited_effort": NamedEffortReasoning(limited, enabled_value="medium"),
        "reasoning_object": ReasoningObject(limited),
        "thinking": ThinkingObjectReasoning({"type": "enabled"}, {"type": "disabled"}),
        "chat_template": ChatTemplateReasoning("enable_thinking"),
    }[name]


async def _omit_cookies(request: httpx.Request) -> None:
    request.headers.pop("cookie", None)


class CustomProvider(BaseProvider):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        definition: CustomProviderDefinition,
        admission: ProviderAdmissionController,
        openai_transport: httpx2.AsyncBaseTransport | None = None,
        messages_transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(config)
        self._definition = definition
        self._admission = admission
        self._snapshot = HttpEndpoint(
            base_url=config.base_url, api_key=config.api_key, headers={}
        )
        self._openai_transport = openai_transport
        self._sdk = None
        self._http = None
        self._chat = None
        self._responses = None
        self._messages = None
        if definition.api_format == "anthropic_messages":
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    write=config.http_write_timeout,
                ),
                transport=messages_transport,
                follow_redirects=False,
                event_hooks={"request": [_omit_cookies]},
            )
            mode = definition.reasoning_format
            capabilities = MessagesModelCapabilities(
                adaptive_thinking="unsupported"
                if mode == "messages_manual"
                else "required"
                if mode == "messages_adaptive"
                else None,
                supports_output_effort=False
                if mode == "messages_manual"
                else True
                if mode == "messages_adaptive"
                else None,
                supported_efforts=("low", "medium", "high", "xhigh", "max")
                if mode == "messages_adaptive"
                else None,
            )
            self._messages = AnthropicMessagesTransport(
                client=self._http,
                admission=admission,
                provider_name=definition.provider_id,
                replay_scope=definition.provider_id,
                read_timeout_s=config.http_read_timeout,
                capabilities=capabilities,
            )
        else:

            async def credential() -> str:
                return config.api_key or ""

            self._sdk = create_chat_client(
                config,
                base_url=config.base_url,
                provider_name=definition.provider_id,
                api_key_provider=credential,
            )
            if definition.api_format == "openai_chat":
                profile = OpenAIChatProfile(
                    OpenAIChatRequestPolicy(
                        definition.provider_id,
                        ReasoningReplayMode(definition.reasoning_history_format),
                    ),
                    _reasoning_encoder(definition.reasoning_format),
                    reasoning_delta_fallback_field="reasoning",
                )
                self._chat = OpenAIChatTransport(
                    client=self._sdk,
                    admission=admission,
                    behavior=OpenAIChatBehavior(profile),
                    read_timeout_s=config.http_read_timeout,
                    log_raw_sse_events=config.log_raw_sse_events,
                    log_api_error_tracebacks=config.log_api_error_tracebacks,
                    endpoint_transport=openai_transport,
                )
            else:
                self._responses = OpenAIResponsesTransport(
                    client=self._sdk,
                    admission=admission,
                    provider_name=definition.provider_id,
                    read_timeout_s=config.http_read_timeout,
                    log_raw_sse_events=config.log_raw_sse_events,
                    endpoint_transport=openai_transport,
                )

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        return self._snapshot

    async def cleanup(self) -> None:
        if self._sdk is not None:
            await self._sdk.close()
        if self._http is not None:
            await self._http.aclose()
        if self._openai_transport is not None:
            await self._openai_transport.aclose()

    def _reasoning(self, reasoning: ReasoningPolicy) -> ReasoningPolicy:
        mode = self._definition.reasoning_format
        if mode == "provider_default":
            return DEFAULT_REASONING_POLICY
        if reasoning.budget_tokens is not None and mode in {
            "openai_effort",
            "limited_effort",
            "native_responses",
            "messages_adaptive",
        }:
            raise InvalidRequestError(
                "The selected reasoning format cannot represent an exact token budget"
            )
        return reasoning

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
        reasoning = self._reasoning(reasoning)
        if (
            self._definition.reasoning_format == "provider_default"
            and self._messages is None
        ):
            request = request.model_copy(
                update={
                    "thinking": None,
                    "output_config": {
                        key: value
                        for key, value in (request.output_config or {}).items()
                        if key != "effort"
                    },
                }
            )
        if self._messages is not None:
            return self._messages.stream_messages(
                request,
                endpoint_context=self,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=model_info,
            )
        transport = self._chat if self._chat is not None else self._responses
        assert transport is not None
        return transport.stream_messages(
            request,
            input_tokens=input_tokens,
            endpoint_context=self,
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
        reasoning = (
            responses_reasoning_policy(request.reasoning)
            if self._definition.reasoning_format == "provider_default"
            and self._responses is not None
            else self._reasoning(reasoning)
        )
        if (
            self._definition.reasoning_format == "provider_default"
            and self._responses is None
        ):
            request = request.model_copy(
                update={
                    "reasoning": {
                        key: value
                        for key, value in (request.reasoning or {}).items()
                        if key != "effort"
                    }
                }
            )
        if self._messages is not None:
            return self._messages.stream_responses(
                request,
                endpoint_context=self,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
            )
        transport = self._chat if self._chat is not None else self._responses
        assert transport is not None
        return transport.stream_responses(
            request,
            input_tokens=input_tokens,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._definition.model_ids:
            return model_infos_from_ids(self._definition.model_ids)
        if self._sdk is not None:
            binding = OpenAIRequestClient(self._openai_transport)
            try:
                client = binding.for_endpoint(self._sdk, self._snapshot)
                payload = await self._admission.start_execution().run_call(
                    lambda: client.models.list(extra_headers=binding.openai_headers()),
                    operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
                )
                return extract_openai_model_infos(
                    payload, provider_name=self._definition.provider_id
                )
            finally:
                await binding.aclose()
        assert self._http is not None
        headers = {"anthropic-version": "2023-06-01"}
        if self._config.api_key:
            headers["x-api-key"] = self._config.api_key
        params = {"limit": "1000"}
        seen: set[str] = set()
        infos: dict[str, ProviderModelInfo] = {}
        for _ in range(100):
            response = await self._admission.start_execution().run_call(
                lambda params=params: self._fetch_models(headers, params),
                operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
            )
            if not isinstance(response, dict) or not isinstance(
                response.get("has_more"), bool
            ):
                raise ModelListResponseError("Invalid Messages model-list page")
            page = extract_openai_model_infos(
                response,
                provider_name=self._definition.provider_id,
                thinking_boolean_path=("capabilities", "thinking", "supported"),
                fixed_input_modalities=frozenset({ModelInputModality.TEXT}),
                input_modality_boolean_paths=(
                    (
                        ModelInputModality.IMAGE,
                        ("capabilities", "image_input", "supported"),
                    ),
                ),
                context_window_tokens_path=("max_input_tokens",),
                max_output_tokens_path=("max_tokens",),
            )
            infos.update((info.model_id, info) for info in page)
            if not response["has_more"]:
                return frozenset(infos.values())
            cursor = response.get("last_id")
            if (
                not isinstance(cursor, str)
                or not cursor
                or cursor in seen
                or cursor not in {info.model_id for info in page}
            ):
                raise ModelListResponseError(
                    "Messages model-list cursor did not advance"
                )
            seen.add(cursor)
            params = {"limit": "1000", "after_id": cursor}
        raise ModelListResponseError("Messages model-list exceeded 100 pages")

    async def _fetch_models(
        self, headers: dict[str, str], params: dict[str, str]
    ) -> object:
        assert self._http is not None
        response = await self._http.get(
            self._config.base_url + "/models", headers=headers, params=params
        )
        response.raise_for_status()
        return response.json()
