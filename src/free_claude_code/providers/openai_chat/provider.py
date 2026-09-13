"""Provider identity, HTTP resource ownership, and model discovery."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Any

import httpx2
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint import EndpointContext
from free_claude_code.providers.model_listing import (
    extract_openai_model_infos,
    merge_model_list_pages,
    model_infos_from_ids,
    validate_model_list_page,
)

from .behavior import OpenAIChatBehavior
from .client import OpenAIAsyncCredentialProvider, create_chat_client
from .profiles import OpenAIChatProfile
from .transport import OpenAIChatTransport


class OpenAIChatProvider(BaseProvider):
    """Own a Chat provider's client and discovery; compose its wire transport."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenAIChatProfile | None = None,
        behavior: OpenAIChatBehavior | None = None,
        admission: ProviderAdmissionController,
        default_headers: Mapping[str, str] | None = None,
        api_key_provider: OpenAIAsyncCredentialProvider | None = None,
        client: AsyncOpenAI | None = None,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        if behavior is None:
            if profile is None:
                raise ValueError("A Chat profile or behavior is required")
            behavior = OpenAIChatBehavior(profile)
        elif profile is not None:
            raise ValueError("Pass a Chat profile or behavior, not both")
        self._behavior = behavior
        self._profile = behavior.profile
        self._provider_name = self._profile.provider_name
        self._api_key = config.api_key
        self._base_url = self._profile.base_url(config.base_url).rstrip("/")
        self._admission = admission
        self._owns_client = client is None
        self._client = client or create_chat_client(
            config,
            base_url=self._base_url,
            provider_name=self._provider_name,
            default_headers=default_headers,
            api_key_provider=api_key_provider,
        )
        self._chat = OpenAIChatTransport(
            client=self._client,
            admission=admission,
            behavior=behavior,
            read_timeout_s=config.http_read_timeout,
            log_raw_sse_events=config.log_raw_sse_events,
            log_api_error_tracebacks=config.log_api_error_tracebacks,
            endpoint_transport=endpoint_transport,
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        client = getattr(self, "_client", None)
        if self._owns_client and client is not None:
            await client.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return model metadata from the OpenAI-compatible models endpoint."""
        payload = await self._list_models_payload()
        if not self._profile.model_ids_are_routable:
            return frozenset()
        listing = self._profile.model_listing
        live_model_infos = extract_openai_model_infos(
            payload,
            provider_name=self._provider_name,
            collection_field=listing.collection_field,
            id_field=listing.id_field,
            aliases_field=listing.aliases_field,
            required_path_values=listing.required_path_values,
            required_null_field=listing.required_null_field,
            required_sequence_items=listing.required_sequence_items,
            exclude_missing_sequence_fields=listing.exclude_missing_sequence_fields,
            tags_field=listing.tags_field,
            thinking_tag=listing.thinking_tag,
            non_thinking_tag=listing.non_thinking_tag,
            thinking_boolean_path=listing.thinking_boolean_path,
            input_modalities_path=listing.input_modalities_path,
            thinking_sequence_path=listing.thinking_sequence_path,
            fixed_input_modalities=listing.fixed_input_modalities,
            input_modality_boolean_paths=listing.input_modality_boolean_paths,
            context_window_tokens_path=listing.context_window_tokens_path,
            max_output_tokens_path=listing.max_output_tokens_path,
            context_window_tokens_resolver=listing.context_window_tokens_resolver,
        )
        model_infos_by_id = {
            model_info.model_id: model_info for model_info in live_model_infos
        }
        for model_info in model_infos_from_ids(listing.additional_model_ids):
            existing = model_infos_by_id.get(model_info.model_id)
            if existing is None:
                model_infos_by_id[model_info.model_id] = model_info
                continue
            model_infos_by_id[model_info.model_id] = replace(
                existing,
                context_window_tokens=None,
                max_output_tokens=None,
            )
        return frozenset(model_infos_by_id.values())

    async def _list_models_payload(self) -> Any:
        """Fetch one OpenAI-compatible model-list payload with shared retries."""
        return await self._fetch_models_payload()

    async def _fetch_models_payload(self) -> Any:
        """Fetch the complete profile-selected model-list payload."""
        listing = self._profile.model_listing
        if listing.path is not None and listing.pagination is not None:
            return await self._fetch_paginated_models_payload(listing.path)
        execution = self._admission.start_execution()
        return await execution.run_call(
            self._fetch_models_payload_once,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
            provider_failure_override=self._behavior.failure_override,
        )

    async def _fetch_models_payload_once(self) -> Any:
        """Fetch the profile-selected model-list endpoint once."""
        listing = self._profile.model_listing
        path = listing.path
        if path is None:
            return await self._client.models.list()
        if listing.query_params:
            return await self._client.get(
                path,
                cast_to=object,
                options={"params": dict(listing.query_params)},
            )
        return await self._client.get(path, cast_to=object)

    async def _fetch_paginated_models_payload(self, path: str) -> Any:
        """Fetch a bounded model catalog with one execution per physical page."""
        listing = self._profile.model_listing
        pagination = listing.pagination
        if pagination is None:
            raise RuntimeError("paginated model fetch requires a pagination policy")

        payloads: list[Any] = []
        total_pages: int | None = None
        page = pagination.first_page
        while total_pages is None or page < pagination.first_page + total_pages:
            params = dict(listing.query_params)
            params[pagination.page_param] = str(page)
            execution = self._admission.start_execution()
            payload = await execution.run_call(
                lambda params=params: self._client.get(
                    path,
                    cast_to=object,
                    options={"params": params},
                ),
                operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
                provider_failure_override=self._behavior.failure_override,
            )
            total_pages = validate_model_list_page(
                payload,
                provider_name=self._provider_name,
                expected_page=page,
                current_page_path=pagination.current_page_path,
                total_pages_path=pagination.total_pages_path,
                max_pages=pagination.max_pages,
                expected_total_pages=total_pages,
            )
            payloads.append(payload)
            page += 1

        return merge_model_list_pages(
            payloads,
            provider_name=self._provider_name,
            collection_field=listing.collection_field,
        )

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        return self._chat.stream_messages(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
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
        endpoint_context: EndpointContext | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        return self._chat.stream_responses(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
        )
