"""Model Studio chat transport with native DashScope model discovery."""

from typing import Any
from urllib.parse import urljoin

from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    OpenAIModelListing,
)

_PROFILE = OpenAIChatProfile(
    OpenAIChatRequestPolicy(
        provider_name="ALIBABA_CLOUD",
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    ),
    # Model Studio hosts several model families with different thinking controls.
    NO_REASONING,
    model_listing=OpenAIModelListing(
        id_field="model",
        required_sequence_items=(("features", "function-calling"),),
        thinking_sequence_path=("capabilities",),
        thinking_tag="Reasoning",
        input_modalities_path=("input_modalities",),
        context_window_tokens_path=("model_info", "context_window"),
        max_output_tokens_path=("model_info", "max_output_tokens"),
    ),
)


class AlibabaCloudProvider(OpenAIChatProvider):
    """Reuse Chat Completions while adapting the paginated model catalog."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(config, profile=_PROFILE, admission=admission, client=client)

    async def _list_models_payload(self) -> Any:
        # An absolute URL avoids the OpenAI SDK appending to /compatible-mode/v1.
        url = urljoin(self._base_url, "/api/v1/models")
        models: list[dict[str, Any]] = []
        expected_total: int | None = None
        for page in range(1, 101):
            params = {
                "page_no": page,
                "page_size": 100,
                "features": "function-calling",
                "supports": "inference",
            }
            payload = await self._admission.start_execution().run_call(
                lambda params=params: self._client.get(
                    url, cast_to=object, options={"params": params}
                ),
                operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
                provider_failure_override=self._behavior.failure_override,
            )
            if not isinstance(payload, dict) or payload.get("success") is not True:
                raise ModelListResponseError("ALIBABA_CLOUD model discovery failed")
            output = payload.get("output")
            if not isinstance(output, dict):
                raise ModelListResponseError(
                    "ALIBABA_CLOUD model-list output is missing"
                )
            total = output.get("total")
            items = output.get("models")
            if (
                type(total) is not int
                or total < 0
                or type(output.get("page_no")) is not int
                or output["page_no"] != page
                or (expected_total is not None and total != expected_total)
                or not isinstance(items, list)
                or not items
                or any(not isinstance(item, dict) for item in items)
                or len(models) + len(items) > total
            ):
                raise ModelListResponseError(
                    "ALIBABA_CLOUD model-list page is malformed"
                )
            expected_total = total
            for item in items:
                metadata = item.get("inference_metadata")
                modalities = (
                    metadata.get("request_modality")
                    if isinstance(metadata, dict)
                    else None
                )
                models.append(
                    {
                        **item,
                        "input_modalities": [value.lower() for value in modalities]
                        if isinstance(modalities, list)
                        and all(isinstance(value, str) for value in modalities)
                        else None,
                    }
                )
            if len(models) == total:
                return {"data": models}
        raise ModelListResponseError(
            "ALIBABA_CLOUD model-list pagination exceeded 100 pages"
        )
