from dataclasses import replace

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality


def test_provider_model_info_preserves_capabilities_when_identity_is_replaced() -> None:
    info = ProviderModelInfo(
        model_id="vendor/model",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=131072,
        max_output_tokens=8192,
    )

    assert replace(info, model_id="provider/vendor/model") == ProviderModelInfo(
        model_id="provider/vendor/model",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=131072,
        max_output_tokens=8192,
    )
