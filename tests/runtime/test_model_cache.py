from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.runtime.model_cache import ProviderModelCache


def test_model_cache_returns_and_prefixes_complete_model_metadata() -> None:
    cache = ProviderModelCache(("open_router",))
    info = ProviderModelInfo(
        model_id="vendor/model",
        supports_thinking=False,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=131072,
        max_output_tokens=8192,
    )

    cache.cache_model_infos("open_router", (info,))

    assert cache.cached_model_info("open_router", "vendor/model") == info
    assert cache.cached_model_info("open_router", "missing") is None
    assert cache.cached_prefixed_model_infos() == (
        ProviderModelInfo(
            model_id="open_router/vendor/model",
            supports_thinking=False,
            input_modalities=frozenset(
                {ModelInputModality.TEXT, ModelInputModality.IMAGE}
            ),
            context_window_tokens=131072,
            max_output_tokens=8192,
        ),
    )
