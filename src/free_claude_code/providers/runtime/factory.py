"""Provider construction from declarative profiles and exceptional adapters."""

import importlib
from collections.abc import Callable, Mapping

from free_claude_code.application.errors import (
    ApplicationUnavailableError,
    UnknownProviderError,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    create_openai_chat_provider,
)

from .config import build_provider_config

ProviderFactory = Callable[
    [ProviderConfig, Settings, ProviderAdmissionController], BaseProvider
]


def _load_nvidia_nim() -> ProviderFactory:
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return NvidiaNimProvider(
            config,
            nim_settings=settings.nim,
            admission=admission,
        )

    return construct


def _load_open_router() -> ProviderFactory:
    from free_claude_code.providers.open_router import OpenRouterProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return OpenRouterProvider(config, admission=admission)

    return construct


def _load_mistral() -> ProviderFactory:
    from free_claude_code.providers.mistral import MistralProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return MistralProvider(config, admission=admission)

    return construct


def _load_kilo() -> ProviderFactory:
    from free_claude_code.providers.kilo import KiloProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return KiloProvider(config, admission=admission)

    return construct


def _load_deepseek() -> ProviderFactory:
    from free_claude_code.providers.deepseek import DeepSeekProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return DeepSeekProvider(config, admission=admission)

    return construct


def _load_lmstudio() -> ProviderFactory:
    from free_claude_code.providers.lmstudio import LMStudioProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return LMStudioProvider(config, admission=admission)

    return construct


def _load_cloudflare() -> ProviderFactory:
    from free_claude_code.providers.cloudflare import CloudflareProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return CloudflareProvider(
            config,
            account_id=_required_setting(settings, "cloudflare_account_id"),
            admission=admission,
        )

    return construct


def _load_gemini() -> ProviderFactory:
    from free_claude_code.providers.gemini import GeminiProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return GeminiProvider(config, admission=admission)

    return construct


def _load_vertex() -> ProviderFactory:
    from free_claude_code.providers.vertex import VertexProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return VertexProvider(
            config,
            project_id=_required_setting(settings, "vertex_project_id"),
            location=settings.vertex_location,
            admission=admission,
        )

    return construct


def _load_groq() -> ProviderFactory:
    from free_claude_code.providers.groq import GroqProvider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return GroqProvider(config, admission=admission)

    return construct


def _load_opencode_zen() -> ProviderFactory:
    from free_claude_code.providers.opencode import create_opencode_provider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return create_opencode_provider("opencode_zen", config, admission)

    return construct


def _load_opencode_go() -> ProviderFactory:
    from free_claude_code.providers.opencode import create_opencode_provider

    def construct(
        config: ProviderConfig,
        settings: Settings,
        admission: ProviderAdmissionController,
    ) -> BaseProvider:
        return create_opencode_provider("opencode_go", config, admission)

    return construct


_SPECIAL_PROVIDER_FACTORIES: dict[str, Callable[[], ProviderFactory]] = {
    "nvidia_nim": _load_nvidia_nim,
    "open_router": _load_open_router,
    "mistral": _load_mistral,
    "kilo": _load_kilo,
    "deepseek": _load_deepseek,
    "lmstudio": _load_lmstudio,
    "cloudflare": _load_cloudflare,
    "gemini": _load_gemini,
    "vertex": _load_vertex,
    "groq": _load_groq,
    "opencode_zen": _load_opencode_zen,
    "opencode_go": _load_opencode_go,
}
_INJECTED_PROVIDER_IDS = {"openai", "github_copilot"}


def _required_setting(settings: Settings, attr_name: str) -> str:
    value = getattr(settings, attr_name, None)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"Provider config did not validate {attr_name!r}")
    return value


_profiled_ids = set(OPENAI_CHAT_PROFILES)
_special_ids = set(_SPECIAL_PROVIDER_FACTORIES)
_construction_ids = _profiled_ids | _special_ids | _INJECTED_PROVIDER_IDS
if (
    _profiled_ids & _special_ids
    or _profiled_ids & _INJECTED_PROVIDER_IDS
    or _special_ids & _INJECTED_PROVIDER_IDS
    or _construction_ids != set(PROVIDER_CATALOG)
):
    raise AssertionError(
        "Every provider must have exactly one construction owner: "
        f"profiles={_profiled_ids!r} special={_special_ids!r} "
        f"injected={_INJECTED_PROVIDER_IDS!r} catalog={set(PROVIDER_CATALOG)!r}"
    )


def prepare_provider(
    provider_id: str,
    provider_loaders: Mapping[str, Callable[[], ProviderFactory]],
) -> Callable[[Settings], BaseProvider]:
    """Load implementation modules in a worker; return a loop-owned constructor."""

    # The SDK lazily imports these on first client resource access. Keep that
    # work in this loader, before constructing clients on their owner loop.
    importlib.import_module("openai.resources")
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if descriptor is None:
        raise UnknownProviderError.for_provider(provider_id, PROVIDER_CATALOG)
    loader = provider_loaders.get(provider_id) or _SPECIAL_PROVIDER_FACTORIES.get(
        provider_id
    )
    if provider_id in _INJECTED_PROVIDER_IDS and loader is None:
        raise ApplicationUnavailableError(
            f"Provider {provider_id!r} is unavailable in this runtime."
        )
    factory = loader() if loader is not None else None

    def construct(settings: Settings) -> BaseProvider:
        config = build_provider_config(descriptor, settings)
        admission = ProviderAdmissionController(
            provider_name=provider_id,
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        if factory is not None:
            return factory(config, settings, admission)
        return create_openai_chat_provider(provider_id, config, admission)

    return construct
