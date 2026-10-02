"""Smoke-suite configuration loaded from the real developer environment."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from free_claude_code.config.loader import clear_settings_cache, get_settings
from free_claude_code.config.model_refs import parse_model_name, parse_provider_type
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderAuthKind,
)
from free_claude_code.config.settings import Settings
from free_claude_code.providers.runtime.config import has_provider_configuration

DEFAULT_TARGETS = frozenset(
    {
        "api",
        "auth",
        "cli",
        "clients",
        "config",
        "extensibility",
        "llamacpp",
        "lmstudio",
        "messaging",
        "ollama",
        "providers",
        "rate_limit",
        "tools",
    }
)
SIDE_EFFECT_TARGETS = frozenset({"discord", "telegram", "voice"})
OPT_IN_TARGETS = frozenset({"nvidia_nim_cli", "openrouter_free_cli"})
ALL_TARGETS = DEFAULT_TARGETS | SIDE_EFFECT_TARGETS | OPT_IN_TARGETS
TARGET_ALIASES = {
    "contract": "api",
    "nim_cli": "nvidia_nim_cli",
    "openrouter_cli": "openrouter_free_cli",
    "openrouter_free": "openrouter_free_cli",
    "optimizations": "api",
    "thinking": "providers",
    "vscode": "clients",
}
SECRET_KEY_PARTS = ("KEY", "TOKEN", "SECRET", "WEBHOOK", "AUTH")

PROVIDER_SMOKE_DEFAULT_MODELS: dict[str, str] = {
    "nvidia_nim": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
    "azure_openai": "azure_openai/gpt-5.1",
    "open_router": "open_router/moonshotai/kimi-k2.6:free",
    "mistral": "mistral/devstral-small-latest",
    "mistral_codestral": "mistral_codestral/codestral-latest",
    "deepseek": "deepseek/deepseek-v4-pro",
    "ollama_cloud": "ollama_cloud/qwen3-coder:480b",
    "lmstudio": "lmstudio/local-model",
    "llamacpp": "llamacpp/local-model",
    "ollama": "ollama/llama3.1",
    "kimi_code": "kimi_code/k3",
    "wafer": "wafer/DeepSeek-V4-Pro",
    "minimax": "minimax/MiniMax-M3",
    "opencode_zen": "opencode_zen/gpt-5.3-codex",
    "opencode_go": "opencode_go/minimax-m2.7",
    "vercel": "vercel/openai/gpt-5.5",
    "bedrock": "bedrock/openai.gpt-oss-120b",
    "huggingface": "huggingface/openai/gpt-oss-120b:fastest",
    "cohere": "cohere/command-a-plus-05-2026",
    "github_models": "github_models/openai/gpt-4.1",
    "zai": "zai/glm-5.2",
    "zai_api": "zai_api/glm-4.7-flash",
    "gemini": "gemini/models/gemini-3.1-flash-lite",
    "vertex": "vertex/google/gemini-3.5-flash",
    "groq": "groq/llama-3.3-70b-versatile",
    "cline_pass": "cline_pass/cline-pass/deepseek-v4-flash",
    "xai": "xai/grok-4.5",
    "qwencloud": "qwencloud/qwen3.7-plus",
    "qwencloud_coding": "qwencloud_coding/qwen3.7-plus",
    "together": "together/zai-org/GLM-5.2",
    "deepinfra": "deepinfra/deepseek-ai/DeepSeek-V4-Flash",
    "siliconflow": "siliconflow/Qwen/Qwen3-32B",
    "nebius": "nebius/Qwen/Qwen3-30B-A3B",
    "chutes": "chutes/Qwen/Qwen3-32B-TEE",
    "featherless": "featherless/Qwen/Qwen3-32B",
    "sambanova": "sambanova/Meta-Llama-3.3-70B-Instruct",
    "kilo": "kilo/kilo-auto/free",
    "cerebras": "cerebras/llama3.1-8b",
    "novita": "novita/deepseek/deepseek-v4-flash-0731",
    "cloudflare": "cloudflare/@cf/moonshotai/kimi-k2.6",
    "tokenrouter": "tokenrouter/moonshotai/kimi-k3-free",
    "nararoute": "nararoute/kimi-k3-free",
    "agnes": "agnes/agnes-2.0-flash",
    "zenmux": "zenmux/deepseek/deepseek-v4-flash-free",
    "wandb": "wandb/openai/gpt-oss-20b",
}
MISTRAL_REASONING_SMOKE_DEFAULT_MODEL = "mistral/mistral-medium-3-5"

NVIDIA_NIM_CLI_DEFAULT_MODELS: tuple[str, ...] = (
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "minimaxai/minimax-m2.7",
    "minimaxai/minimax-m3",
    "nvidia/nemotron-3-super-120b-a12b",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
)

OPENROUTER_FREE_CLI_DEFAULT_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "poolside/laguna-m.1:free",
)


TARGET_REQUIRED_ENV: dict[str, tuple[str, ...]] = {
    "api": (),
    "auth": (),
    "cli": ("FCC_SMOKE_CLAUDE_BIN", "configured provider for Claude CLI prompt"),
    "clients": (),
    "config": (),
    "extensibility": (),
    "messaging": (),
    "providers": ("configured provider credentials/endpoints or FCC_SMOKE_MODEL_*",),
    "rate_limit": ("configured provider model",),
    "tools": ("configured tool-capable provider model",),
    "lmstudio": ("LM_STUDIO_BASE_URL with a running LM Studio server",),
    "llamacpp": ("LLAMACPP_BASE_URL with a running llama-server",),
    "ollama": ("OLLAMA_BASE_URL with a running Ollama server",),
    "nvidia_nim_cli": (
        "NVIDIA_NIM_API_KEY",
        "FCC_SMOKE_CLAUDE_BIN or claude on PATH",
    ),
    "openrouter_free_cli": (
        "OPENROUTER_API_KEY",
        "FCC_SMOKE_CLAUDE_BIN or claude on PATH",
    ),
    "telegram": (
        "TELEGRAM_BOT_TOKEN",
        "ALLOWED_TELEGRAM_USER_ID or FCC_SMOKE_TELEGRAM_CHAT_ID",
    ),
    "discord": (
        "DISCORD_BOT_TOKEN",
        "ALLOWED_DISCORD_CHANNELS or FCC_SMOKE_DISCORD_CHANNEL_ID",
    ),
    "voice": ("VOICE_NOTE_ENABLED=true", "FCC_SMOKE_RUN_VOICE=1"),
}


@dataclass(frozen=True, slots=True)
class ProviderModel:
    provider: str
    full_model: str
    source: str

    @property
    def model_name(self) -> str:
        return parse_model_name(self.full_model)


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    root: Path
    results_dir: Path
    live: bool
    interactive: bool
    targets: frozenset[str]
    provider_matrix: frozenset[str]
    timeout_s: float
    prompt: str
    claude_bin: str
    worker_id: str
    settings: Settings

    @classmethod
    def load(cls) -> SmokeConfig:
        root = Path(__file__).resolve().parents[2]
        clear_settings_cache()
        settings = get_settings()
        return cls(
            root=root,
            results_dir=root / ".smoke-results",
            live=os.getenv("FCC_LIVE_SMOKE") == "1",
            interactive=os.getenv("FCC_SMOKE_INTERACTIVE") == "1",
            targets=_parse_targets(os.getenv("FCC_SMOKE_TARGETS")),
            provider_matrix=_parse_csv(os.getenv("FCC_SMOKE_PROVIDER_MATRIX")),
            timeout_s=float(os.getenv("FCC_SMOKE_TIMEOUT_S", "45")),
            prompt=os.getenv("FCC_SMOKE_PROMPT", "Reply with exactly: FCC_SMOKE_PONG"),
            claude_bin=os.getenv("FCC_SMOKE_CLAUDE_BIN", "claude"),
            worker_id=os.getenv("PYTEST_XDIST_WORKER", "main"),
            settings=settings,
        )

    def target_enabled(self, *names: str) -> bool:
        return any(name in self.targets for name in names)

    def provider_models(self) -> list[ProviderModel]:
        candidates = (
            ("MODEL", self.settings.model),
            ("MODEL_FABLE", self.settings.model_fable),
            ("MODEL_OPUS", self.settings.model_opus),
            ("MODEL_SONNET", self.settings.model_sonnet),
            ("MODEL_HAIKU", self.settings.model_haiku),
        )
        seen: set[str] = set()
        models: list[ProviderModel] = []
        for source, model in candidates:
            if not model or model in seen:
                continue
            provider = parse_provider_type(model)
            if self.provider_matrix and provider not in self.provider_matrix:
                continue
            if not self.has_provider_configuration(provider):
                continue
            seen.add(model)
            models.append(
                ProviderModel(provider=provider, full_model=model, source=source)
            )
        return models

    def provider_smoke_models(self) -> list[ProviderModel]:
        """Return one smoke model per configured provider, independent of MODEL_*."""
        models: list[ProviderModel] = []
        mapped_providers = {model.provider for model in self.provider_models()}
        for provider in SUPPORTED_PROVIDER_IDS:
            if self.provider_matrix and provider not in self.provider_matrix:
                continue
            if not self.has_provider_configuration(provider):
                continue
            if not self._include_provider_in_smoke(provider, mapped_providers):
                continue
            full_model, source = _provider_smoke_model(provider)
            models.append(
                ProviderModel(provider=provider, full_model=full_model, source=source)
            )
        return models

    def nvidia_nim_cli_models(self) -> list[ProviderModel]:
        """Return the NVIDIA NIM models for Claude Code CLI characterization."""
        return [
            ProviderModel(provider="nvidia_nim", full_model=full_model, source=source)
            for full_model, source in nvidia_nim_cli_model_refs().items()
        ]

    def openrouter_free_cli_models(self) -> list[ProviderModel]:
        """Return OpenRouter free models for Claude Code CLI characterization."""
        return [
            ProviderModel(provider="open_router", full_model=full_model, source=source)
            for full_model, source in openrouter_free_cli_model_refs().items()
        ]

    def mistral_reasoning_smoke_model(self) -> ProviderModel | None:
        """Return a Mistral model expected to accept native reasoning input."""
        if self.provider_matrix and "mistral" not in self.provider_matrix:
            return None
        if not self.has_provider_configuration("mistral"):
            return None
        override_env = "FCC_SMOKE_MODEL_MISTRAL_REASONING"
        if override := os.getenv(override_env):
            full_model = _normalize_provider_model("mistral", override)
            source = override_env
        else:
            full_model = MISTRAL_REASONING_SMOKE_DEFAULT_MODEL
            source = "mistral_reasoning_default"
        return ProviderModel(provider="mistral", full_model=full_model, source=source)

    def _include_provider_in_smoke(
        self, provider: str, mapped_providers: set[str]
    ) -> bool:
        descriptor = PROVIDER_CATALOG[provider]
        if not descriptor.local:
            return True
        if provider in mapped_providers:
            return True
        if self.provider_matrix and provider in self.provider_matrix:
            return True
        return bool(os.getenv(f"FCC_SMOKE_MODEL_{provider.upper()}"))

    def has_provider_configuration(self, provider: str) -> bool:
        descriptor = PROVIDER_CATALOG.get(provider)
        if descriptor is None:
            return False
        if descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT:
            return bool(os.getenv(f"FCC_SMOKE_MODEL_{provider.upper()}"))
        return has_provider_configuration(descriptor, self.settings)


def _parse_csv(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _parse_csv_ordered(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_targets(raw: str | None) -> frozenset[str]:
    if not raw:
        return DEFAULT_TARGETS
    parsed = _parse_csv(raw)
    if "all" in parsed:
        return ALL_TARGETS
    return frozenset(TARGET_ALIASES.get(target, target) for target in parsed)


def _provider_smoke_model(provider: str) -> tuple[str, str]:
    override_env = f"FCC_SMOKE_MODEL_{provider.upper()}"
    if override := os.getenv(override_env):
        return _normalize_provider_model(provider, override), override_env

    default = PROVIDER_SMOKE_DEFAULT_MODELS.get(provider)
    if default is None:
        descriptor = PROVIDER_CATALOG[provider]
        default = f"{descriptor.provider_id}/smoke-default"
    return default, "provider_default"


def _normalize_provider_model(provider: str, raw_model: str) -> str:
    model = raw_model.strip()
    if not model:
        msg = f"FCC_SMOKE_MODEL_{provider.upper()} must not be empty"
        raise ValueError(msg)
    if "/" in model and parse_provider_type(model) == provider:
        return model
    return f"{provider}/{model}"


def nvidia_nim_cli_model_refs(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return normalized NIM CLI matrix model refs in deterministic order.

    Values are returned as ``full_model -> source`` so callers can preserve both
    de-duplicated order and provenance in reports.
    """
    source = env if env is not None else os.environ
    explicit_models = _parse_csv_ordered(source.get("FCC_SMOKE_NIM_MODELS"))
    extra_models = _parse_csv_ordered(source.get("FCC_SMOKE_NIM_EXTRA_MODELS"))

    if "FCC_SMOKE_NIM_MODELS" in source and not explicit_models:
        raise ValueError("FCC_SMOKE_NIM_MODELS must list at least one model")

    models: list[tuple[str, str]] = []
    base_models = explicit_models or NVIDIA_NIM_CLI_DEFAULT_MODELS
    base_source = (
        "FCC_SMOKE_NIM_MODELS" if explicit_models else "nvidia_nim_cli_default"
    )
    models.extend((model, base_source) for model in base_models)
    models.extend((model, "FCC_SMOKE_NIM_EXTRA_MODELS") for model in extra_models)

    normalized: dict[str, str] = {}
    for raw_model, model_source in models:
        full_model = _normalize_provider_model("nvidia_nim", raw_model)
        normalized.setdefault(full_model, model_source)
    return normalized


def openrouter_free_cli_model_refs(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return normalized OpenRouter free CLI matrix model refs in deterministic order."""
    source = env if env is not None else os.environ
    explicit_models = _parse_csv_ordered(source.get("FCC_SMOKE_OPENROUTER_FREE_MODELS"))
    extra_models = _parse_csv_ordered(
        source.get("FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS")
    )

    if "FCC_SMOKE_OPENROUTER_FREE_MODELS" in source and not explicit_models:
        raise ValueError(
            "FCC_SMOKE_OPENROUTER_FREE_MODELS must list at least one model"
        )

    models: list[tuple[str, str]] = []
    base_models = explicit_models or OPENROUTER_FREE_CLI_DEFAULT_MODELS
    base_source = (
        "FCC_SMOKE_OPENROUTER_FREE_MODELS"
        if explicit_models
        else "openrouter_free_cli_default"
    )
    models.extend((model, base_source) for model in base_models)
    models.extend(
        (model, "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS") for model in extra_models
    )

    normalized: dict[str, str] = {}
    for raw_model, model_source in models:
        full_model = _normalize_provider_model("open_router", raw_model)
        normalized.setdefault(full_model, model_source)
    return normalized


def auth_headers(token: str | None = None) -> dict[str, str]:
    settings = get_settings()
    resolved = token if token is not None else settings.proxy_auth_token
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if resolved:
        headers["authorization"] = f"Bearer {resolved}"
    return headers


def redacted(value: str, env: Mapping[str, str] | None = None) -> str:
    """Redact known secrets from a string before writing smoke artifacts."""
    if not value:
        return value

    source = env if env is not None else os.environ
    result = value
    for key, secret in source.items():
        if not secret or len(secret) < 4:
            continue
        if any(part in key.upper() for part in SECRET_KEY_PARTS):
            result = result.replace(secret, f"<redacted:{key}>")
    return result
