"""Process-local Hermes configuration for FCC model routing."""

from dataclasses import dataclass, field

from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.model_catalog import ClientModel

HERMES_PROVIDER_PREFIX = "fcc-"
HERMES_KEY_ENV_PREFIX = "FCC_HERMES_"

_AUXILIARY_TASKS = (
    "vision",
    "web_extract",
    "compression",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "memory_query_rewrite",
    "tts_audio_tags",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "goal_judge",
    "curator",
    "monitor",
    "background_review",
    "moa_reference",
    "moa_aggregator",
)


@dataclass(frozen=True, slots=True)
class HermesManagedConfig:
    """Secret-free managed overlay for one attached Hermes process."""

    config: JsonObject = field(repr=False)
    provider_ref: str
    key_env: str
    default_model: str


def build_hermes_managed_config(
    models: tuple[ClientModel, ...],
    *,
    proxy_root_url: str,
    nonce: str,
) -> HermesManagedConfig:
    """Translate an FCC catalog into one invocation-specific Hermes overlay."""

    if not models:
        raise ValueError("Hermes requires at least one routable FCC model")
    if not nonce or not nonce.isalnum():
        raise ValueError("Hermes configuration nonce must be alphanumeric")

    wire_slugs = tuple(model.wire_slug for model in models)
    active_model = wire_slugs[0]

    provider_key = f"{HERMES_PROVIDER_PREFIX}{nonce.lower()}"
    provider_ref = f"custom:{provider_key}"
    key_env = f"{HERMES_KEY_ENV_PREFIX}{nonce.upper()}"
    auxiliary: JsonObject = {
        task: _auxiliary_task_config(task, provider_ref=provider_ref)
        for task in _AUXILIARY_TASKS
    }
    model_overrides = {
        model.wire_slug: override
        for model in models
        if (override := _model_override(model))
    }
    config: JsonObject = {
        "providers": {
            provider_key: {
                "name": "Free Claude Code",
                "api": proxy_v1_url(proxy_root_url),
                "key_env": key_env,
                "extra_headers": {
                    "Authorization": f"Bearer ${{{key_env}}}",
                },
                "transport": "codex_responses",
                "default_model": active_model,
                "models": {wire_slug: {} for wire_slug in wire_slugs},
                "discover_models": False,
            }
        },
        "model": {
            "provider": provider_ref,
            "default": active_model,
            "base_url": "",
            "api_key": "",
            "api_mode": "codex_responses",
            "default_headers": {"x-fcc-launch-id": nonce},
        },
        "fallback_providers": [],
        "fallback_model": [],
        "auxiliary": auxiliary,
    }
    if model_overrides:
        config["model_overrides"] = {"custom": model_overrides}
    return HermesManagedConfig(
        config=config,
        provider_ref=provider_ref,
        key_env=key_env,
        default_model=active_model,
    )


def _auxiliary_task_config(task: str, *, provider_ref: str) -> JsonObject:
    config: JsonObject = {
        "provider": provider_ref,
        "model": "",
        "base_url": "",
        "api_key": "",
        "extra_body": {},
        "fallback_chain": [],
    }
    if task == "title_generation":
        config["prefer_fast_model"] = False
    return config


def _model_override(model: ClientModel) -> JsonObject:
    override: JsonObject = {}
    if model.supports_reasoning is not None:
        override["supports_reasoning"] = model.supports_reasoning
    if model.input_modalities is not None:
        override["supports_vision"] = ModelInputModality.IMAGE in model.input_modalities
    if model.context_window_tokens is not None:
        override["context_window"] = model.context_window_tokens
    if model.max_output_tokens is not None:
        override["max_output_tokens"] = model.max_output_tokens
    return override
