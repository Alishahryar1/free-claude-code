"""Shared Codex provider configuration and native launch preparation."""

import json
from collections.abc import Mapping, Sequence

from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.harnesses.environment import client_environment

from .codex_model_catalog import build_codex_model_catalog
from .launch import PreparedLaunch
from .model_catalog import ClientModel, catalog_wire_slug_for_ref
from .resources import LaunchResources

PRINT_PROXY_AUTH_TOKEN_FLAG = "--print-proxy-auth-token"
CODEX_INSTALL_HINT = "Install Codex with: npm install -g @openai/codex"
_STRIPPED_CODEX_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "CODEX_API_KEY",
        "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
        "CODEX_PERMISSION_PROFILE",
        "CODEX_SHELL",
        "CODEX_THREAD_ID",
    }
)


def codex_config_values(
    *, api_url: str, model: str | None = None
) -> dict[str, str | list[str]]:
    """Shared configuration for the FCC Responses provider."""

    values: dict[str, str | list[str]] = {
        "model_provider": "fcc",
        "model_providers.fcc.name": "Free Claude Code",
        "model_providers.fcc.base_url": proxy_v1_url(api_url),
        "model_providers.fcc.auth.command": "fcc-codex",
        "model_providers.fcc.auth.args": [PRINT_PROXY_AUTH_TOKEN_FLAG],
        "model_providers.fcc.wire_api": "responses",
    }
    if model:
        values["model"] = model
    return values


def codex_config_args(*, api_url: str, model: str | None = None) -> list[str]:
    """Build native TOML overrides for the FCC Responses provider."""
    values = codex_config_values(api_url=api_url, model=model)
    return [
        arg
        for key, value in values.items()
        for arg in ("-c", f"{key}={json.dumps(value)}")
    ]


def prepare_codex_launch(
    *,
    binary_path: str,
    proxy_root_url: str,
    model: str,
    models: Sequence[ClientModel],
    base_env: Mapping[str, str],
    args: Sequence[str],
    files: LaunchResources,
) -> PreparedLaunch:
    catalog_path = files.write_json(
        "model-catalog.json", build_codex_model_catalog(models)
    )
    configuration = codex_config_args(
        api_url=proxy_root_url,
        model=catalog_wire_slug_for_ref(models, model),
    )
    return PreparedLaunch(
        [
            binary_path,
            *configuration,
            "-c",
            f"model_catalog_json={json.dumps(str(catalog_path))}",
            *args,
        ],
        client_environment(
            base_env,
            proxy_root_url=proxy_root_url,
            remove_keys=tuple(_STRIPPED_CODEX_ENV_KEYS),
            remove_prefixes=("OPENAI_",),
        ),
    )
