"""Installed Codex launcher and external-client credential handoff."""

import json
import sys
from collections.abc import Sequence

from free_claude_code.cli.environment import client_environment
from free_claude_code.config.loader import get_settings

from .codex_model_catalog import build_codex_model_catalog
from .common import proxy_v1_url
from .model_catalog import catalog_wire_slug_for_ref
from .resources import LaunchResources
from .runner import HarnessSpec, LaunchContext, PreparedLaunch, launch_harness

_PRINT_PROXY_AUTH_TOKEN_FLAG = "--print-proxy-auth-token"
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
        "model_providers.fcc.auth.args": [_PRINT_PROXY_AUTH_TOKEN_FLAG],
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
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog_path = files.write_json(
        "model-catalog.json", build_codex_model_catalog(ctx.models)
    )
    configuration = codex_config_args(
        api_url=ctx.proxy_root_url,
        model=catalog_wire_slug_for_ref(ctx.models, ctx.settings.model),
    )
    return PreparedLaunch(
        [
            ctx.binary_path,
            *configuration,
            "-c",
            f"model_catalog_json={json.dumps(str(catalog_path))}",
            *args,
        ],
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            remove_keys=tuple(_STRIPPED_CODEX_ENV_KEYS),
            remove_prefixes=("OPENAI_",),
        ),
    )


SPEC = HarnessSpec(
    binary_name="codex",
    display_name="Codex CLI",
    install_hint="Install Codex with: npm install -g @openai/codex",
    configure=prepare_codex_launch,
    catalog_view="responses",
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == [_PRINT_PROXY_AUTH_TOKEN_FLAG]:
        print(get_settings().proxy_auth_token)
        return
    launch_harness(SPEC, args)
