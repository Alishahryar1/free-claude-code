"""Installed Grok Build launcher with native FCC connection settings."""

import json
import re
from collections.abc import Sequence

from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.harnesses.environment import (
    client_environment,
    require_unset_environment,
)
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_INSTALL_HINT = (
    "Install Grok Build with `irm https://x.ai/cli/install.ps1 | iex` on Windows "
    "or `curl -fsSL https://x.ai/cli/install.sh | bash` on macOS/Linux."
)
_VERSION_PATTERN = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?"
    r"(?:\s+\([^\r\n]*\))?$"
)
_PROCESS_CONFIG_KEYS = ("GROK_CONFIG", "GROK_CONFIG_PATH")
_ROUTING_ENV_KEYS = frozenset(
    {
        "GROK_CODE_XAI_API_KEY",
        "GROK_DEFAULT_MODEL",
        "GROK_IMAGE_DESCRIPTION_MODEL",
        "GROK_MODELS_BASE_URL",
        "GROK_MODELS_LIST_URL",
        "GROK_PROMPT_SUGGESTIONS_MODEL",
        "GROK_SESSION_SUMMARY_MODEL",
        "GROK_WEB_SEARCH_MODEL",
        "GROK_XAI_API_BASE_URL",
        "XAI_API_KEY",
    }
)


def _compatible_version(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    version = payload.get("currentVersion")
    return isinstance(version, str) and version_at_least(
        version.strip(), _VERSION_PATTERN, (1, 0, 5)
    )


def _configure(
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    require_unset_environment(ctx.base_env, _PROCESS_CONFIG_KEYS)
    v1_url = proxy_v1_url(ctx.proxy_root_url)
    default_model = ctx.models[0].wire_slug
    env = client_environment(
        ctx.base_env,
        proxy_root_url=ctx.proxy_root_url,
        remove_keys=tuple(_ROUTING_ENV_KEYS),
        remove_prefixes=("FCC_GROK_",),
        updates={
            "GROK_XAI_API_BASE_URL": v1_url,
            "GROK_MODELS_BASE_URL": v1_url,
            "GROK_MODELS_LIST_URL": f"{v1_url}/models?view=responses",
            "GROK_DEFAULT_MODEL": default_model,
            "GROK_IMAGE_DESCRIPTION_MODEL": default_model,
            "GROK_PROMPT_SUGGESTIONS_MODEL": default_model,
            "GROK_SESSION_SUMMARY_MODEL": default_model,
            "GROK_WEB_SEARCH_MODEL": default_model,
            "XAI_API_KEY": ctx.auth_token,
            "GROK_CONFIG": json.dumps(
                {
                    "models": {
                        "allowed_models": [model.wire_slug for model in ctx.models]
                    },
                    "shell_environment_policy": {"ignore_default_excludes": False},
                },
                separators=(",", ":"),
            ),
        },
    )
    if args and args[0] == "agent":
        command = [
            ctx.binary_path,
            "--disable-web-search",
            "agent",
            "--no-leader",
            *args[1:],
        ]
    else:
        command = [ctx.binary_path, "--disable-web-search", "--no-leader", *args]
    return PreparedLaunch(command, env)


SPEC = HarnessSpec(
    binary_name="grok",
    display_name="Grok Build",
    install_hint=_INSTALL_HINT,
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("version", "--json"),
        _compatible_version,
        "FCC requires Grok Build 1.0.5 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
