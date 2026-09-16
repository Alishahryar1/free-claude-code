"""Installed Hermes launcher with a native managed provider overlay."""

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.harnesses.environment import (
    client_environment,
    require_unset_environment,
)
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .hermes_config import HERMES_KEY_ENV_PREFIX, build_hermes_managed_config
from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_INSTALL_HINT = (
    "Install Hermes Agent from: https://hermes-agent.nousresearch.com/docs/installation"
)
_VERSION_PATTERN = re.compile(r"(?i)\b(?:Hermes Agent\s+)?v?(\d+)\.(\d+)\.(\d+)\b")


def _overlay_loaded(output: str, expected_provider: str) -> bool:
    lines = output.strip().splitlines()
    try:
        return bool(lines) and json.loads(lines[-1]) == expected_provider
    except json.JSONDecodeError:
        return False


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    require_unset_environment(ctx.base_env, ("HERMES_MANAGED_DIR",))
    if os.name != "nt" and Path("/etc/hermes").exists():
        raise ValueError("An existing Hermes managed policy cannot be replaced by FCC.")
    managed = build_hermes_managed_config(
        catalog.models,
        default_model_id=catalog.default_model_id,
        proxy_root_url=ctx.proxy_root_url,
        nonce=ctx.launch_id,
    )
    path = files.write_json("config.yaml", managed.config)
    env = client_environment(
        ctx.base_env,
        proxy_root_url=ctx.proxy_root_url,
        remove_prefixes=(HERMES_KEY_ENV_PREFIX,),
        updates={
            "HERMES_MANAGED_DIR": str(path.parent),
            "HERMES_INFERENCE_PROVIDER": managed.provider_ref,
            "HERMES_INFERENCE_MODEL": managed.default_model,
            managed.key_env: ctx.auth_token,
        },
    )
    return PreparedLaunch(
        [ctx.binary_path, *args],
        env,
        NativeCheck(
            ("config", "get", "model.provider", "--json"),
            lambda output: _overlay_loaded(output, managed.provider_ref),
            "Could not activate the FCC Hermes managed provider.",
            timeout_seconds=15.0,
        ),
    )


SPEC = HarnessSpec(
    binary_name="hermes",
    display_name="Hermes Agent",
    install_hint=_INSTALL_HINT,
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (0, 20, 4)),
        "FCC requires Hermes 0.20.4 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
