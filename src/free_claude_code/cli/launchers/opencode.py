"""Installed OpenCode launcher with process-local FCC configuration."""

import json
import re
from collections.abc import Sequence

from free_claude_code.harnesses.environment import (
    client_environment,
    require_unset_environment,
)
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .opencode_config import OPENCODE_API_KEY_ENV, build_opencode_config
from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_VERSION_PATTERN = re.compile(
    r"(?m)^\s*(?:(?:opencode(?:\s+version)?\s+)|v)?"
    r"(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?\s*$"
)
_PROCESS_CONFIG_KEYS = ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT")


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    require_unset_environment(ctx.base_env, _PROCESS_CONFIG_KEYS)
    config = build_opencode_config(ctx.models, proxy_root_url=ctx.proxy_root_url)
    path = files.write_json("opencode.json", config.file)
    return PreparedLaunch(
        [ctx.binary_path, *args],
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            remove_keys=_PROCESS_CONFIG_KEYS,
            remove_prefixes=("FCC_OPENCODE_",),
            updates={
                "OPENCODE_CONFIG": str(path),
                "OPENCODE_CONFIG_CONTENT": json.dumps(
                    config.overlay, separators=(",", ":")
                ),
                OPENCODE_API_KEY_ENV: ctx.auth_token,
            },
        ),
    )


SPEC = HarnessSpec(
    binary_name="opencode",
    display_name="OpenCode CLI",
    install_hint="Install OpenCode from: https://opencode.ai/docs/",
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (1, 18, 18)),
        "FCC requires OpenCode 1.18.18 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
