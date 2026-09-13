"""Installed Cline launcher with private native provider settings."""

import re
from collections.abc import Sequence

from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .cline_config import CLINE_PROVIDER_ID, build_cline_config
from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_VERSION_PATTERN = re.compile(
    r"(?m)^\s*(?:cline(?:\s+version)?\s+|v)?"
    r"(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?\s*$"
)


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    config = build_cline_config(
        ctx.models,
        proxy_root_url=ctx.proxy_root_url,
        auth_token=ctx.auth_token,
        launch_id=ctx.launch_id,
    )
    providers_path = files.write_json("settings/providers.json", config.providers)
    files.write_json("settings/models.json", config.models)
    return PreparedLaunch(
        [ctx.binary_path, "--provider", CLINE_PROVIDER_ID, *args],
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            updates={
                "CLINE_PROVIDER_SETTINGS_PATH": str(providers_path),
                "CLINE_SESSION_BACKEND_MODE": "local",
            },
        ),
    )


SPEC = HarnessSpec(
    binary_name="cline",
    display_name="Cline CLI",
    install_hint="Install Cline from: https://docs.cline.bot/getting-started/installing-cline",
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (3, 0, 55)),
        "FCC requires Cline 3.0.55 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
