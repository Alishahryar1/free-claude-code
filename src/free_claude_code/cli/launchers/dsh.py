"""Installed DeepSeek Harness launcher with an FCC connection patch."""

import re
from collections.abc import Sequence

from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .dsh_config import DSH_API_KEY_ENV, DSH_ENV_PREFIX, build_dsh_launch_config
from .runner import HarnessSpec, LaunchContext, launch_harness

_VERSION_PATTERN = re.compile(
    r"(?im)^\s*(?:dsh\s+)?v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?)\s*$"
)


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    settings_path = files.write_json("settings.yaml", {})
    credentials_path = files.write_json(".credentials.yaml", {})
    config = build_dsh_launch_config(
        catalog.models,
        default_model_id=catalog.default_model_id,
        proxy_root_url=ctx.proxy_root_url,
        settings_path=settings_path,
        credentials_path=credentials_path,
        provider_progress_timeout=ctx.settings.provider_progress_timeout,
    )
    patch_path = files.write_json("fcc.patch.yml", config)
    patch_args = ["--patch", str(patch_path)]
    if not args:
        command = [ctx.binary_path, "web", *patch_args]
    elif args[0] == "web":
        command = [ctx.binary_path, "web", *patch_args, *args[1:]]
    else:
        command = [ctx.binary_path, "--profile", "web", *patch_args, *args]
    return PreparedLaunch(
        command,
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            remove_prefixes=(DSH_ENV_PREFIX,),
            updates={DSH_API_KEY_ENV: ctx.auth_token, "DSH_TELEMETRY_DISABLED": "1"},
        ),
    )


SPEC = HarnessSpec(
    binary_name="dsh",
    display_name="DeepSeek Harness",
    install_hint="Install the supported DeepSeek Harness release with: npm install -g @deepseek-ai/dsh@0.1.0-rc.8",
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: (
            bool(match := _VERSION_PATTERN.search(output))
            and match.group(1) == "0.1.0-rc.8"
        ),
        "FCC requires DeepSeek Harness 0.1.0-rc.8.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
