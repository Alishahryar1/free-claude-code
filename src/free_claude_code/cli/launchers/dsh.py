"""Installed DeepSeek Harness launcher with an FCC connection patch."""

import os
import re
import sys
from collections.abc import Sequence

from free_claude_code.harnesses.dsh_config import (
    DSH_API_KEY_ENV,
    DSH_ENV_PREFIX,
    build_dsh_launch_config,
)
from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .common import resolve_client_binary, run_client_process
from .runner import HarnessSpec, LaunchContext, launch_harness

_VERSION_PATTERN = re.compile(
    r"(?im)^\s*(?:dsh\s+)?v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?\s*$"
)


def _supported_version(output: str) -> bool:
    match = _VERSION_PATTERN.search(output)
    if match is None:
        return False
    release = tuple(int(match[index]) for index in (1, 2, 3))
    preview = match[4].split(".") if match[4] is not None else []
    if any(part.isdigit() and len(part) > 1 and part[0] == "0" for part in preview):
        return False
    if release != (0, 2, 0):
        return release > (0, 2, 0)
    if not preview:
        return True
    if preview[0] != "rc":
        return not preview[0].isdigit() and preview[0] > "rc"
    return len(preview) > 1 and (not preview[1].isdigit() or int(preview[1]) >= 2)


def _launcher_options(args: list[str]) -> list[str]:
    """Scan option names only, stopping at DSH's first inner-app argument."""
    expanded = ["--profile", *args] if args and not args[0].startswith("-") else args
    index = 0
    options = []
    while index < len(expanded):
        option = expanded[index].split("=", 1)[0]
        if option in {"--profile", "--from-default-profile", "--patch"}:
            options.append(option)
            index += 1 if "=" in expanded[index] else 2
        elif option in {
            "--dump-config",
            "--dump-config-schema",
            "--dump-default-config",
            "--version",
            "-V",
        }:
            options.append(option)
            index += 1
        else:
            if option in {"--help", "-h"}:
                options.append(option)
            break
    return options


def _has_profile(prefix: list[str]) -> bool:
    return any(arg == "--profile" or arg.startswith("--profile=") for arg in prefix)


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    credentials_path = files.write_json(".credentials.yaml", {})
    config = build_dsh_launch_config(
        catalog.models,
        default_model_id=catalog.default_model_id,
        proxy_root_url=ctx.proxy_root_url,
        credentials_path=credentials_path,
        provider_progress_timeout=ctx.settings.provider_progress_timeout,
    )
    patch_path = files.write_json("fcc.patch.yml", config)
    patch_args = ["--patch", str(patch_path)]
    native_args = _launcher_options(args)
    if not args:
        command = [ctx.binary_path, "web", *patch_args]
    elif not args[0].startswith("-"):
        command = [ctx.binary_path, args[0], *patch_args, *args[1:]]
    elif _has_profile(native_args):
        command = [ctx.binary_path, *patch_args, *args]
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
    install_hint="Install DeepSeek Harness with: npm install -g @deepseek-ai/dsh@latest",
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        _supported_version,
        "FCC requires DeepSeek Harness >=0.2.0-rc.2.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    prefix = _launcher_options(args)
    if (
        args[:1] == ["plugin"]
        or any(flag in prefix for flag in ("--version", "-V", "--dump-default-config"))
        or (
            any(flag in prefix for flag in ("--help", "-h"))
            and not _has_profile(prefix)
        )
    ):
        binary = resolve_client_binary(
            binary_name=SPEC.binary_name,
            display_name=SPEC.display_name,
            install_hint=SPEC.install_hint,
        )
        run_client_process(
            command=[binary, *args],
            env=os.environ,
            binary_name=SPEC.binary_name,
            display_name=SPEC.display_name,
            install_hint=SPEC.install_hint,
        )
        return
    launch_harness(SPEC, args)
