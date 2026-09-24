"""Installed OpenCode launcher with process-local FCC configuration."""

import json
from collections.abc import Sequence

from free_claude_code.harnesses.environment import (
    client_environment,
    require_unset_environment,
)
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.opencode import STABLE_VERSION_PATTERN
from free_claude_code.harnesses.resources import LaunchResources

from .opencode_config import OPENCODE_API_KEY_ENV, build_opencode_config
from .runner import HarnessSpec, LaunchContext, launch_harness

_PROCESS_CONFIG_KEYS = ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT")


def _standalone_args(args: list[str]) -> list[str]:
    end = args.index("--") if "--" in args else len(args)
    if any(
        arg.partition("=")[0] in {"--standalone", "--no-standalone"}
        for arg in args[:end]
    ):
        raise ValueError("FCC manages OpenCode standalone mode; omit that option.")
    # ACP already owns a private server and rejects the standalone flag.
    if args[:1] == ["acp"]:
        return args
    # OpenCode's flag belongs to the leaf command, before literal positionals.
    return [*args[:end], "--standalone", *args[end:]]


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    require_unset_environment(ctx.base_env, _PROCESS_CONFIG_KEYS)
    args = _standalone_args(args)
    config = build_opencode_config(
        catalog.models,
        default_model_id=catalog.default_model_id,
        proxy_root_url=ctx.proxy_root_url,
    )
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
    install_hint=(
        "Install or upgrade by rerunning FCC's installer with OpenCode selected. "
        "For external installations: https://opencode.ai/v2/docs/migrate-v1/"
    ),
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: STABLE_VERSION_PATTERN.search(output) is not None,
        "FCC requires stable OpenCode 2.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
