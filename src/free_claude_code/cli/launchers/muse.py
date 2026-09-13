"""Installed Muse Code launcher with a native FCC connection."""

import re
from collections.abc import Sequence

from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_INSTALL_HINT = (
    "Install Muse Code on native Windows with "
    '`& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/'
    'Alishahryar1/free-claude-code/main/scripts/install-muse.ps1")))`, '
    "or on macOS/Linux/WSL with "
    "`curl -fsSL https://dev.meta.ai/install.sh | bash`."
)
_VERSION_PATTERN = re.compile(
    r"(?m)^\s*Muse Code\s+(\d+)\.(\d+)\.(\d+)"
    r"(?:\s+\([^\r\n]+\))?\s*$"
)
_ROUTING_ENV_KEYS = frozenset(
    {"META_API_KEY", "MUSE_MODEL", "MUSE_CUSTOM_HEADERS", "MUSE_WWW_ROUTING"}
)


def _configure(
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    connection = ["--provider", "meta", "--base-url", proxy_v1_url(ctx.proxy_root_url)]
    if args and args[0] in {"exec", "resume"}:
        command = [ctx.binary_path, args[0], *connection, *args[1:]]
    else:
        command = [ctx.binary_path, *connection, *args]
    return PreparedLaunch(
        command,
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            remove_keys=tuple(_ROUTING_ENV_KEYS),
            remove_prefixes=("FCC_MUSE_",),
            updates={
                "META_API_KEY": ctx.auth_token,
                "MUSE_MODEL": ctx.models[0].wire_slug,
            },
        ),
    )


SPEC = HarnessSpec(
    binary_name="muse",
    display_name="Muse Code",
    install_hint=_INSTALL_HINT,
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (0, 2, 1)),
        "FCC requires Muse Code 0.2.1 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
