"""Installed Pi launcher using the bundled FCC provider extension."""

import re
import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_VERSION_PATTERN = re.compile(r"(?m)^\s*(\d+)\.(\d+)\.(\d+)\s*$")


def pi_extension_path() -> Path:
    return Path(__file__).with_name("pi_extension.ts").resolve()


def pi_install_hint(platform: str | None = None) -> str:
    if (platform or sys.platform) == "win32":
        return 'Install Pi with: powershell -c "irm https://pi.dev/install.ps1 | iex"'
    return "Install Pi with: curl -fsSL https://pi.dev/install.sh | sh"


def _configure(
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    extension_path = pi_extension_path()
    if not extension_path.is_file():
        raise ValueError(
            "Free Claude Code's bundled Pi extension is missing. Reinstall FCC."
        )
    return PreparedLaunch(
        [
            ctx.binary_path,
            "-e",
            str(extension_path),
            "--models",
            "free-claude-code/**",
            *args,
        ],
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            remove_prefixes=("FCC_PI_",),
            updates={
                "FCC_PI_BASE_URL": ctx.proxy_root_url.rstrip("/"),
                "FCC_PI_API_KEY": ctx.auth_token,
            },
        ),
    )


SPEC = HarnessSpec(
    binary_name="pi",
    display_name="Pi",
    install_hint=pi_install_hint(),
    configure=_configure,
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (0, 80, 4)),
        "FCC requires Pi 0.80.4 or newer for provider request headers.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
