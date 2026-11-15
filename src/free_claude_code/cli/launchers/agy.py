"""Installed Antigravity CLI launcher with session-only FCC settings."""

import re
import sys
from collections.abc import Sequence

from free_claude_code.harnesses import antigravity
from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import (
    HarnessSpec,
    LaunchContext,
    launch_harness,
    version_at_least,
)

_VERSION_PATTERN = re.compile(r"(?m)^\s*(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?\s*$")
_INSTALL_HINT = (
    "Install Antigravity CLI with `irm https://antigravity.google/cli/install.ps1 | iex` "
    "on Windows or `curl -fsSL https://antigravity.google/cli/install.sh | bash` "
    "on macOS/Linux. Upgrade an existing installation with: agy update"
)


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    antigravity.cleanup_previous_connection(
        antigravity.config_path(), antigravity.state_path()
    )
    path = files.write_json(
        "antigravity-settings.json", antigravity.session_settings(catalog)
    )
    command = [ctx.binary_path, "--settings", str(path)]
    if args and not args[0].startswith("-"):
        # Global settings precede native subcommands; model is session-only.
        command.extend(args)
    else:
        command.extend(["--model", catalog.default_model_id, *args])
    return PreparedLaunch(
        command,
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            updates={
                "GEMINI_API_KEY": ctx.auth_token,
                "GOOGLE_GEMINI_BASE_URL": ctx.proxy_root_url.rstrip("/"),
            },
        ),
    )


SPEC = HarnessSpec(
    binary_name="agy",
    display_name="Antigravity CLI",
    install_hint=_INSTALL_HINT,
    configure=_configure,
    catalog_view="responses",
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (1, 5, 9)),
        "FCC requires Antigravity CLI 1.5.9 or newer for session-only settings.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    arguments = iter(args)
    for arg in arguments:
        if arg == "--":
            break
        if arg.split("=", 1)[0] in {"--settings", "-settings"}:
            print(
                "fcc-agy supplies --settings for the FCC connection. "
                "Use Antigravity's saved settings or other native options instead.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        # Prompt values are opaque, even when the entire prompt is a flag name.
        if arg in {
            "-p",
            "--p",
            "-print",
            "--print",
            "-prompt",
            "--prompt",
            "-i",
            "--i",
            "-prompt-interactive",
            "--prompt-interactive",
        }:
            next(arguments, None)
    launch_harness(SPEC, args)
