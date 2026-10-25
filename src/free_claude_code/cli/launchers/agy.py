"""Installed Antigravity CLI launcher with an explicit native FCC connection."""

import re
import sys
from collections.abc import Sequence

from free_claude_code.harnesses import antigravity_integration
from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness, version_at_least

_VERSION_PATTERN = re.compile(r"(?m)^\s*(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?\s*$")
_INSTALL_HINT = (
    "Install Antigravity CLI with `irm https://antigravity.google/cli/install.ps1 | iex` "
    "on Windows or `curl -fsSL https://antigravity.google/cli/install.sh | bash` "
    "on macOS/Linux. Upgrade an existing installation with: agy update"
)


def _configure(
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    connect = args == ["--connect"]
    antigravity_integration.configure(
        antigravity_integration.config_path(),
        antigravity_integration.state_path(),
        catalog,
        connect=connect,
    )
    if connect:
        print(
            "Antigravity is configured for FCC. Launch with fcc-agy. "
            "Restore your previous provider with: fcc-agy --disconnect"
        )
        command = [ctx.binary_path, "models"]
    elif args and not args[0].startswith("-"):
        # Native subcommands dispatch before the session flag parser.
        command = [ctx.binary_path, *args]
    else:
        command = [ctx.binary_path, "--model", catalog.default_model_id, *args]
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
        lambda output: version_at_least(output, _VERSION_PATTERN, (1, 4, 7)),
        "FCC requires Antigravity CLI 1.4.7 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--disconnect"]:
        try:
            antigravity_integration.disconnect(
                antigravity_integration.config_path(),
                antigravity_integration.state_path(),
            )
        except (OSError, ValueError) as exc:
            print(f"Could not disconnect Antigravity: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print("Antigravity is disconnected from FCC.")
        return
    launch_harness(SPEC, args)
