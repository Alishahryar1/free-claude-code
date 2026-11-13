"""Installed Antigravity CLI launcher with an explicit native FCC connection."""

import re
import sys
from collections.abc import Sequence

from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_admin_url
from free_claude_code.harnesses import antigravity_integration
from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .catalog_http import fetch_proxy_model_catalog
from .runner import (
    HarnessSpec,
    LaunchContext,
    LaunchError,
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
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    admin_url = ctx.proxy_root_url.rstrip("/") + "/admin/integrations"
    path = antigravity_integration.config_path()
    state = antigravity_integration.state_path()
    try:
        if not antigravity_integration.status(path, state)["connected"]:
            raise antigravity_integration.DisconnectedError()
        catalog = fetch_proxy_model_catalog(ctx.proxy_root_url, ctx.auth_token)
        antigravity_integration.configure(path, state, catalog)
    except antigravity_integration.DisconnectedError as exc:
        raise LaunchError(
            f"{exc}\nConnect it on the Integrations page first:\n{admin_url}"
        ) from None
    if args and not args[0].startswith("-"):
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
    compatibility_check=NativeCheck(
        ("--version",),
        lambda output: version_at_least(output, _VERSION_PATTERN, (1, 4, 7)),
        "FCC requires Antigravity CLI 1.4.7 or newer.",
    ),
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--connect"], ["--disconnect"]):
        try:
            url = local_admin_url(get_settings()) + "/integrations"
        except OSError, ValueError:
            print(
                "Could not load FCC settings. Run fcc-server to check the configuration.",
                file=sys.stderr,
            )
        else:
            print(
                "Manage the Antigravity CLI connection on the Integrations page:\n"
                f"{url}\nStart FCC with fcc-server if it is not running.",
                file=sys.stderr,
            )
        raise SystemExit(1)
    launch_harness(SPEC, args)
