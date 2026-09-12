"""Installed Cline launcher with private native provider settings.

Sessions run with an isolated Hub discovery path so a stale or user-run Cline
Hub can never capture an fcc-cline session: the child resolves Hub state only
inside FCC's ephemeral configuration directory, which is deleted on exit. A
Hub started with ordinary cline keeps using its own discovery file.
"""

import re
from collections.abc import Sequence

from free_claude_code.cli.environment import client_environment

from .cline_config import CLINE_PROVIDER_ID, build_cline_config
from .resources import LaunchResources
from .runner import (
    HarnessSpec,
    LaunchContext,
    NativeCheck,
    PreparedLaunch,
    launch_harness,
    version_at_least,
)

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
    # Interactive Cline passes backendMode:"auto" to core, which overrides
    # CLINE_SESSION_BACKEND_MODE and may attach to (or start) a detached Hub
    # that resolves provider settings from the user's global store instead of
    # FCC's ephemeral files. Redirecting Hub discovery into the ephemeral tree
    # makes that attachment impossible without touching user-run Hub state.
    # (Verified: shipped Cline 3.0.57 honors CLINE_HUB_DISCOVERY_PATH as the
    # discovery-path override in its hub resolver.)
    hub_discovery_path = files.directory / "hub" / "production.json"
    # Pre-create the directory, not the file: the child records Hub discovery
    # state here on startup, and the Hub itself must own the file.
    hub_discovery_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    return PreparedLaunch(
        [ctx.binary_path, "--provider", CLINE_PROVIDER_ID, *args],
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            updates={
                "CLINE_PROVIDER_SETTINGS_PATH": str(providers_path),
                "CLINE_SESSION_BACKEND_MODE": "local",
                "CLINE_HUB_DISCOVERY_PATH": str(hub_discovery_path),
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
