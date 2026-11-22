"""Installed Claude launcher: native command handling with an FCC connection."""

from collections.abc import Sequence

from free_claude_code.harnesses.claude import (
    CLAUDE_BINARY_NAME,
    build_claude_proxy_env,
    claude_session_settings,
)
from free_claude_code.harnesses.launch import PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    # A user --settings flag replaces this one, so it is not supported with FCC.
    settings_path = files.write_json(
        "claude-settings.json",
        claude_session_settings(ctx.proxy_root_url, ctx.auth_token),
    )
    return PreparedLaunch(
        [ctx.binary_path, "--settings", str(settings_path), *args],
        build_claude_proxy_env(
            proxy_root_url=ctx.proxy_root_url,
            auth_token=ctx.auth_token,
            base_env=ctx.base_env,
        ),
    )


SPEC = HarnessSpec(
    binary_name=CLAUDE_BINARY_NAME,
    display_name="Claude Code",
    install_hint="Install Claude Code with: npm install -g @anthropic-ai/claude-code",
    configure=_configure,
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
