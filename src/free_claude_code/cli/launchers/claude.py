"""Installed Claude launcher: native command handling with an FCC connection."""

from collections.abc import Sequence

from free_claude_code.harnesses.claude import CLAUDE_BINARY_NAME, build_claude_proxy_env
from free_claude_code.harnesses.launch import PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness


def _configure(
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    return PreparedLaunch(
        [ctx.binary_path, *args],
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
