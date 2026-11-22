"""Installed Claude launcher: native command handling with an FCC connection."""

from collections.abc import Sequence
from pathlib import Path

from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.harnesses.claude import (
    CLAUDE_BINARY_NAME,
    build_claude_proxy_env,
    claude_session_settings,
)
from free_claude_code.harnesses.launch import PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness


def _take_user_settings(args: list[str]) -> tuple[list[str], JsonObject | None]:
    """Remove the user's ``--settings`` flags; Claude keeps only the last one."""
    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            remaining.extend(args[index:])
            break
        if arg == "--settings" and index + 1 < len(args):
            value = args[index + 1]
            index += 2
            continue
        if arg.startswith("--settings="):
            value = arg.removeprefix("--settings=")
        else:
            remaining.append(arg)
        index += 1
    if value is None:
        return remaining, None
    # Claude accepts either inline JSON or a settings file path.
    source = (
        value
        if value.lstrip().startswith("{")
        else Path(value).read_text(encoding="utf-8")
    )
    settings = load_json_value(source)
    if not isinstance(settings, dict):
        raise ValueError("Claude --settings must be a JSON object")
    return remaining, settings


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    args, user_settings = _take_user_settings(args)
    settings_path = files.write_json(
        "claude-settings.json",
        claude_session_settings(ctx.proxy_root_url, ctx.auth_token, user_settings),
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
