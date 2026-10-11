"""Private installer entry point for stock Headroom MCP registration."""

import argparse
import asyncio
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.cli.mcp_check import verify_mcp
from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.paths import managed_env_path
from free_claude_code.core.version import package_version
from free_claude_code.harnesses.headroom_clients import AGENTS, NativeClients
from free_claude_code.harnesses.headroom_desktops import prepare_desktops
from free_claude_code.harnesses.headroom_mcp import (
    McpSetupError,
    configure_registrations,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Configure Headroom for the installer-selected native clients."
    )
    parser.add_argument("--version", action="version", version=package_version())
    parser.add_argument("--headroom", required=True, type=Path)
    parser.add_argument("--agent", action="append", choices=AGENTS, default=[])
    parser.add_argument("--include-connected-desktops", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the command and preflight all targets without writing registrations.",
    )
    args = parser.parse_args(argv)
    try:
        if not args.headroom.is_absolute() or not args.headroom.is_file():
            raise McpSetupError(
                "Headroom must be an existing absolute executable path."
            )
        with tempfile.TemporaryDirectory(prefix="fcc-mcp-setup-") as folder:
            clients = NativeClients(args.headroom, os.environ, Path(folder))
            version = clients.run([str(args.headroom), "--version"])
            if "headroom" not in version.casefold():
                raise McpSetupError(
                    f"The command at {args.headroom} is not a compatible Headroom installation."
                )
            asyncio.run(verify_mcp([str(args.headroom), "mcp", "serve"]))
            targets = [
                target
                for agent in dict.fromkeys(args.agent)
                if (target := clients.prepare(agent)) is not None
            ]
            if args.include_connected_desktops:
                path = managed_env_path()
                managed = dotenv_values_from_file(path) if path.exists() else {}
                settings = compose_settings_snapshot(managed, os.environ).settings
                targets.extend(prepare_desktops(clients, settings))
            # Keep different profile checks even when they share one write scope.
            # The first target configures it; the second then sees it already set.
            results = configure_registrations(targets, check_only=args.check)
            for result in (*clients.outcomes, *results):
                print(result)
            if not targets:
                print(
                    "Headroom is installed; no supported agent scope was available for registration."
                )
    except McpSetupError as exc:
        print(f"Headroom setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except OSError, ValueError:
        # Native config parsing may include credential-bearing source in errors.
        print(
            "Headroom setup failed: a native or FCC configuration could not be read safely. Correct it and rerun the installer.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
