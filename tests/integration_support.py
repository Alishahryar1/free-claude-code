"""Synthetic installed clients and configuration homes for integration tests."""

import json
from pathlib import Path

from free_claude_code.cli.launchers.model_catalog import ClientModel
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.targets import Connection


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf-8")
    path.chmod(0o700)
    return path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FixtureInstallations(LocalInstallations):
    def _node_version(self, node: Path) -> tuple[int, int, int] | None:
        return (22, 0, 0) if node.is_file() else None


def installed_clients(home: Path) -> LocalInstallations:
    bins, apps = home / "bin", home / "apps"
    env = {"PATH": str(bins), "XDG_CONFIG_HOME": str(home / ".config")}
    locator = FixtureInstallations(home, "linux", env, bins, (apps,))
    for name in ("fcc-codex", "claude", "node", "chatgpt"):
        executable(bins / name)
    write_json(apps / "code/resources/app/product.json", {"applicationName": "code"})
    executable(apps / "code/code")
    descriptors = []
    for extension_id in ("anthropic.claude-code", "openai.chatgpt"):
        publisher, name = extension_id.split(".", 1)
        extension = home / ".vscode/extensions" / f"{extension_id}-1.0.0"
        write_json(
            extension / "package.json",
            {"publisher": publisher, "name": name, "version": "1.0.0"},
        )
        descriptors.append(
            {
                "identifier": {"id": extension_id},
                "relativeLocation": extension.name,
                "version": "1.0.0",
            }
        )
    write_json(home / ".vscode/extensions/extensions.json", descriptors)
    executable(apps / "idea/bin/idea.sh")
    write_json(
        apps / "idea/product-info.json",
        {"productCode": "IU", "launch": [{"launcherPath": "bin/idea.sh"}]},
    )
    package = home / "lib/node_modules/@agentclientprotocol/claude-agent-acp"
    write_json(
        package / "package.json",
        {
            "name": "@agentclientprotocol/claude-agent-acp",
            "bin": {"claude-agent-acp": "dist/index.js"},
            "engines": {"node": ">=22"},
        },
    )
    executable(package / "dist/index.js")
    return locator


def connection(
    home: Path, *, port: int = 8082, token: str = "fixture-proxy-token"
) -> Connection:
    catalog = home / ".fcc/codex-model-catalog.json"
    write_json(catalog, {"models": [{"slug": "test/model"}]})
    return Connection(
        Settings(
            MODEL="open_router/test-model",
            PORT=port,
            ANTHROPIC_AUTH_TOKEN=token,
            MESSAGING_PLATFORM="none",
        ),
        (
            ClientModel(
                "test/model",
                "open_router/test-model",
                "Test model",
                True,
                None,
                100_000,
                10_000,
            ),
        ),
        catalog,
        token,
    )
