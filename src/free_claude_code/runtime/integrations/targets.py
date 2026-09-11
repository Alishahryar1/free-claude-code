"""Explicit settings for the four supported Admin integration items."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from free_claude_code.application.integrations import IntegrationError, IntegrationId
from free_claude_code.cli.claude_env import claude_proxy_values
from free_claude_code.cli.launchers.codex import codex_config_values
from free_claude_code.cli.launchers.model_catalog import (
    ClientModel,
    catalog_wire_slug_for_ref,
)
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings

from .discovery import InstalledClients, LocalInstallations
from .documents import MISSING, JsonDocument, SettingPath, TomlDocument

ENV_SETTING = "claudeCode.environmentVariables"
AGENT_PATH = ("agent_servers", "Claude Code (FCC)")
type Document = JsonDocument | TomlDocument


@dataclass(frozen=True)
class Connection:
    settings: Settings
    models: tuple[ClientModel, ...]
    catalog_path: Path
    saved_auth_token: str

    @property
    def url(self) -> str:
        return local_proxy_root_url(self.settings)

    @property
    def model(self) -> str | None:
        return catalog_wire_slug_for_ref(self.models, self.settings.model)


def same_url(value: object, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        first, second = urlsplit(value), urlsplit(expected)
        local = {"localhost", "127.0.0.1", "::1"}
        return (
            first.scheme == second.scheme
            and first.port == second.port
            and first.path.rstrip("/") == second.path.rstrip("/")
            and not (first.username or first.password or first.query or first.fragment)
            and (
                first.hostname == second.hostname
                or {first.hostname, second.hostname} <= local
            )
        )
    except ValueError:
        return False


@dataclass(frozen=True)
class Target:
    id: IntegrationId
    path: Path
    title: str
    recipe: dict[str, object]
    missing: tuple[str, ...]
    badges: tuple[str, ...]
    instructions: str
    documentation_url: str

    def document(self, source: bytes | None) -> Document:
        if self.id == IntegrationId.CODEX:
            return TomlDocument(source if source is not None else b"")
        return JsonDocument(
            source if source is not None else b"{}\n",
            jsonc=self.id == IntegrationId.CLAUDE_VSCODE,
        )

    def containers(self) -> tuple[tuple[str, ...], ...]:
        if self.id == IntegrationId.CODEX:
            return (
                ("model_providers",),
                ("model_providers", "fcc"),
                ("model_providers", "fcc", "auth"),
            )
        if self.id == IntegrationId.CLAUDE_VSCODE:
            return ((ENV_SETTING,),)
        if self.id == IntegrationId.CLAUDE_JETBRAINS:
            return (("agent_servers",), AGENT_PATH, (*AGENT_PATH, "env"))
        return ()

    def _env_index(self, document: Document, name: str) -> int | None:
        entries = document.get((ENV_SETTING,))
        if entries is MISSING:
            return None
        if not isinstance(entries, list):
            raise IntegrationError(
                "Claude's environmentVariables setting must be an array."
            )
        matches = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(
                document.get((ENV_SETTING, index, "name")), str
            ):
                raise IntegrationError(
                    "Claude's environmentVariables entries must have a string name."
                )
            if document.get((ENV_SETTING, index, "name")) == name:
                matches.append(index)
        if len(matches) > 1:
            raise IntegrationError(
                f"Remove duplicate environment entries for {name} before continuing."
            )
        return matches[0] if matches else None

    def path_for(self, key: str) -> SettingPath:
        if self.id == IntegrationId.CLAUDE_VSCODE:
            return (key,)
        if self.id == IntegrationId.CLAUDE_JETBRAINS:
            return (*AGENT_PATH, *key.split("."))
        return tuple(key.split("."))

    def read(self, document: Document, key: str) -> object:
        if self.id == IntegrationId.CLAUDE_VSCODE and key.startswith("env."):
            index = self._env_index(document, key[4:])
            return (
                MISSING
                if index is None
                else document.get((ENV_SETTING, index, "value"))
            )
        return document.get(self.path_for(key))

    def write(self, document: Document, key: str, value: object) -> None:
        if self.id == IntegrationId.CLAUDE_VSCODE and key.startswith("env."):
            name = key[4:]
            index = self._env_index(document, name)
            if index is None:
                if value is MISSING:
                    return
                if document.get((ENV_SETTING,)) is MISSING:
                    document.set((ENV_SETTING,), [])
                entries = document.get((ENV_SETTING,))
                assert isinstance(entries, list)
                document.set(
                    (ENV_SETTING, len(entries)), {"name": name, "value": value}
                )
            elif value is MISSING:
                document.delete((ENV_SETTING, index))
            else:
                document.set((ENV_SETTING, index, "value"), value)
            return
        if value is MISSING:
            document.delete(self.path_for(key))
        else:
            document.set(self.path_for(key), value)

    def values(self, document: Document) -> dict[str, object]:
        values = {key: self.read(document, key) for key in self.recipe}
        for key, value in values.items():
            if value is MISSING:
                continue
            if key in {
                "hasCompletedOnboarding",
                "claudeCode.disableLoginPrompt",
                "model_providers.fcc.requires_openai_auth",
            }:
                valid = type(value) is bool
            elif key.endswith("args"):
                valid = isinstance(value, list) and all(
                    isinstance(arg, str) for arg in value
                )
            else:
                valid = isinstance(value, str)
            if not valid:
                raise IntegrationError(
                    f"The setting {key} has an unsupported value type."
                )
        return values

    def recognized(self, values: dict[str, object], connection: Connection) -> bool:
        if self.id == IntegrationId.CODEX:
            command = values.get("model_providers.fcc.auth.command")
            helper = (
                isinstance(command, str)
                and Path(command).name in {"fcc-codex", "fcc-codex.exe"}
                and values.get("model_providers.fcc.auth.args")
                == ["--print-proxy-auth-token"]
            )
            return values.get("model_provider") == "fcc" and (
                values.get("model_providers.fcc.name") == "Free Claude Code" or helper
            )
        if self.id == IntegrationId.CLAUDE_LOGIN:
            return values.get("hasCompletedOnboarding") is True
        return (
            same_url(values.get("env.ANTHROPIC_BASE_URL"), connection.url)
            and values.get("env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"
        )

    def manual_owned(self, key: str, value: object, connection: Connection) -> bool:
        if value is MISSING:
            return False
        if self.id == IntegrationId.CODEX:
            if key == "model":
                return value in {model.wire_slug for model in connection.models}
            if key == "model_catalog_json":
                return isinstance(value, str) and Path(value) == connection.catalog_path
            return key.startswith("model_providers.fcc.") or (
                key == "model_provider" and value == "fcc"
            )
        if key == "env.ANTHROPIC_AUTH_TOKEN":
            return True
        if key == "env.ANTHROPIC_BASE_URL":
            return same_url(value, connection.url)
        return value == self.recipe[key]


def make_targets(
    locator: LocalInstallations, installed: InstalledClients, connection: Connection
) -> tuple[Target, ...]:
    claude = {
        f"env.{key}": value
        for key, value in claude_proxy_values(
            proxy_root_url=connection.url,
            auth_token=connection.settings.proxy_auth_token,
        ).items()
    }
    claude["env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = ""
    codex: dict[str, object] = dict(
        codex_config_values(
            api_url=connection.url,
            model=connection.model,
            auth_command=str(installed.fcc_command or "fcc-codex"),
        )
    )
    codex["model"] = connection.model
    codex["model_catalog_json"] = str(connection.catalog_path)
    for name in ("env_key", "experimental_bearer_token", "requires_openai_auth"):
        codex[f"model_providers.fcc.{name}"] = MISSING
    codex_missing = []
    if not installed.codex_app and not installed.codex_vscode:
        codex_missing.append(
            "Install Codex App or the Codex extension in VS Code's default local profile."
        )
    if installed.fcc_command is None:
        codex_missing.append(
            "The installed fcc-codex credential helper was not found. Reinstall FCC using its normal installer."
        )
    jetbrains_missing = []
    if not installed.jetbrains:
        jetbrains_missing.append(
            "A supported JetBrains IDE was not detected in the standard installation locations."
        )
    if not installed.acp_command:
        jetbrains_missing.append(
            "Install the Claude ACP adapter and its required Node runtime. A usable installed adapter was not detected."
        )
    shared = (installed.scope_issue,) if installed.scope_issue else ()
    return (
        Target(
            IntegrationId.CLAUDE_VSCODE,
            locator.vscode_settings,
            "Claude Code in VS Code",
            {"claudeCode.disableLoginPrompt": True, **claude},
            shared
            + (
                ()
                if installed.claude_vscode
                else (
                    "Install the Claude Code extension in VS Code's default local profile.",
                )
            ),
            ("VS Code extension installed",) if installed.claude_vscode else (),
            "Run Developer: Reload Window in VS Code, then start a new Claude session.",
            "https://code.claude.com/docs/en/vs-code",
        ),
        Target(
            IntegrationId.CODEX,
            locator.home / ".codex/config.toml",
            "Codex App and VS Code",
            codex,
            shared + tuple(codex_missing),
            tuple(
                label
                for found, label in (
                    (installed.codex_app, "App installed"),
                    (installed.codex_vscode, "VS Code extension installed"),
                )
                if found
            ),
            "Restart Codex App and reload VS Code. This shared configuration also affects normal Codex CLI use.",
            "https://learn.chatgpt.com/docs/developer-settings",
        ),
        Target(
            IntegrationId.CLAUDE_JETBRAINS,
            locator.home / ".jetbrains/acp.json",
            "Claude Code in JetBrains",
            {
                "command": installed.acp_command[0] if installed.acp_command else "",
                "args": list(installed.acp_command[1:]),
                **claude,
            },
            shared + tuple(jetbrains_missing),
            ("JetBrains IDE installed",) if installed.jetbrains else (),
            "Restart the IDE and select Claude Code (FCC) in a new chat. The existing registry agent is unchanged.",
            "https://www.jetbrains.com/help/ai-assistant/acp.html",
        ),
        Target(
            IntegrationId.CLAUDE_LOGIN,
            locator.home / ".claude.json",
            "Fix Claude Code login prompt",
            {"hasCompletedOnboarding": True},
            shared
            + (
                ()
                if installed.claude_command
                or installed.claude_vscode
                or (installed.jetbrains and installed.acp_command)
                else ("An installed Claude Code client was not detected.",)
            ),
            (),
            "Restart Claude Code or your IDE. This changes shared first-run state for this user; it does not sign in to an account.",
            "https://github.com/Alishahryar1/free-claude-code#connect-your-client",
        ),
    )
