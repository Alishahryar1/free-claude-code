"""Inspect, preview, and confirm one local integration change at a time."""

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass

from free_claude_code.application.integrations import IntegrationError
from free_claude_code.application.integrations import IntegrationId as Item
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.core.json_types import JsonObject

from .discovery import InstalledClients, LocalInstallations
from .documents import MISSING, JsonDocument
from .storage import FileSnapshot, atomic_write, check_path
from .targets import AGENT_PATH, ENV_SETTING, Connection, Target, make_targets, same_url


@dataclass
class Capture:
    target: Target
    source: FileSnapshot
    dependencies: tuple[FileSnapshot, ...]
    installed: InstalledClients


@dataclass
class Prepared:
    capture: Capture
    output: bytes | None
    preview: JsonObject


def _codex_extension_in_wsl(vscode: FileSnapshot) -> bool:
    value = (
        JsonDocument(vscode.data, jsonc=True).get(
            ("chatgpt.runCodexInWindowsSubsystemForLinux",)
        )
        if vscode.data is not None
        else MISSING
    )
    if value is MISSING:
        return False
    if type(value) is not bool:
        raise IntegrationError(
            "The Codex extension's WSL setting must be a boolean. Correct the VS Code settings before continuing."
        )
    return value


class IntegrationService:
    def __init__(self, installations: LocalInstallations) -> None:
        self.installations = installations
        self._lock_path = installations.home / ".fcc/integrations/write.lock"
        self._revision_key = secrets.token_bytes(32)

    def _capture(
        self, target: Target, installed: InstalledClients, connection: Connection
    ) -> Capture:
        source = FileSnapshot.read(target.path)
        dependencies = []
        if target.id in {Item.CLAUDE_VSCODE, Item.CLAUDE_JETBRAINS}:
            dependencies.append(
                FileSnapshot.read(self.installations.home / ".claude/settings.json")
            )
        if target.id == Item.CODEX:
            dependencies.append(FileSnapshot.read(connection.catalog_path))
            if not installed.codex_app and installed.codex_vscode:
                vscode = FileSnapshot.read(self.installations.vscode_settings)
                if _codex_extension_in_wsl(vscode):
                    raise IntegrationError(
                        "This Codex extension runs in WSL and needs manual setup. Its WSL mode will not be changed."
                    )
                dependencies.append(vscode)
            if installed.fcc_command is not None:
                dependencies.append(FileSnapshot.read(installed.fcc_command))
        if target.id == Item.CLAUDE_JETBRAINS:
            dependencies.extend(
                FileSnapshot.read(self.installations.home / path)
                for path in installed.acp.command
            )
            if installed.acp.manifest is not None:
                dependencies.append(FileSnapshot.read(installed.acp.manifest))
        return Capture(target, source, tuple(dependencies), installed)

    def _revision(self, capture: Capture, connection: Connection) -> str:
        context = (
            {}
            if capture.target.id == Item.CLAUDE_LOGIN
            else {
                "url": connection.url,
                "token": connection.settings.proxy_auth_token,
                "auth_enabled": connection.settings.proxy_auth_enabled,
                "saved_auth_token": connection.saved_auth_token,
                "model": connection.model,
                "models": [model.wire_slug for model in connection.models],
                "catalog": str(connection.catalog_path),
            }
        )
        payload = {
            "id": capture.target.id,
            "context": context,
            "installed": asdict(capture.installed),
            "files": [
                (str(file.path), file.digest, file.identity)
                for file in (capture.source, *capture.dependencies)
            ],
        }
        return hmac.new(
            self._revision_key,
            json.dumps(payload, sort_keys=True, default=str).encode(),
            hashlib.sha256,
        ).hexdigest()

    def _claude_conflicts(self, capture: Capture, connection: Connection) -> None:
        target = capture.target
        document = target.document(capture.source.data)
        environments: list[dict[str, object]] = []
        if target.id == Item.CLAUDE_VSCODE:
            entries = document.get((ENV_SETTING,))
            if isinstance(entries, list):
                extra = {
                    entry["name"]: entry.get("value")
                    for entry in entries
                    if isinstance(entry, dict)
                    and isinstance(entry.get("name"), str)
                    and f"env.{entry['name']}" not in target.recipe
                }
                environments.append(extra)
        else:
            entries = document.get((*AGENT_PATH, "env"))
            if isinstance(entries, dict):
                environments.append(
                    {
                        key: value
                        for key, value in entries.items()
                        if f"env.{key}" not in target.recipe
                    }
                )
        global_settings = capture.dependencies[0]
        if global_settings.data is not None:
            global_env = JsonDocument(global_settings.data, jsonc=True).get(("env",))
            if global_env is not MISSING and not isinstance(global_env, dict):
                raise IntegrationError(
                    "The user-level Claude env setting must be an object. Correct .claude/settings.json first."
                )
            if isinstance(global_env, dict):
                environments.append(global_env)
        for environment in environments:
            for key, value in environment.items():
                if not (
                    key.startswith("ANTHROPIC_")
                    or key.startswith("CLAUDE_CODE_USE_")
                    or key
                    in {
                        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
                    }
                ):
                    continue
                expected = target.recipe.get(f"env.{key}", MISSING)
                matches = (
                    same_url(value, connection.url)
                    if key == "ANTHROPIC_BASE_URL"
                    else value == expected
                )
                if value not in (None, "") and not matches:
                    raise IntegrationError(
                        f"Conflicting {key} in Claude environment settings. Correct the editor environment or .claude/settings.json manually before setup."
                    )

    def _require_ready(
        self, capture: Capture, connection: Connection, *, needs_default: bool
    ) -> None:
        target = capture.target
        if target.missing:
            raise IntegrationError(" ".join(target.missing))
        if target.id == Item.CODEX:
            if (
                connection.settings.proxy_auth_enabled
                and connection.saved_auth_token != connection.settings.proxy_auth_token
            ):
                raise IntegrationError(
                    "Save the FCC authentication token in its user configuration before setup; a desktop helper cannot inherit server-only credentials."
                )
            catalog = capture.dependencies[0]
            try:
                data = json.loads(catalog.data) if catalog.data is not None else None
                models = data.get("models", []) if isinstance(data, dict) else []
                slugs = {
                    model["slug"]
                    for model in models
                    if isinstance(model, dict) and isinstance(model.get("slug"), str)
                }
            except ValueError:
                slugs = set()
            if (
                not connection.models
                or not slugs
                or (needs_default and connection.model not in slugs)
            ):
                raise IntegrationError(
                    "FCC's model catalog/default is not ready. Select an available FCC model before setup."
                )

    def _prepare(self, capture: Capture, connection: Connection) -> Prepared:
        target = capture.target
        document = target.document(capture.source.data)
        original = document.render()
        values = target.values(document)
        desired = dict(target.recipe)
        needs_default = True
        if target.id == Item.CODEX:
            model = values["model"]
            if (
                values["model_provider"] == "fcc"
                and isinstance(model, str)
                and model.strip()
            ):
                desired["model"] = model
                needs_default = False
            key = "model_providers.fcc.requires_openai_auth"
            if values[key] is False:
                desired[key] = False
        self._require_ready(capture, connection, needs_default=needs_default)
        if target.id in {Item.CLAUDE_VSCODE, Item.CLAUDE_JETBRAINS}:
            self._claude_conflicts(capture, connection)
        for key, value in desired.items():
            target.write(document, key, value)
        target.validate_values(target.values(document))
        rendered = document.render()
        output = rendered if rendered != original else None
        if output is not None:
            capture.source.require_writable()
        summary = (
            "Apply the Claude Code first-run fix for this user."
            if target.id == Item.CLAUDE_LOGIN
            else "Apply replaces FCC connection settings, including manual changes to those fields. Manual disconnect instructions are on the card."
        )
        if output is not None:
            clients = {
                Item.CLAUDE_VSCODE: "VS Code",
                Item.CODEX: "these clients",
                Item.CLAUDE_JETBRAINS: "your JetBrains IDEs",
                Item.CLAUDE_LOGIN: "Claude Code and IDE sessions using it",
            }[target.id]
            summary = (
                f"Save and close {clients} before confirming; avoid other edits until FCC finishes. "
                + summary
            )
        if target.id == Item.CODEX:
            summary = (
                "This file is shared by Codex App, VS Code, and the CLI. " + summary
            )
        preview: JsonObject = {
            "id": target.id,
            "title": target.title,
            "path": str(target.path),
            "summary": summary,
            "writes_file": output is not None,
            "instructions": target.instructions,
            "revision": self._revision(capture, connection),
        }
        return Prepared(capture, output, preview)

    def _one(self, item: Item, connection: Connection) -> Prepared:
        installed = self.installations.scan()
        target = next(
            target
            for target in make_targets(self.installations, installed, connection)
            if target.id == item
        )
        return self._prepare(self._capture(target, installed, connection), connection)

    def preview(self, item: Item, connection: Connection) -> JsonObject:
        return self._one(item, connection).preview

    def inspect(self, connection: Connection) -> JsonObject:
        installed = self.installations.scan()
        items: list[JsonObject] = []
        for target in make_targets(self.installations, installed, connection):
            entry: JsonObject = {
                "id": target.id,
                "title": target.title,
                "path": str(target.path),
                "badges": list(target.badges),
                "missing": list(target.missing),
                "documentation_url": target.documentation_url,
                "instructions": target.instructions,
                "can_apply": False,
                "status": "not_configured",
                "message": "",
            }
            try:
                prepared = self._prepare(
                    self._capture(target, installed, connection), connection
                )
                entry["can_apply"] = True
                if prepared.output is None:
                    entry["status"] = "configured"
                    if target.id == Item.CLAUDE_LOGIN:
                        entry["message"] = (
                            "Onboarding already completed. Remaining login failures require checking the connection."
                        )
            except IntegrationError as error:
                entry["status"], entry["message"] = "needs_attention", str(error)
            if target.id == Item.CODEX and installed.codex_vscode:
                vscode_path = self.installations.vscode_settings
                try:
                    if _codex_extension_in_wsl(FileSnapshot.read(vscode_path)):
                        entry["badges"] = [
                            *target.badges,
                            "VS Code uses WSL: manual setup",
                        ]
                except IntegrationError as error:
                    entry["message"] = (
                        f"VS Code extension information unavailable ({vscode_path}): {error}"
                    )
            items.append(entry)
        return {
            "items": items,
            "scope": "Standard local installations for this OS and user account; VS Code default profile.",
        }

    def apply(self, item: Item, revision: str, connection: Connection) -> JsonObject:
        lock = InterprocessFileLock(self._lock_path)
        try:
            check_path(self._lock_path.parent)
            self._lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not lock.acquire(wait=True, timeout=5):
                raise IntegrationError(
                    "Another integration change is in progress. Refresh and try again.",
                    status_code=409,
                )
            prepared = self._one(item, connection)
            current_revision = prepared.preview["revision"]
            assert isinstance(current_revision, str)
            if not hmac.compare_digest(revision, current_revision):
                raise IntegrationError(
                    "Settings changed since this preview. Refresh and confirm the new changes.",
                    status_code=409,
                )
            capture = prepared.capture
            for snapshot in (capture.source, *capture.dependencies):
                snapshot.require_unchanged()
            output = prepared.output
            if output is None:
                return {
                    "applied": False,
                    "instructions": prepared.preview["instructions"],
                    "message": "No client settings changed.",
                }
            atomic_write(
                capture.target.path,
                output,
                mode=capture.source.identity[3] if capture.source.identity else 0o600,
                expected=capture.source,
            )
            actual = FileSnapshot.read(capture.target.path)
            if actual.data != output:
                raise IntegrationError(
                    "The file changed during verification. No further changes were made; inspect it before retrying.",
                    status_code=409,
                )
            capture.target.document(actual.data)
            return {
                "applied": True,
                "instructions": capture.target.instructions,
                "message": "Onboarding setting saved."
                if item == Item.CLAUDE_LOGIN
                else "Configuration saved.",
            }
        except OSError:
            raise IntegrationError(
                "Could not save configuration. Check file permissions and refresh before retrying."
            ) from None
        finally:
            lock.release()
