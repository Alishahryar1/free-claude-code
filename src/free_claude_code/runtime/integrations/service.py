"""Inspect, preview, and confirm one local integration change at a time."""

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass

from pydantic import ValidationError

from free_claude_code.application.integrations import IntegrationAction as Action
from free_claude_code.application.integrations import IntegrationError, validate_action
from free_claude_code.application.integrations import IntegrationId as Item
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.core.json_types import JsonObject

from .discovery import AcpInstallation, InstalledClients, LocalInstallations
from .documents import MISSING, JsonDocument
from .storage import (
    FieldHistory,
    FileSnapshot,
    Ownership,
    PendingWrite,
    SavedValue,
    atomic_write,
    check_path,
    content_hash,
    write_record,
)
from .targets import AGENT_PATH, ENV_SETTING, Connection, Target, make_targets, same_url

_RECOVERY_GUIDE = (
    "https://github.com/Alishahryar1/free-claude-code#integration-recovery"
)


@dataclass(frozen=True)
class Recovery:
    action: Action
    completed: bool
    ownership: Ownership | None


@dataclass
class Capture:
    target: Target
    source: FileSnapshot
    record_file: FileSnapshot
    record: Ownership | None
    dependencies: tuple[FileSnapshot, ...]
    installed: InstalledClients
    recovery: Recovery | None


@dataclass
class Prepared:
    capture: Capture
    output: bytes | None
    following: Ownership | None
    record_changed: bool
    preview: JsonObject


def _validate_ownership(target: Target, record: Ownership | None) -> None:
    if record is None:
        return
    if (
        record.item != target.id
        or record.path != str(target.path)
        or not set(record.fields) <= target.recipe.keys()
        or any(
            tuple(path) not in target.containers() for path in record.created_containers
        )
    ):
        raise ValueError("target")
    target.validate_values(
        {key: field.last.unpack() for key, field in record.fields.items()}
    )
    target.validate_values(
        {
            key: field.prior.unpack()
            for key, field in record.fields.items()
            if field.prior is not None
        }
    )


def _codex_extension_in_wsl(capture: Capture) -> bool:
    vscode = capture.dependencies[1]
    return (
        vscode.data is not None
        and JsonDocument(vscode.data, jsonc=True).get(
            ("chatgpt.runCodexInWindowsSubsystemForLinux",)
        )
        is True
    )


def _display(key: str, value: object) -> str:
    if value is MISSING:
        return "Not set"
    if any(part in key.lower() for part in ("auth", "token", "env_key")):
        return "********"
    if (
        "base_url" in key.lower()
        and isinstance(value, str)
        and ("@" in value or "?" in value)
    ):
        return "********"
    return json.dumps(value, ensure_ascii=False)


class IntegrationService:
    def __init__(self, installations: LocalInstallations) -> None:
        self.installations = installations
        self.state_dir = installations.home / ".fcc/integrations"
        self._revision_key = secrets.token_bytes(32)

    def _capture(
        self, target: Target, installed: InstalledClients, connection: Connection
    ) -> Capture:
        source = FileSnapshot.read(target.path)
        record_file = FileSnapshot.read(self.state_dir / f"{target.id}.json")
        record = None
        recovery = None
        if record_file.data is not None:
            try:
                payload = json.loads(record_file.data)
                if isinstance(payload, dict) and payload.get("phase") == "pending":
                    pending = PendingWrite.model_validate(payload)
                    validate_action(target.id, pending.action)
                    if (
                        pending.item != target.id
                        or pending.path != str(target.path)
                        or pending.action == Action.RECOVER
                        or pending.before == pending.after
                    ):
                        raise ValueError("target")
                    _validate_ownership(target, pending.previous)
                    _validate_ownership(target, pending.following)
                    if source.digest not in {pending.before, pending.after}:
                        raise IntegrationError(
                            "An interrupted write needs attention because the file has changed. Use manual cleanup; FCC will not overwrite the uncertain file.",
                            status_code=409,
                        )
                    completed = source.digest == pending.after
                    recovery = Recovery(
                        pending.action,
                        completed,
                        pending.following if completed else pending.previous,
                    )
                else:
                    record = Ownership.model_validate(payload)
                    _validate_ownership(target, record)
            except ValueError, ValidationError:
                raise IntegrationError(
                    "The saved undo record cannot be read. Previous settings cannot be restored; use manual cleanup."
                ) from None
        if recovery is not None:
            return Capture(target, source, record_file, None, (), installed, recovery)
        dependencies = []
        if target.id in {Item.CLAUDE_VSCODE, Item.CLAUDE_JETBRAINS}:
            dependencies.append(
                FileSnapshot.read(self.installations.home / ".claude/settings.json")
            )
        if target.id == Item.CODEX:
            dependencies.append(FileSnapshot.read(connection.catalog_path))
            dependencies.append(FileSnapshot.read(self.installations.vscode_settings))
            if installed.fcc_command is not None:
                dependencies.append(FileSnapshot.read(installed.fcc_command))
        if target.id == Item.CLAUDE_JETBRAINS:
            dependencies.extend(
                FileSnapshot.read(self.installations.home / path)
                for path in installed.acp.command
            )
            if installed.acp.manifest is not None:
                dependencies.append(FileSnapshot.read(installed.acp.manifest))
        return Capture(
            target, source, record_file, record, tuple(dependencies), installed, None
        )

    def _revision(
        self, capture: Capture, action: Action, connection: Connection
    ) -> str:
        context = (
            {}
            if capture.target.id == Item.CLAUDE_LOGIN or action == Action.RECOVER
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
            "action": action,
            "context": context,
            "installed": {} if action == Action.RECOVER else asdict(capture.installed),
            "files": [
                (str(file.path), file.digest, file.identity)
                for file in (capture.source, capture.record_file, *capture.dependencies)
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
        self, capture: Capture, action: Action, connection: Connection
    ) -> None:
        target = capture.target
        if capture.installed.scope_issue:
            raise IntegrationError(capture.installed.scope_issue)
        if action == Action.DISCONNECT:
            return
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
                or (action == Action.SETUP and connection.model not in slugs)
            ):
                raise IntegrationError(
                    "FCC's model catalog/default is not ready. Select an available FCC model before setup."
                )
            if not capture.installed.codex_app and _codex_extension_in_wsl(capture):
                raise IntegrationError(
                    "This Codex extension runs in WSL and needs manual setup. Its WSL mode will not be changed."
                )

    def _prepare_recovery(self, capture: Capture, connection: Connection) -> Prepared:
        recovery = capture.recovery
        if recovery is None:
            raise IntegrationError(
                "There is no interrupted action to recover.", status_code=409
            )
        summary = (
            f"The file already matches the saved {recovery.action} result. Finish recovering FCC's setup history."
            if recovery.completed
            else "The file matches its earlier state. Recover FCC's setup history so you can retry the action."
        )
        preview: JsonObject = {
            "id": capture.target.id,
            "action": Action.RECOVER,
            "title": capture.target.title,
            "path": str(capture.target.path),
            "summary": summary,
            "writes_file": False,
            "changes": [],
            "notes": [],
            "instructions": capture.target.instructions if recovery.completed else "",
            "revision": self._revision(capture, Action.RECOVER, connection),
            "endpoint": None,
            "helper_path": None,
        }
        return Prepared(capture, None, recovery.ownership, True, preview)

    def _prepare(
        self, capture: Capture, action: Action, connection: Connection
    ) -> Prepared:
        target = capture.target
        validate_action(target.id, action)
        if action == Action.RECOVER:
            return self._prepare_recovery(capture, connection)
        if capture.recovery is not None:
            raise IntegrationError(
                "Recover the interrupted action before starting another change.",
                status_code=409,
            )
        document = target.document(capture.source.data)
        original = document.render()
        values = target.values(document)
        recognized = target.recognized(values, connection)
        self._require_ready(capture, action, connection)
        record = capture.record
        if action == Action.SETUP and (record is not None or recognized):
            raise IntegrationError(
                "An FCC setup already exists. Use Update or Disconnect; later edits will be preserved.",
                status_code=409,
            )
        if (
            action in {Action.UPDATE, Action.DISCONNECT}
            and record is None
            and not recognized
        ):
            raise IntegrationError(
                "An FCC connection could not be identified. Use Set up or follow the manual cleanup instructions."
            )
        if action != Action.DISCONNECT and target.id in {
            Item.CLAUDE_VSCODE,
            Item.CLAUDE_JETBRAINS,
        }:
            self._claude_conflicts(capture, connection)
            if (
                target.id == Item.CLAUDE_JETBRAINS
                and record is None
                and not recognized
                and document.get(AGENT_PATH) is not MISSING
            ):
                raise IntegrationError(
                    "A different agent already uses the name Claude Code (FCC). Rename that entry manually before setup."
                )
        desired = dict(target.recipe)
        if target.id == Item.CODEX:
            if action == Action.UPDATE:
                desired.pop("model", None)
            key = "model_providers.fcc.requires_openai_auth"
            if values[key] is False:
                desired[key] = False
        notes = [
            "Only the default local user profile is configured. Other profiles and remote environments use manual setup."
        ]
        fields = (
            {}
            if record is None
            else {
                key: value.model_copy(deep=True) for key, value in record.fields.items()
            }
        )
        containers = (
            [
                list(path)
                for path in target.containers()
                if document.get(path) is MISSING
            ]
            if record is None
            else record.created_containers
        )
        if action == Action.DISCONNECT:
            if record is None:
                notes.append(
                    "Previous settings are unknown. Remove only identifiable FCC values; the client may need its normal setup again."
                )
                desired = {
                    key: MISSING
                    for key, value in values.items()
                    if target.manual_owned(key, value, connection)
                }
            else:
                notes.append(
                    "Restore saved previous settings and remove values introduced by FCC. Later user edits stop the whole action."
                )
                desired = {
                    key: field.prior.unpack() if field.prior is not None else MISSING
                    for key, field in fields.items()
                }
            if (
                record is not None
                and target.id == Item.CLAUDE_JETBRAINS
                and list(AGENT_PATH) in record.created_containers
            ):
                agent = document.get(AGENT_PATH)
                expected: dict[str, object] = {}
                env: dict[str, object] = {}
                if isinstance(agent, dict) and "env" in agent:
                    expected["env"] = env
                for key in fields:
                    value = values[key]
                    if value is not MISSING:
                        if key.startswith("env."):
                            env[key[4:]] = value
                            expected["env"] = env
                        else:
                            expected[key] = value
                if agent is not MISSING and agent != expected:
                    raise IntegrationError(
                        "The Claude Code (FCC) agent was edited or has added fields. Preserve those changes and remove it manually.",
                        status_code=409,
                    )
        elif recognized and record is None and action == Action.UPDATE:
            notes.append(
                "Adopt this manual FCC connection for future updates. Previous non-FCC settings are unknown for its existing FCC values."
            )
            for key, value in values.items():
                if target.manual_owned(key, value, connection):
                    fields[key] = FieldHistory(
                        prior=None, last=SavedValue.capture(value)
                    )
        elif action == Action.REPAIR:
            notes.append(
                "Change only shared Claude first-run state. This does not authenticate an account, and Disconnect will not reset it."
            )
        else:
            notes.append(
                "Save previous values so Disconnect can restore them when they have not been edited afterward."
            )
        changes: list[JsonObject] = []
        for key, value in desired.items():
            current = values[key]
            if current == value and type(current) is type(value):
                continue
            if (
                record is not None
                and key in fields
                and SavedValue.capture(current) != fields[key].last
            ):
                raise IntegrationError(
                    f"The setting {key} was edited after FCC setup. No settings will change; correct or remove it manually.",
                    status_code=409,
                )
            changes.append(
                {
                    "setting": key,
                    "before": _display(key, current),
                    "after": _display(key, value),
                }
            )
            if action not in {Action.DISCONNECT, Action.REPAIR}:
                if key not in fields:
                    prior = (
                        None
                        if recognized and target.manual_owned(key, current, connection)
                        else SavedValue.capture(current)
                    )
                    fields[key] = FieldHistory(
                        prior=prior, last=SavedValue.capture(value)
                    )
                else:
                    fields[key].last = SavedValue.capture(value)
            target.write(document, key, value)
        if action == Action.DISCONNECT:
            if record is None and target.id == Item.CLAUDE_JETBRAINS:
                if document.get(AGENT_PATH) in ({}, {"env": {}}):
                    document.delete(AGENT_PATH)
                    notes.append("Remove the empty Claude Code (FCC) agent entry.")
                else:
                    notes.append(
                        "Keep remaining custom agent fields; finish any further cleanup manually."
                    )
            for path in reversed(containers):
                remaining = document.get(tuple(path))
                if remaining == {} or remaining == []:
                    document.delete(tuple(path))
        rendered = document.render()
        output = rendered if rendered != original else None
        if output is not None:
            capture.source.require_writable()
        following = (
            None
            if action in {Action.DISCONNECT, Action.REPAIR}
            else Ownership(
                item=target.id,
                path=str(target.path),
                fields=fields,
                created_containers=containers,
            )
        )
        record_changed = following != record
        if target.id == Item.CODEX:
            notes.append(
                "The App, VS Code extension, and normal Codex CLI share this file and provider selection."
            )
        unknown_history = (recognized and record is None) or any(
            field.prior is None for field in fields.values()
        )
        if action == Action.REPAIR:
            summary = "Apply the Claude Code first-run fix for this user. This one-time repair has no Disconnect action."
        elif action == Action.DISCONNECT:
            summary = (
                "Remove the recognized FCC settings. Previous settings are unknown."
                if unknown_history
                else "Restore the settings saved before FCC setup."
            )
        else:
            summary = (
                "You can Disconnect later to remove FCC settings. Previous settings are unknown."
                if unknown_history
                else "You can Disconnect later to restore the settings saved before setup."
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
            "action": action,
            "title": target.title,
            "path": str(target.path),
            "summary": summary,
            "writes_file": output is not None,
            "changes": changes,
            "notes": notes,
            "instructions": target.instructions,
            "revision": self._revision(capture, action, connection),
            "endpoint": connection.url if target.id != Item.CLAUDE_LOGIN else None,
            "helper_path": str(capture.installed.fcc_command)
            if target.id == Item.CODEX and capture.installed.fcc_command
            else None,
        }
        return Prepared(capture, output, following, record_changed, preview)

    def _one(self, item: Item, action: Action, connection: Connection) -> Prepared:
        installed = (
            InstalledClients(
                claude_vscode=False,
                codex_vscode=False,
                codex_app=False,
                jetbrains=False,
                claude_command=None,
                fcc_command=None,
                acp=AcpInstallation(),
                scope_issue="",
            )
            if action == Action.RECOVER
            else self.installations.scan()
        )
        target = next(
            target
            for target in make_targets(self.installations, installed, connection)
            if target.id == item
        )
        return self._prepare(
            self._capture(target, installed, connection), action, connection
        )

    def preview(self, item: Item, action: Action, connection: Connection) -> JsonObject:
        return self._one(item, action, connection).preview

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
                "actions": [],
                "status": "not_configured",
                "message": "",
                "manual": False,
            }
            try:
                capture = self._capture(target, installed, connection)
                if capture.recovery is not None:
                    entry["status"] = "needs_attention"
                    entry["message"] = (
                        "A previous change was interrupted. Recover its saved setup history before continuing."
                    )
                    entry["missing"] = []
                    entry["documentation_url"] = _RECOVERY_GUIDE
                    entry["actions"] = [Action.RECOVER.value]
                    items.append(entry)
                    continue
                if (
                    target.id == Item.CODEX
                    and installed.codex_vscode
                    and _codex_extension_in_wsl(capture)
                ):
                    entry["badges"] = [*target.badges, "VS Code uses WSL: manual setup"]
                values = target.values(target.document(capture.source.data))
                recognized = target.recognized(values, connection)
                entry["manual"] = (
                    recognized
                    and capture.record is None
                    and target.id != Item.CLAUDE_LOGIN
                )
                actions = []
                if target.id == Item.CLAUDE_LOGIN:
                    if recognized:
                        entry["status"], entry["message"] = (
                            "configured",
                            "Onboarding already completed. Remaining login failures require checking the connection.",
                        )
                    elif not target.missing:
                        actions.append(Action.REPAIR.value)
                elif capture.record is not None or recognized:
                    entry["status"] = "configured"
                    actions.append(Action.DISCONNECT.value)
                    try:
                        prepared = self._prepare(capture, Action.UPDATE, connection)
                        if prepared.output is not None or prepared.record_changed:
                            entry["status"] = "update_available"
                            actions.insert(0, Action.UPDATE.value)
                    except IntegrationError as error:
                        entry["status"], entry["message"] = (
                            "needs_attention",
                            str(error),
                        )
                elif not target.missing:
                    self._require_ready(capture, Action.SETUP, connection)
                    actions.append(Action.SETUP.value)
                entry["actions"] = actions
            except IntegrationError as error:
                entry["status"], entry["message"] = "needs_attention", str(error)
            if (
                entry["status"] == "needs_attention"
                and (self.state_dir / f"{target.id}.json").is_file()
            ):
                entry["documentation_url"] = _RECOVERY_GUIDE
            items.append(entry)
        return {
            "items": items,
            "scope": "Standard local installations for this OS and user account; VS Code default profile.",
        }

    def apply(
        self, item: Item, action: Action, revision: str, connection: Connection
    ) -> JsonObject:
        lock = InterprocessFileLock(self.state_dir / "write.lock")
        try:
            check_path(self.state_dir)
            self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not lock.acquire(wait=True, timeout=5):
                raise IntegrationError(
                    "Another integration change is in progress. Refresh and try again.",
                    status_code=409,
                )
            prepared = self._one(item, action, connection)
            current_revision = prepared.preview["revision"]
            assert isinstance(current_revision, str)
            if not hmac.compare_digest(revision, current_revision):
                raise IntegrationError(
                    "Settings changed since this preview. Refresh and confirm the new changes.",
                    status_code=409,
                )
            capture = prepared.capture
            for snapshot in (
                capture.source,
                capture.record_file,
                *capture.dependencies,
            ):
                snapshot.require_unchanged()
            output = prepared.output
            if output is None:
                if prepared.record_changed:
                    write_record(capture.record_file.path, prepared.following)
                return {
                    "applied": prepared.record_changed,
                    "instructions": prepared.preview["instructions"],
                    "message": "Recovery finished."
                    if action == Action.RECOVER
                    else "Disconnected. The client settings were already restored."
                    if action == Action.DISCONNECT and prepared.record_changed
                    else "FCC setup history saved."
                    if prepared.record_changed
                    else "No client settings changed.",
                }
            pending = PendingWrite(
                item=item,
                action=action,
                path=str(capture.target.path),
                before=capture.source.digest,
                after=content_hash(output),
                previous=capture.record,
                following=prepared.following,
            )
            write_record(capture.record_file.path, pending)
            capture.source.require_unchanged()
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
            write_record(capture.record_file.path, prepared.following)
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
