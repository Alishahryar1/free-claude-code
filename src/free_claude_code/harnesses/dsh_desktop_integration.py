"""Persist only FCC's route and credential in the native DSH desktop profile."""

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.harnesses import dsh_files
from free_claude_code.harnesses.config_file import decode_json
from free_claude_code.harnesses.dsh_config import DSH_PROVIDER_ID, build_dsh_provider
from free_claude_code.harnesses.dsh_files import DshConfigError, mapping

DSH_DESKTOP_API_KEY = "FCC_DSH_DESKTOP_API_KEY"
_ENTRIES = {
    "llm-pi-ai": "@deepseek-ai/dsh-llm-pi-ai",
    "agent-default-model": "@deepseek-ai/dsh-agent-default-model",
    "credentials": "@deepseek-ai/dsh-credentials-local",
}
_REFERENCE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def config_home() -> Path:
    value = os.environ.get("DSH_HOME", "").strip()
    if value == "~":
        return Path.home()
    if value.startswith(("~/", "~\\")):
        return (Path.home() / value[2:]).resolve()
    return Path(value).resolve() if value else Path.home() / ".dsh"


def _json(value: object) -> JsonValue:
    try:
        return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))
    except TypeError, ValueError, RecursionError:
        raise DshConfigError(
            "DSH has dynamic or invalid values in an FCC-owned field. Finish that edit before configuring FCC."
        ) from None


def _hash(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DshConfigError("DSH credentials must contain nonempty string references.")
    return hashlib.sha256(value.encode()).hexdigest()


def _rows(document: CommentedSeq) -> Iterator[CommentedMap]:
    for row in document:
        if not isinstance(row, CommentedMap):
            raise DshConfigError("DSH profile patches must be a list of mappings.")
        identity = row.get("id")
        if identity is not None and not isinstance(identity, str):
            raise DshConfigError(
                "DSH profile patch identities must be literal strings."
            )
        if identity in _ENTRIES and row.get("name") not in (None, _ENTRIES[identity]):
            raise DshConfigError(
                "A DSH integration plugin was replaced. Restore its stock plugin before configuring FCC."
            )
        inserted = row.get("insert")
        if inserted is not None:

            def check(entries: object) -> None:
                if not isinstance(entries, list):
                    raise DshConfigError("Invalid DSH inserted entries.")
                for entry in entries:
                    if not isinstance(entry, dict):
                        raise DshConfigError("Invalid DSH inserted entry.")
                    if (
                        entry.get("id") in tuple(_ENTRIES)
                        or entry.get("name") in _ENTRIES.values()
                    ):
                        raise DshConfigError(
                            "DSH has an inserted duplicate integration plugin. Keep the stock plugin before configuring FCC."
                        )
                    if entry.get("group") and isinstance(entry.get("config"), list):
                        check(entry["config"])

            check(inserted)
            continue
        yield row


def _plain_tree(value: object) -> None:
    """Avoid changing shared anchors in a mapping FCC must copy or edit."""
    if isinstance(value, (CommentedMap, CommentedSeq)):
        if value.anchor.value:
            raise DshConfigError(
                "DSH integration fields share a YAML anchor. Separate those fields before configuring FCC."
            )
        children = value.values() if isinstance(value, CommentedMap) else value
        for child in children:
            _plain_tree(child)


@dataclass
class _Profile:
    home: Path
    rows: CommentedSeq
    credentials: CommentedMap

    @property
    def patch_path(self) -> Path:
        return self.home / "profiles/desktop/cordis.patch.yml"

    def matches(self, identity: str) -> list[CommentedMap]:
        return [
            row
            for row in _rows(self.rows)
            if row.get("id") == identity
            and row.get("name") in (None, _ENTRIES[identity])
        ]

    def effective(self, identity: str) -> CommentedMap | None:
        result = None
        disabled: object = False
        for row in self.matches(identity):
            if set(row) - {"id", "name", "config", "disabled"}:
                raise DshConfigError(
                    "DSH integration plugins have custom composition overrides. Restore their stock configuration first."
                )
            disabled = row.get("disabled", disabled)
            if "config" in row:
                result = mapping(row["config"])
                _plain_tree(result)
        if disabled not in (False, None):
            raise DshConfigError(
                "A DSH integration plugin is disabled or dynamic. Enable its stock configuration first."
            )
        return result

    def config(self, identity: str) -> CommentedMap:
        return self.effective(identity) or CommentedMap()

    def providers(self) -> CommentedMap:
        config = self.config("llm-pi-ai")
        return mapping(config["providers"]) if "providers" in config else CommentedMap()

    def refs(self) -> CommentedMap:
        refs = self.credentials.get("refs")
        return mapping(refs) if refs is not None else CommentedMap()

    def inherited_config(self, identity: str) -> CommentedMap:
        inherited = CommentedMap()
        for row in self.matches(identity)[:-1]:
            if "config" in row:
                inherited = mapping(row["config"])
        return inherited

    def set_config(
        self,
        identity: str,
        value: CommentedMap | JsonObject | None,
        *,
        created: bool = False,
    ) -> None:
        rows = self.matches(identity)
        if not rows:
            if value is None:
                return
            row = CommentedMap(id=identity)
            self.rows.append(row)
        else:
            row = rows[-1]
        if value is not None:
            row["config"] = value
        else:
            row.pop("config", None)
            if created and not set(row) - {"id", "name"}:
                self.rows.remove(row)

    def save_profile(self) -> bool:
        return dsh_files.write_yaml(self.patch_path, self.rows)

    def save_credentials(self) -> bool:
        return dsh_files.write_yaml(
            self.home / ".credentials.yaml", self.credentials, private=True
        )


def _initialized(home: Path) -> bool:
    try:
        manifest = decode_json(
            (home / "profiles/desktop/package.json").read_text(encoding="utf-8-sig")
        )
    except FileNotFoundError:
        return False
    try:
        if not isinstance(manifest, dict):
            raise ValueError
        dsh = manifest["dsh"]
        if not isinstance(dsh, dict) or not isinstance(
            profile := dsh.get("profile"), dict
        ):
            raise ValueError
        bundles = profile.get("bundles")
        if not isinstance(bundles, list) or not all(
            name in bundles
            for name in ("@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app")
        ):
            raise ValueError
    except KeyError, ValueError:
        raise DshConfigError(
            "DSH Desktop needs its stock base and Web profile bundles. Open Desktop to finish setup."
        ) from None
    return True


def _load(home: Path) -> _Profile:
    if not _initialized(home):
        raise DshConfigError(
            "Install and open DeepSeek Harness Desktop once, then retry Configure."
        )
    rows = dsh_files.read_yaml(
        home / "profiles/desktop/cordis.patch.yml", sequence=True
    )
    assert isinstance(rows, CommentedSeq)
    higher = dsh_files.read_yaml(home / "cordis.patch.yml", sequence=True)
    assert isinstance(higher, CommentedSeq)
    for row in _rows(higher):
        if row.get("id") in _ENTRIES and set(row) - {"id", "name"}:
            raise DshConfigError(
                "The DSH home patch overrides a Desktop integration field. Move that override to its intended profile first."
            )
    credentials_path = home / ".credentials.yaml"
    try:
        mode = credentials_path.stat().st_mode
    except FileNotFoundError:
        pass
    else:
        if os.name != "nt" and stat.S_IMODE(mode) & 0o077:
            raise DshConfigError(
                "DSH credentials require owner-only permissions. Set .credentials.yaml to mode 600 and retry."
            )
    credentials = dsh_files.read_yaml(credentials_path)
    assert isinstance(credentials, CommentedMap)
    mapping(credentials)
    if credentials and (
        type(credentials.get("version")) is not int
        or credentials.get("version") != 1
        or set(credentials) - {"version", "refs", "records"}
    ):
        raise DshConfigError(
            "DSH credentials need the native version-1 document. Open Desktop to finish its migration first."
        )
    result = _Profile(home, rows, credentials)
    for key, value in result.refs().items():
        if not isinstance(key, str) or not _REFERENCE.fullmatch(key):
            raise DshConfigError("DSH credentials contain an invalid reference name.")
        _hash(value)
    records = credentials.get("records")
    if records is not None:
        for key, record in mapping(records).items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[a-z][a-z0-9-]*/[a-z][a-z0-9-]*", key)
                or not isinstance(record, dict)
            ):
                raise DshConfigError(
                    "DSH credentials contain an invalid account record."
                )
            if record.get("kind") == "grant" and set(record) == {"kind", "payload"}:
                _json(record["payload"])
            elif record.get("kind") == "api-key" and not set(record) - {
                "kind",
                "key",
                "env",
            }:
                if "key" in record:
                    _hash(record["key"])
                if "env" in record:
                    for name, value in mapping(record["env"]).items():
                        if not isinstance(name, str) or not _REFERENCE.fullmatch(name):
                            raise DshConfigError(
                                "DSH credentials contain an invalid account environment."
                            )
                        _hash(value)
            else:
                raise DshConfigError(
                    "DSH credentials contain an unsupported account record."
                )
    for identity in _ENTRIES:
        result.effective(identity)
    if set(result.config("credentials")) - {"watch", "debounceMs"}:
        raise DshConfigError(
            "DSH Desktop uses a custom credential store. Use its default local store before configuring FCC."
        )
    return result


def _state(path: Path, home: Path) -> JsonObject | None:
    try:
        record = decode_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    required = {
        "version",
        "home",
        "intent",
        "route",
        "credential_hash",
        "manage_default",
        "default",
        "baseline_default",
        "shape",
    }
    shape_keys = {
        "provider_row_created",
        "provider_config_created",
        "providers_created",
        "default_row_created",
    }
    pending_keys = {"next_route", "next_credential_hash", "next_default"}
    if (
        not isinstance(record, dict)
        or not required <= record.keys()
        or type(record.get("version")) is not int
        or record.get("version") != 1
        or record.get("home") != str(home)
        or record.get("intent") not in ("connect", "connected", "disconnect")
        or not isinstance(shape := record.get("shape"), dict)
        or not shape_keys <= shape.keys()
        or any(type(shape[key]) is not bool for key in shape_keys)
        or type(record.get("manage_default")) is not bool
        or not isinstance(record.get("baseline_default"), (str, type(None)))
        or any(
            not isinstance(record.get(key), (dict, type(None)))
            for key in ("route", "default", "next_route", "next_default")
        )
        or any(
            value is not None
            and (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            )
            for key in ("credential_hash", "next_credential_hash")
            for value in [record.get(key)]
        )
        or (bool(pending_keys & record.keys()) and not pending_keys <= record.keys())
    ):
        raise DshConfigError(
            "The FCC DSH ownership record is invalid or belongs to another DSH home. Restore the matching record before changing this integration."
        )
    saved = record["baseline_default"]
    if isinstance(saved, str):
        try:
            _json(mapping(dsh_files.yaml_parser().load(saved)))
        except YAMLError, ValueError:
            raise DshConfigError(
                "The FCC DSH ownership record has an invalid default restoration value."
            ) from None
    return record


def _save_state(path: Path, record: JsonObject) -> None:
    dsh_files.write_text(
        path, json.dumps(record, indent=2, allow_nan=False) + "\n", private=True
    )


@contextmanager
def _locked(home: Path) -> Iterator[_Profile]:
    if not _initialized(home):
        raise DshConfigError(
            "Install and open DeepSeek Harness Desktop once, then retry Configure."
        )
    with (
        dsh_files.file_lock(home / "profiles/desktop/package.json.lock", wait=2),
        dsh_files.file_lock(home / ".credentials.yaml.lock", wait=30),
    ):
        yield _load(home)


def _check_owned(
    profile: _Profile, record: JsonObject, *, removing: bool = False
) -> None:
    route = _json(profile.providers().get(DSH_PROVIDER_ID))
    credential = _hash(profile.refs().get(DSH_DESKTOP_API_KEY))
    routes = [record["route"]]
    hashes = [record["credential_hash"]]
    if "next_route" in record:
        routes.append(record["next_route"])
        hashes.append(record["next_credential_hash"])
    if removing:
        routes.append(None)
        hashes.append(None)
    if route not in routes or credential not in hashes:
        raise DshConfigError(
            "The FCC route or credential was edited in DSH. Restore that FCC entry before retrying; other entries will be preserved."
        )


def _new_state(profile: _Profile) -> JsonObject:
    provider_rows = profile.matches("llm-pi-ai")
    default_rows = profile.matches("agent-default-model")
    config = profile.config("llm-pi-ai")
    return {
        "version": 1,
        "home": str(profile.home),
        "intent": "connect",
        "route": None,
        "credential_hash": None,
        "manage_default": True,
        "default": _json(profile.effective("agent-default-model")),
        "baseline_default": dsh_files.yaml_text(mapping(default_rows[-1]["config"]))
        if default_rows and "config" in default_rows[-1]
        else None,
        "shape": {
            "provider_row_created": not provider_rows,
            "provider_config_created": not provider_rows
            or "config" not in provider_rows[-1],
            "providers_created": "providers" not in config,
            "default_row_created": not default_rows,
        },
    }


def _owns_default(profile: _Profile, record: JsonObject) -> bool:
    current = _json(profile.effective("agent-default-model"))
    return current == record["default"] or (
        "next_default" in record and current == record["next_default"]
    )


def configure(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    state_path: Path,
    provider_progress_timeout: float,
    only_existing: bool = False,
) -> JsonObject:
    home = home.resolve()
    if not auth_token.strip():
        raise DshConfigError(
            "Configure an FCC proxy authentication token before connecting DSH Desktop."
        )
    override = os.environ.get(DSH_DESKTOP_API_KEY)
    if override is not None and override != auth_token:
        raise DshConfigError(
            "FCC_DSH_DESKTOP_API_KEY in the environment overrides the saved Desktop credential. Remove or update it first."
        )
    provider = build_dsh_provider(
        catalog.models,
        proxy_root_url=proxy_root_url,
        credential_ref=DSH_DESKTOP_API_KEY,
        provider_progress_timeout=provider_progress_timeout,
    )
    if catalog.default_model_id not in {model.wire_slug for model in catalog.models}:
        raise DshConfigError("FCC's default model is missing from its DSH catalog.")
    changed = False
    with _locked(home) as profile:
        record = _state(state_path, home)
        if record is None:
            if only_existing:
                return {"changed": False, "connected": False}
            if (
                DSH_PROVIDER_ID in profile.providers()
                or DSH_DESKTOP_API_KEY in profile.refs()
            ):
                raise DshConfigError(
                    "A reserved FCC provider or credential already exists without an ownership record. Rename that entry before configuring FCC."
                )
            record = _new_state(profile)
        elif record["intent"] == "disconnect":
            raise DshConfigError(
                "Finish disconnecting DSH Desktop before connecting again."
            )
        _check_owned(profile, record)
        if not _owns_default(profile, record):
            record["manage_default"] = False
        desired_default: JsonObject | None = (
            {"provider": DSH_PROVIDER_ID, "model": catalog.default_model_id}
            if record["manage_default"]
            else None
        )
        record.update(
            intent="connect",
            next_route=provider,
            next_credential_hash=_hash(auth_token),
            next_default=desired_default,
        )
        _save_state(state_path, record)
        refs = profile.refs()
        refs[DSH_DESKTOP_API_KEY] = auth_token
        profile.credentials["version"] = 1
        profile.credentials["refs"] = refs
        changed |= profile.save_credentials()
        config = copy.deepcopy(profile.config("llm-pi-ai"))
        providers = (
            mapping(config["providers"]) if "providers" in config else CommentedMap()
        )
        providers[DSH_PROVIDER_ID] = provider
        config["providers"] = providers
        profile.set_config("llm-pi-ai", config)
        if desired_default is not None:
            profile.set_config("agent-default-model", desired_default)
        changed |= profile.save_profile()
        record.update(
            intent="connected",
            route=record.pop("next_route"),
            credential_hash=record.pop("next_credential_hash"),
            default=record.pop("next_default"),
        )
        _save_state(state_path, record)
    return status(home, proxy_root_url, auth_token, state_path=state_path) | {
        "changed": changed
    }


def status(
    home: Path, proxy_root_url: str, auth_token: str, *, state_path: Path
) -> JsonObject:
    home = home.resolve()
    result: JsonObject = {
        "installed": _initialized(home),
        "connected": False,
        "disconnect_pending": False,
        "paths": {
            "desktop_profile": str(home / "profiles/desktop/cordis.patch.yml"),
            "credentials": str(home / ".credentials.yaml"),
        },
    }
    record = _state(state_path, home)
    result["configured"] = record is not None
    if record is not None:
        result["disconnect_pending"] = record["intent"] == "disconnect"
        result["pending"] = record["intent"] != "connected"
    if not result["installed"] or record is None or record["intent"] != "connected":
        return result
    profile = _load(home)
    _check_owned(profile, record)
    provider = profile.providers().get(DSH_PROVIDER_ID)
    if isinstance(provider, dict):
        expected = proxy_root_url.rstrip("/") + "/v1"
        result["connected"] = (
            provider.get("baseURL") == expected
            and _hash(auth_token) == record["credential_hash"]
        )
    return result


def disconnect(home: Path, *, state_path: Path) -> JsonObject:
    home = home.resolve()
    with _locked(home) as profile:
        record = _state(state_path, home)
        if record is None:
            return {
                "connected": False,
                "configured": False,
                "disconnect_pending": False,
            }
        _check_owned(profile, record, removing=True)
        record["intent"] = "disconnect"
        _save_state(state_path, record)
        shape = cast(JsonObject, record["shape"])
        config = copy.deepcopy(profile.config("llm-pi-ai"))
        providers = (
            mapping(config["providers"]) if "providers" in config else CommentedMap()
        )
        providers.pop(DSH_PROVIDER_ID, None)
        if not providers and shape["providers_created"]:
            config.pop("providers", None)
        else:
            config["providers"] = providers
        restore_inheritance = shape["provider_config_created"] and dsh_files.yaml_text(
            config
        ) == dsh_files.yaml_text(profile.inherited_config("llm-pi-ai"))
        profile.set_config(
            "llm-pi-ai",
            None if restore_inheritance else config,
            created=shape["provider_row_created"] is True,
        )
        if record["manage_default"] and _owns_default(profile, record):
            saved = record["baseline_default"]
            if saved is not None and not isinstance(saved, str):
                raise DshConfigError(
                    "The FCC DSH default restoration record is invalid."
                )
            baseline = (
                mapping(dsh_files.yaml_parser().load(saved))
                if saved is not None
                else None
            )
            profile.set_config(
                "agent-default-model",
                baseline,
                created=shape["default_row_created"] is True,
            )
        profile.save_profile()
        # Another profile may deliberately reference the same credential. Never
        # revoke it while that reference remains, even after our route is gone.
        for path in [
            home / "cordis.patch.yml",
            *home.glob("profiles/*/cordis.patch.yml"),
        ]:
            if path.exists() and DSH_DESKTOP_API_KEY in path.read_text(
                encoding="utf-8-sig"
            ):
                raise DshConfigError(
                    "Another DSH setting still references the FCC Desktop credential. Remove that reference, then retry Disconnect to finish cleanup."
                )
        refs = profile.refs()
        refs.pop(DSH_DESKTOP_API_KEY, None)
        if "refs" in profile.credentials:
            profile.credentials["refs"] = refs
            profile.save_credentials()
        state_path.unlink()
    return {"connected": False, "configured": False, "disconnect_pending": False}


def refresh_connected(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    state_path: Path,
    provider_progress_timeout: float,
) -> bool:
    record = _state(state_path, home.resolve())
    if record is None:
        return False
    if record["intent"] == "disconnect":
        disconnect(home, state_path=state_path)
        return True
    result = configure(
        home,
        proxy_root_url,
        auth_token,
        catalog,
        state_path=state_path,
        provider_progress_timeout=provider_progress_timeout,
        only_existing=True,
    )
    return result["changed"] is True
