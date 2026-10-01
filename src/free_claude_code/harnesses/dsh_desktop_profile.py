"""Interpret and edit the supported stock DSH Desktop documents."""

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.harnesses import dsh_files
from free_claude_code.harnesses.config_file import decode_json
from free_claude_code.harnesses.dsh_config import DSH_PROVIDER_ID
from free_claude_code.harnesses.dsh_desktop_state import Projection, Restoration
from free_claude_code.harnesses.dsh_files import DshConfigError, mapping

DSH_DESKTOP_API_KEY = "FCC_DSH_DESKTOP_API_KEY"
_ENTRIES = {
    "llm-pi-ai": "@deepseek-ai/dsh-llm-pi-ai",
    "agent-default-model": "@deepseek-ai/dsh-agent-default-model",
    "credentials": "@deepseek-ai/dsh-credentials-local",
}
_REFERENCE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STOCK_BUNDLES = ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"]


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
class Profile:
    home: Path
    rows: CommentedSeq
    credentials: CommentedMap
    strict: bool = True

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
        if self.strict and disabled not in (False, None):
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

    def set_config(
        self,
        identity: str,
        value: CommentedMap | JsonObject,
    ) -> None:
        rows = self.matches(identity)
        if not rows:
            row = CommentedMap(id=identity)
            self.rows.append(row)
        else:
            row = rows[-1]
        row["config"] = (
            value if isinstance(value, CommentedMap) else CommentedMap(value)
        )

    def save_profile(self) -> bool:
        return dsh_files.write_yaml(self.patch_path, self.rows)

    def save_credentials(self) -> bool:
        return dsh_files.write_yaml(
            self.home / ".credentials.yaml", self.credentials, private=True
        )

    def projection(self, owned_defaults: list[JsonObject]) -> Projection:
        routes: list[JsonObject] = []
        defaults: list[JsonObject] = []
        for row in self.matches("llm-pi-ai"):
            config = mapping(row["config"]) if "config" in row else CommentedMap()
            providers = (
                mapping(config["providers"])
                if "providers" in config
                else CommentedMap()
            )
            if DSH_PROVIDER_ID in providers:
                value = _json(mapping(providers[DSH_PROVIDER_ID]))
                assert isinstance(value, dict)
                if value not in routes:
                    routes.append(value)
        for row in self.matches("agent-default-model"):
            if "config" in row:
                value = _json(mapping(row["config"]))
                if value in owned_defaults and value not in defaults:
                    assert isinstance(value, dict)
                    defaults.append(value)
        current = _json(self.effective("agent-default-model"))
        assert current is None or isinstance(current, dict)
        return Projection(
            routes=routes,
            credential_hash=_hash(self.refs().get(DSH_DESKTOP_API_KEY)),
            defaults=defaults,
            effective_default=current,
        )

    def restoration(self) -> Restoration:
        providers = self.matches("llm-pi-ai")
        defaults = self.matches("agent-default-model")
        return Restoration(
            default_yaml=dsh_files.yaml_text(mapping(defaults[-1]["config"]))
            if defaults and "config" in defaults[-1]
            else None,
            follow_default=True,
            provider_row_created=not providers,
            provider_config_created=not providers or "config" not in providers[-1],
            providers_created="providers" not in self.config("llm-pi-ai"),
            default_row_created=not defaults,
        )

    def configure(
        self,
        provider: JsonObject,
        default: JsonObject | None,
        owned_defaults: list[JsonObject],
    ) -> None:
        for row in self.matches("llm-pi-ai"):
            if "config" in row:
                config = mapping(row["config"])
                if "providers" in config and DSH_PROVIDER_ID in mapping(
                    config["providers"]
                ):
                    config["providers"][DSH_PROVIDER_ID] = CommentedMap(
                        copy.deepcopy(provider)
                    )
        config = copy.deepcopy(self.config("llm-pi-ai"))
        providers = (
            mapping(config["providers"]) if "providers" in config else CommentedMap()
        )
        providers[DSH_PROVIDER_ID] = CommentedMap(copy.deepcopy(provider))
        config["providers"] = providers
        self.set_config("llm-pi-ai", config)
        if default is not None:
            for row in self.matches("agent-default-model"):
                if "config" in row and _json(row["config"]) in owned_defaults:
                    row["config"] = CommentedMap(copy.deepcopy(default))
            self.set_config("agent-default-model", copy.deepcopy(default))

    def disconnect(
        self, restoration: Restoration, owned_defaults: list[JsonObject]
    ) -> None:
        inherited = CommentedMap()
        for row in list(self.matches("llm-pi-ai")):
            if "config" not in row:
                continue
            config = mapping(row["config"])
            providers = (
                mapping(config["providers"])
                if "providers" in config
                else CommentedMap()
            )
            owned = DSH_PROVIDER_ID in providers
            if owned:
                providers.pop(DSH_PROVIDER_ID)
                if (
                    not providers
                    and restoration.providers_created
                    and not has_comments(providers)
                ):
                    config.pop("providers", None)
                if (
                    restoration.provider_config_created
                    and dsh_files.yaml_text(config) == dsh_files.yaml_text(inherited)
                    and not has_comments(row)
                ):
                    row.pop("config")
                    self.prune_row(row, restoration.provider_row_created)
                    continue
            inherited = config
        for row in list(self.matches("agent-default-model")):
            if "config" not in row or _json(row["config"]) not in owned_defaults:
                continue
            if restoration.default_yaml is None:
                row.pop("config")
                self.prune_row(row, restoration.default_row_created)
            else:
                row["config"] = mapping(
                    dsh_files.yaml_parser().load(restoration.default_yaml)
                )

    def prune_row(self, row: CommentedMap, created: bool) -> None:
        if created and not set(row) - {"id", "name"} and not has_comments(row):
            self.rows.remove(row)


def has_comments(value: object) -> bool:
    if not isinstance(value, (CommentedMap, CommentedSeq)):
        return False
    if value.ca.comment or value.ca.items or value.ca.end:
        return True
    return any(
        has_comments(child)
        for child in (value.values() if isinstance(value, CommentedMap) else value)
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
        if bundles != STOCK_BUNDLES:
            raise ValueError
    except KeyError, ValueError:
        raise DshConfigError(
            "DSH Desktop needs exactly its stock base and Web profile bundles in that order. Custom bundles are not supported by this integration."
        ) from None
    return True


def validate_paths(home: Path) -> None:
    for name in (
        "profiles/desktop/package.json",
        "profiles/desktop/cordis.yml",
        "profiles/desktop/cordis.patch.yml",
        "cordis.patch.yml",
        ".credentials.yaml",
    ):
        dsh_files.regular_path(home / name, root=home)


def load(home: Path, *, strict: bool = True) -> Profile:
    validate_paths(home)
    if strict and not _initialized(home):
        raise DshConfigError(
            "Install and open DeepSeek Harness Desktop once, then retry Configure."
        )
    rows = dsh_files.read_yaml(
        home / "profiles/desktop/cordis.patch.yml", sequence=True
    )
    assert isinstance(rows, CommentedSeq)
    if strict:
        if dsh_files.read_yaml(home / "profiles/desktop/cordis.yml", sequence=True):
            raise DshConfigError(
                "DSH Desktop needs its stock empty profile root. Custom root entries are not supported."
            )
        higher = dsh_files.read_yaml(home / "cordis.patch.yml", sequence=True)
        assert isinstance(higher, CommentedSeq)
        for row in _rows(higher):
            if row.get("id") not in _ENTRIES or not set(row) - {"id", "name"}:
                continue
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
    result = Profile(home, rows, credentials, strict)
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
    if strict and set(result.config("credentials")) - {"watch", "debounceMs"}:
        raise DshConfigError(
            "DSH Desktop uses a custom credential store. Use its default local store before configuring FCC."
        )
    return result


def reference_usage(rows: CommentedSeq, higher: CommentedSeq) -> str:
    """Classify literal stock credential fields; opaque consumers retain the key."""
    targets = {
        "llm-pi-ai": "@deepseek-ai/dsh-llm-pi-ai",
        "llm-deepseek": "@deepseek-ai/dsh-llm-deepseek-api-key",
        "web-search-deepseek": "@deepseek-ai/dsh-web-search-deepseek",
    }
    # Native onboarding writes these named UI rows alongside ordinary id-only edits.
    known_names = (
        targets
        | _ENTRIES
        | {
            "ui-chat": "@deepseek-ai/dsh-client-ui-chat",
            "ui-settings": "@deepseek-ai/dsh-client-ui-settings",
            "ui-settings-account": "@deepseek-ai/dsh-client-ui-settings-account",
        }
    )
    configs: dict[str, object] = {}
    unknown = False
    try:
        for row in [*rows, *higher]:
            if (
                not isinstance(row, CommentedMap)
                or "insert" in row
                or row.get("group")
                or row.get("name", "").startswith("file:")
            ):
                unknown = True
                continue
            identity = row.get("id")
            name = row.get("name")
            if name is not None and name != known_names.get(identity):
                unknown = True
                continue
            if identity not in targets:
                continue
            if "config" in row:
                configs[identity] = row["config"]
        for identity, value in configs.items():
            config = mapping(value)
            values = (
                mapping(config["providers"]).values()
                if identity == "llm-pi-ai" and "providers" in config
                else ([] if identity == "llm-pi-ai" else [config])
            )
            for value in values:
                entry = mapping(value)
                ref = entry.get("apiKeyEnv")
                if ref == DSH_DESKTOP_API_KEY:
                    return "used"
                if ref is not None and (
                    not isinstance(ref, str) or not _REFERENCE.fullmatch(ref)
                ):
                    unknown = True
    except ValueError, TypeError, AttributeError:
        return "unknown"
    return "unknown" if unknown else "unused"
