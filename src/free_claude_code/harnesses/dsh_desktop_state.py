"""Bounded ownership records for interrupted DSH Desktop operations."""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from ruamel.yaml.error import YAMLError

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import dsh_files
from free_claude_code.harnesses.config_file import decode_json
from free_claude_code.harnesses.dsh_files import DshConfigError, mapping


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Projection(_Record):
    routes: list[JsonObject]
    credential_hash: str | None
    defaults: list[JsonObject]
    effective_default: JsonObject | None

    @field_validator("credential_hash")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid credential fingerprint")
        return value


class Restoration(_Record):
    default_yaml: str | None
    follow_default: bool
    provider_row_created: bool
    provider_config_created: bool
    providers_created: bool
    default_row_created: bool

    @field_validator("default_yaml")
    @classmethod
    def valid_default(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                mapping(dsh_files.yaml_parser().load(value))
            except YAMLError, ValueError:
                raise ValueError("Invalid restoration value") from None
        return value


class Ownership(_Record):
    version: Literal[2] = 2
    home: str
    phase: Literal["connected", "pending_connect", "pending_disconnect", "retained"]
    before: Projection
    after: Projection | None
    restoration: Restoration | None
    retention_reason: str | None

    @model_validator(mode="after")
    def valid_phase(self) -> Self:
        pending = self.phase.startswith("pending_")
        if pending != (self.after is not None):
            raise ValueError("Invalid operation projections")
        if (self.phase == "retained") != (self.restoration is None):
            raise ValueError("Invalid restoration state")
        if self.phase == "retained" and (
            self.before.routes
            or self.before.defaults
            or not self.before.credential_hash
            or not self.retention_reason
        ):
            raise ValueError("Invalid retained credential")
        return self

    @property
    def projections(self) -> list[Projection]:
        return [self.before] if self.after is None else [self.before, self.after]

    @property
    def owned_defaults(self) -> list[JsonObject]:
        return [value for part in self.projections for value in part.defaults]

    def recognize(self, observed: Projection, *, removing: bool = False) -> None:
        routes = [value for part in self.projections for value in part.routes]
        hashes = [part.credential_hash for part in self.projections]
        if removing:
            hashes.append(None)
        if (
            any(value not in routes for value in observed.routes)
            or observed.credential_hash not in hashes
        ):
            raise DshConfigError(
                "The FCC route or credential was edited in DSH. Restore that FCC entry before retrying; other entries will be preserved."
            )
        if not removing and self.phase == "connected" and not observed.routes:
            raise DshConfigError(
                "The FCC route was removed in DSH. Disconnect its saved ownership before configuring again."
            )


def read(path: Path, home: Path) -> Ownership | None:
    dsh_files.regular_path(path)
    try:
        raw = decode_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError, UnicodeError:
        raise DshConfigError(
            "The FCC DSH ownership record is invalid. Restore the matching record before changing this integration."
        ) from None
    if isinstance(raw, dict) and raw.get("version") == 1:
        raise DshConfigError(
            "This FCC DSH ownership record uses the earlier development format. Clean up that development connection before configuring it with this version."
        )
    try:
        record = Ownership.model_validate(raw)
        if record.home != str(home):
            raise ValueError("Different DSH home")
    except ValidationError, ValueError:
        raise DshConfigError(
            "The FCC DSH ownership record is invalid or belongs to another DSH home. Restore the matching record before changing this integration."
        ) from None
    return record


def save(path: Path, record: Ownership) -> None:
    dsh_files.regular_path(path)
    dsh_files.write_text(
        path,
        json.dumps(record.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        private=True,
    )
