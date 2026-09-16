"""Typed metadata owned by the Admin configuration boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from free_claude_code.config.settings import Settings

type FieldType = Literal[
    "text",
    "secret",
    "number",
    "boolean",
    "model",
    "optional_model",
    "model_list",
    "select",
    "textarea",
]


@dataclass(frozen=True, slots=True)
class ConfigOptionSpec:
    """A persisted option value and its user-facing label."""

    value: str
    label: str


@dataclass(frozen=True, slots=True)
class ConfigSectionSpec:
    """A group of config fields rendered together in the Admin UI."""

    section_id: str
    label: str
    description: str
    advanced: bool = False


@dataclass(frozen=True, slots=True)
class ConfigFieldSpec:
    """Typed metadata for one env-backed Admin setting."""

    key: str
    label: str
    section_id: str
    field_type: FieldType = "text"
    settings_attr: str | None = None
    provider_ids: tuple[str, ...] = ()
    options: tuple[str | ConfigOptionSpec, ...] = ()
    secret: bool = False
    advanced: bool = False
    restart_required: bool = False
    session_sensitive: bool = False
    description: str = ""

    def resolved_default(self) -> str | None:
        """Return the Settings-owned default or Admin-only fallback."""

        if self.settings_attr is None:
            return None
        value = Settings.model_fields[self.settings_attr].get_default(
            call_default_factory=True
        )
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    @property
    def nullable(self) -> bool:
        """Return whether Admin may explicitly remove this setting."""

        return self.settings_attr is None or self.resolved_default() is None
