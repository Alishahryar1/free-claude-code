"""Settings-backed Admin config validation."""

from collections.abc import Mapping

from pydantic import ValidationError

from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.settings import Settings


def validate_values(
    values: Mapping[str, str], process: Mapping[str, str]
) -> tuple[bool, list[str]]:
    """Validate proposed env values against the Settings model."""

    settings, errors = settings_from_values(values, process)
    return settings is not None, errors


def settings_from_values(
    values: Mapping[str, str],
    process: Mapping[str, str],
) -> tuple[Settings | None, list[str]]:
    """Build the prospective Settings snapshot without reading dotenv files."""

    try:
        return compose_settings_snapshot(values, process).settings, []
    except ValidationError as exc:
        return None, format_validation_errors(exc)


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Return user-readable validation errors from a Pydantic exception."""

    errors: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "Invalid value"))
        errors.append(f"{loc}: {message}" if loc else message)
    return errors
