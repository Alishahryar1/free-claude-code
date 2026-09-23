"""Managed and legacy dotenv discovery helpers."""

import os
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream

FCC_ENV_FILE_ENV = "FCC_ENV_FILE"
FCC_CONFIG_SCHEMA_ENV = "FCC_CONFIG_SCHEMA"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
PROXY_AUTH_ENABLED_ENV = "PROXY_AUTH_ENABLED"


def legacy_explicit_env_path(
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the pre-schema explicit dotenv import path, when configured."""

    source = env if env is not None else os.environ
    value = source.get(FCC_ENV_FILE_ENV, "").strip()
    return Path(value).expanduser() if value else None


def verified_checkout_env_path() -> Path | None:
    """Return this package checkout's legacy dotenv, never the caller's CWD."""

    checkout = Path(__file__).resolve().parents[3]
    package = checkout / "src" / "free_claude_code"
    pyproject = checkout / "pyproject.toml"
    if package != Path(__file__).resolve().parents[1]:
        return None
    if not pyproject.is_file():
        return None
    try:
        project_text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    if 'name = "free-claude-code"' not in project_text:
        return None
    env_path = checkout / ".env"
    return env_path if env_path.is_file() else None


def dotenv_values_from_text(text: str) -> dict[str, str]:
    """Parse dotenv text while preserving explicit blank assignments."""

    malformed = next(
        (binding for binding in parse_stream(StringIO(text)) if binding.error),
        None,
    )
    if malformed is not None:
        raise ValueError(f"Invalid dotenv syntax on line {malformed.original.line}")
    parsed = dotenv_values(stream=StringIO(text), interpolate=False)
    return {key: "" if value is None else value for key, value in parsed.items()}


def dotenv_values_from_file(path: Path) -> dict[str, str]:
    """Parse a dotenv file, raising an actionable error when it cannot be read."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not read configuration file {path}: {exc}") from exc
    try:
        return dotenv_values_from_text(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse configuration file {path}: {exc}") from exc
