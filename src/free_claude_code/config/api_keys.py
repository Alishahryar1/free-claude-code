"""Ordered secret credentials and explicit credential selection."""

import json
from typing import Annotated

from pydantic import BeforeValidator, SecretStr


def normalize_api_keys(value: object) -> tuple[SecretStr, ...]:
    """Accept a single key or array without discarding invalid entries."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError("API keys must be a valid JSON array") from None
        else:
            value = [value]
    if not isinstance(value, list | tuple):
        raise ValueError("API keys must be a key or an array of keys")
    keys: list[SecretStr] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(value):
        text = entry.get_secret_value() if isinstance(entry, SecretStr) else entry
        if not isinstance(text, str):
            raise ValueError(f"Key {index + 1} must be a string")
        text = text.strip()
        if not text:
            raise ValueError(f"Key {index + 1} cannot be blank")
        if text in seen:
            raise ValueError(f"Key {index + 1} duplicates Key {seen[text] + 1}")
        seen[text] = index
        keys.append(SecretStr(text))
    return tuple(keys)


type ApiKeyList = Annotated[tuple[SecretStr, ...], BeforeValidator(normalize_api_keys)]


def first_api_key(keys: ApiKeyList) -> str | None:
    """Select the first credential for default provider and voice resources."""
    return keys[0].get_secret_value() if keys else None


def selected_api_key(keys: ApiKeyList, number: int) -> str | None:
    """Select the one-based numbered credential, without substituting another."""
    return keys[number - 1].get_secret_value() if 1 <= number <= len(keys) else None


def api_key_values(keys: ApiKeyList) -> list[str]:
    """Extract plaintext for explicit persistence and internal edit resolution."""
    return [key.get_secret_value() for key in keys]


def encode_api_keys(keys: ApiKeyList) -> str | None:
    """Serialize a configured collection, leaving empty settings sparse."""
    return (
        json.dumps(api_key_values(keys), ensure_ascii=False, separators=(",", ":"))
        if keys
        else None
    )
