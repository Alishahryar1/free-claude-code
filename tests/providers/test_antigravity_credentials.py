"""Credential parsing for the native Antigravity CLI session."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from free_claude_code.providers.antigravity.credentials import (
    AntigravityCredentialError,
    _decode_json_document,
    _parse_credential_document,
    load_native_credentials,
)


def test_parses_nested_agy_token_and_rfc3339_expiry() -> None:
    payload = {
        "auth_method": "consumer",
        "token": {
            "access_token": "access",
            "refresh_token": "refresh",
            "token_type": "Bearer",
            "expiry": "2030-01-02T03:04:05Z",
        },
    }

    credentials = _parse_credential_document(payload, source="test")

    assert credentials.access_token == "access"
    assert credentials.refresh_token == "refresh"
    assert credentials.auth_method == "consumer"
    assert credentials.source == "test"
    assert credentials.expires_at == datetime(
        2030, 1, 2, 3, 4, 5, tzinfo=UTC
    ).timestamp()


def test_parses_flat_expiry_date_milliseconds() -> None:
    credentials = _parse_credential_document(
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expiry_date": 2_000_000_000_000,
        }
    )

    assert credentials.expires_at == 2_000_000_000.0


def test_decodes_go_keyring_base64_document() -> None:
    encoded = (
        "go-keyring-base64:"
        "eyJ0b2tlbiI6eyJhY2Nlc3NfdG9rZW4iOiJhIiwicmVmcmVzaF90b2tlbiI6InIifX0="
    )

    assert _decode_json_document(encoded) == {
        "token": {"access_token": "a", "refresh_token": "r"}
    }


def test_loads_file_backed_native_session(tmp_path: Path) -> None:
    path = tmp_path / "antigravity-oauth-token"
    path.write_text(
        json.dumps(
            {
                "token": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expiry": "2030-01-02T03:04:05Z",
                }
            }
        ),
        encoding="utf-8",
    )

    credentials = load_native_credentials(token_paths=(path,))

    assert credentials.access_token == "access"
    assert credentials.refresh_token == "refresh"
    assert credentials.source == str(path)


def test_missing_session_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(AntigravityCredentialError, match="Sign in with `agy` first"):
        load_native_credentials(token_paths=(tmp_path / "missing",))


def test_rejects_missing_refresh_token() -> None:
    with pytest.raises(AntigravityCredentialError, match="refresh_token"):
        _parse_credential_document({"access_token": "access"})
