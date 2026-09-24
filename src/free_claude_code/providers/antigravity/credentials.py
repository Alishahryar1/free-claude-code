"""Read Google Antigravity credentials owned by the native ``agy`` CLI."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WINDOWS_CREDENTIAL_TARGET = "gemini:antigravity"
GO_KEYRING_BASE64_PREFIX = "go-keyring-base64:"
TOKEN_PATH_ENV = "ANTIGRAVITY_OAUTH_TOKEN_PATH"


class AntigravityCredentialError(RuntimeError):
    """The native Antigravity credential could not be loaded safely."""


@dataclass(frozen=True, slots=True, repr=False)
class AntigravityCredentials:
    """A credential snapshot borrowed from the native Antigravity CLI."""

    access_token: str
    refresh_token: str
    expires_at: float | None
    token_type: str = "Bearer"
    auth_method: str = "consumer"
    source: str = "native"

    def expires_soon(self, early_seconds: float = 300.0) -> bool:
        """Return whether the access token is expired or inside a refresh window."""

        return self.expires_at is not None and (
            time.time() + early_seconds >= self.expires_at
        )


def native_token_paths() -> tuple[Path, ...]:
    """Return file-backed credential locations used by Antigravity CLI variants."""

    override = os.getenv(TOKEN_PATH_ENV)
    home = Path.home()
    paths: list[Path] = []
    if override:
        paths.append(Path(override).expanduser())
    paths.extend(
        (
            home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token",
            home / ".gemini" / "oauth_creds.json",
        )
    )
    return tuple(dict.fromkeys(paths))


def load_native_credentials(
    *, token_paths: tuple[Path, ...] | None = None
) -> AntigravityCredentials:
    """Load credentials without copying or mutating the native ``agy`` session."""

    failures: list[str] = []
    if os.name == "nt":
        try:
            payload = _read_windows_credential(WINDOWS_CREDENTIAL_TARGET)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, UnicodeError) as error:
            failures.append(f"Windows Credential Manager: {error}")
        else:
            return _parse_credential_document(
                payload,
                source=f"windows:{WINDOWS_CREDENTIAL_TARGET}",
            )

    for path in token_paths or native_token_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as error:
            failures.append(f"{path}: {error}")
            continue
        try:
            payload = _decode_json_document(text)
            return _parse_credential_document(payload, source=str(path))
        except (ValueError, AntigravityCredentialError) as error:
            failures.append(f"{path}: {error}")

    suffix = f" ({'; '.join(failures)})" if failures else ""
    raise AntigravityCredentialError(
        "No usable Antigravity CLI session was found. Sign in with `agy` first."
        + suffix
    )


def _parse_credential_document(
    payload: Any, *, source: str = "native"
) -> AntigravityCredentials:
    if not isinstance(payload, dict):
        raise AntigravityCredentialError("credential document must be an object")

    token = payload.get("token", payload)
    if not isinstance(token, dict):
        raise AntigravityCredentialError("credential document is missing token data")

    access_token = token.get("access_token")
    refresh_token = token.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        raise AntigravityCredentialError("credential is missing access_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise AntigravityCredentialError("credential is missing refresh_token")

    token_type = token.get("token_type")
    auth_method = payload.get("auth_method")
    expiry = token.get("expiry")
    if expiry is None:
        expiry = token.get("expiry_date")
    if expiry is None:
        expiry = token.get("expires_at")

    return AntigravityCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=_parse_expiry(expiry),
        token_type=token_type if isinstance(token_type, str) and token_type else "Bearer",
        auth_method=(
            auth_method
            if isinstance(auth_method, str) and auth_method
            else "consumer"
        ),
        source=source,
    )


def _parse_expiry(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AntigravityCredentialError("credential expiry is invalid")
    if isinstance(value, int | float):
        number = float(value)
        return number / 1000.0 if number > 10_000_000_000 else number
    if not isinstance(value, str) or not value.strip():
        raise AntigravityCredentialError("credential expiry is invalid")

    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return number / 1000.0 if number > 10_000_000_000 else number

    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise AntigravityCredentialError("credential expiry is invalid") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _decode_json_document(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith(GO_KEYRING_BASE64_PREFIX):
        encoded = candidate.removeprefix(GO_KEYRING_BASE64_PREFIX)
        try:
            candidate = base64.b64decode(encoded).decode("utf-8")
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid go-keyring base64 payload") from error

    payload = json.loads(candidate)
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _read_windows_credential(target: str) -> Any:
    """Read one Windows generic credential blob via Credential Manager."""

    from ctypes import wintypes

    class Credential(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", wintypes.LPVOID),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    pointer = ctypes.POINTER(Credential)()
    cred_read = advapi32.CredReadW
    cred_read.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(Credential)),
    ]
    cred_read.restype = wintypes.BOOL
    cred_free = advapi32.CredFree
    cred_free.argtypes = [ctypes.c_void_p]
    cred_free.restype = None

    if not cred_read(target, 1, 0, ctypes.byref(pointer)):
        error = ctypes.get_last_error()
        if error == 1168:  # ERROR_NOT_FOUND
            raise FileNotFoundError(target)
        raise OSError(error, f"CredReadW failed for {target!r}")

    try:
        credential = pointer.contents
        blob = ctypes.string_at(
            credential.CredentialBlob, credential.CredentialBlobSize
        )
    finally:
        cred_free(pointer)

    return _decode_json_document(blob.decode("utf-8"))
