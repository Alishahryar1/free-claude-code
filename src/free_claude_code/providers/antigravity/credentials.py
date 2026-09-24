"""Read Google Antigravity credentials owned by the native ``agy`` CLI."""

import base64
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WINDOWS_CREDENTIAL_TARGET = "gemini:antigravity"
KEYRING_SERVICE = "gemini"
KEYRING_ACCOUNT = "antigravity"
GO_KEYRING_BASE64_PREFIX = "go-keyring-base64:"
TOKEN_PATH_ENV = "ANTIGRAVITY_OAUTH_TOKEN_PATH"


class AntigravityCredentialError(RuntimeError):
    """The native Antigravity credential could not be loaded safely."""

    status_code = 401


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
    paths.append(home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token")
    return tuple(dict.fromkeys(paths))


def load_native_credentials(
    *, token_paths: tuple[Path, ...] | None = None
) -> AntigravityCredentials:
    """Load credentials without copying or mutating the native ``agy`` session."""

    failures: list[str] = []
    paths = token_paths or native_token_paths()

    # An explicit path (argument or environment override) is authoritative. This
    # also makes tests/headless deployments independent of a desktop keyring.
    explicit_path = token_paths is not None or bool(os.getenv(TOKEN_PATH_ENV))
    if explicit_path:
        credential = _load_file_credentials(paths, failures)
        if credential is not None:
            return credential
        raise _missing_credentials(failures)

    try:
        payload, source = _read_platform_credential()
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError, ValueError, UnicodeError) as error:
        failures.append(f"OS keyring: {error}")
    else:
        try:
            return _parse_credential_document(payload, source=source)
        except AntigravityCredentialError as error:
            failures.append(f"{source}: {error}")

    credential = _load_file_credentials(paths, failures)
    if credential is not None:
        return credential
    raise _missing_credentials(failures)


def _load_file_credentials(
    paths: tuple[Path, ...], failures: list[str]
) -> AntigravityCredentials | None:
    for path in paths:
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
    return None


def _missing_credentials(failures: list[str]) -> AntigravityCredentialError:
    suffix = f" ({'; '.join(failures)})" if failures else ""
    return AntigravityCredentialError(
        "No usable Antigravity CLI session was found. Sign in with `agy` first."
        + suffix
    )


def _read_platform_credential() -> tuple[Any, str]:
    if sys.platform == "win32":
        return (
            _read_windows_credential(WINDOWS_CREDENTIAL_TARGET),
            f"windows:{WINDOWS_CREDENTIAL_TARGET}",
        )
    if sys.platform == "darwin":
        text = _run_credential_command(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYRING_SERVICE,
                "-a",
                KEYRING_ACCOUNT,
                "-w",
            ]
        )
        return _decode_json_document(text), "macos-keychain:gemini/antigravity"
    if sys.platform.startswith("linux"):
        text = _run_credential_command(
            [
                "secret-tool",
                "lookup",
                "service",
                KEYRING_SERVICE,
                "username",
                KEYRING_ACCOUNT,
            ]
        )
        return _decode_json_document(text), "linux-secret-service:gemini/antigravity"
    raise FileNotFoundError("no supported native keyring backend")


def _run_credential_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired as error:
        raise OSError(f"credential command timed out: {command[0]}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise FileNotFoundError(
            f"{command[0]} could not read Antigravity credential: {detail}"
        )
    if not result.stdout.strip():
        raise FileNotFoundError(f"{command[0]} returned an empty credential")
    return result.stdout.strip()


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
        if error == 1168:
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
