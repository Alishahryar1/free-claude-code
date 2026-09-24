"""Google Antigravity connected-account provider support."""

from .auth import AntigravityAuthManager
from .credentials import (
    AntigravityCredentialError,
    AntigravityCredentials,
    load_native_credentials,
)

__all__ = [
    "AntigravityAuthManager",
    "AntigravityCredentialError",
    "AntigravityCredentials",
    "load_native_credentials",
]
