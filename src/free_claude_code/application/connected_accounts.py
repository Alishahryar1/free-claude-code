"""Safe application boundary for provider connected accounts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from free_claude_code.core.json_types import JsonObject, JsonValue


class ConnectedAccountLoginMode(StrEnum):
    """Supported interactive authorization flows."""

    BROWSER = "browser"
    DEVICE = "device"


class ConnectedAccountState(StrEnum):
    """Customer-visible connected-account lifecycle states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ConnectedAccountStatus:
    """Credential-free state safe to return from the local Admin API."""

    provider_id: str
    state: ConnectedAccountState
    connected: bool
    revision: int
    attempt_id: str | None = None
    email: str | None = None
    mode: ConnectedAccountLoginMode | None = None
    authorization_url: str | None = None
    verification_url: str | None = None
    user_code: str | None = None
    expires_at: int | None = None
    model_count: int | None = None
    message: str | None = None
    display_identity: str | None = None
    supported_login_modes: tuple[ConnectedAccountLoginMode, ...] = (
        ConnectedAccountLoginMode.BROWSER,
        ConnectedAccountLoginMode.DEVICE,
    )
    default_login_mode: ConnectedAccountLoginMode = ConnectedAccountLoginMode.BROWSER

    def as_dict(self) -> JsonObject:
        """Serialize only the explicitly safe status contract."""

        payload: JsonObject = {
            "provider_id": self.provider_id,
            "state": self.state,
            "connected": self.connected,
            "revision": self.revision,
            "supported_login_modes": [
                mode.value for mode in self.supported_login_modes
            ],
            "default_login_mode": self.default_login_mode.value,
        }
        optional_fields: tuple[tuple[str, JsonValue], ...] = (
            ("attempt_id", self.attempt_id),
            ("email", self.email),
            ("display_identity", self.display_identity),
            ("mode", self.mode),
            ("authorization_url", self.authorization_url),
            ("verification_url", self.verification_url),
            ("user_code", self.user_code),
            ("expires_at", self.expires_at),
            ("model_count", self.model_count),
            ("message", self.message),
        )
        for key, value in optional_fields:
            if value is not None:
                payload[key] = value
        return payload


class ConnectedAccountPort(Protocol):
    """One process-lifetime provider authentication owner."""

    def is_connected(self) -> bool: ...

    def status(self) -> ConnectedAccountStatus: ...

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus: ...

    async def cancel_login(self) -> ConnectedAccountStatus: ...

    async def disconnect(self) -> ConnectedAccountStatus: ...

    async def close(self) -> None: ...
