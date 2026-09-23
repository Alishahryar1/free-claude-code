"""Immutable Copilot metadata and the narrow runtime boundary."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint


class CopilotUnavailable(RuntimeError):
    """A safe, actionable message about Copilot runtime availability."""


class CopilotAuthenticationRequired(CopilotUnavailable):
    """The selected native profile needs interactive authentication."""


class CopilotEgress(StrEnum):
    MESSAGES = "messages"
    RESPONSES = "responses"
    CHAT = "chat"


@dataclass(frozen=True, slots=True)
class CopilotIdentity:
    host: str
    login: str

    @property
    def display(self) -> str:
        return (
            f"@{self.login}"
            if self.host == "github.com"
            else f"@{self.login} ({self.host})"
        )


@dataclass(frozen=True, slots=True)
class CopilotModel:
    info: ProviderModelInfo
    messages: MessagesModelCapabilities
    supported_efforts: tuple[str, ...] | None = None
    default_effort: str | None = None


@dataclass(frozen=True, slots=True)
class CopilotEndpoint:
    egress: CopilotEgress
    http: HttpEndpoint = field(repr=False)
    expires_at: float | None = None


class CopilotSession(Protocol):
    async def endpoint(self) -> CopilotEndpoint: ...
    async def close(self) -> None: ...


class CopilotRuntime(Protocol):
    async def start(self) -> None: ...
    async def identity(self) -> CopilotIdentity | None: ...
    async def models(self) -> tuple[CopilotModel, ...]: ...
    async def session(self, model_id: str) -> CopilotSession: ...
    async def close(self) -> None: ...
