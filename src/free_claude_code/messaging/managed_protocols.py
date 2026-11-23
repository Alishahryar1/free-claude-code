"""Protocols for messaging-owned managed Claude sessions."""

from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from free_claude_code.core.json_types import JsonObject


@runtime_checkable
class ManagedClaudeSessionProtocol(Protocol):
    """Protocol for managed Claude sessions used by messaging."""

    def start_task(
        self, prompt: str, session_id: str | None = None, fork_session: bool = False
    ) -> AsyncGenerator[JsonObject]: ...

    @property
    def is_busy(self) -> bool: ...


@runtime_checkable
class ManagedClaudeSessionManagerProtocol(Protocol):
    """Protocol for the managed Claude session pool used by messaging."""

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> tuple[ManagedClaudeSessionProtocol, str, bool]: ...

    async def register_real_session_id(
        self, temp_id: str, real_session_id: str
    ) -> bool: ...

    async def stop_all(self) -> None: ...

    async def remove_session(self, session_id: str) -> bool: ...

    def get_stats(self) -> dict[str, int]: ...
