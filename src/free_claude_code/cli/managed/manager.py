"""Managed Claude Code session pool for messaging."""

import asyncio
import uuid

from loguru import logger

from free_claude_code.harnesses.claude import CLAUDE_BINARY_NAME

from .session import ManagedClaudeSession


class ManagedClaudeSessionManager:
    """
    Manages multiple Claude Code sessions for parallel conversation processing.

    Each new conversation gets its own subprocess. Replies to existing
    conversations reuse the same session instance.
    """

    def __init__(
        self,
        workspace_path: str,
        proxy_root_url: str,
        allowed_dirs: list[str] | None = None,
        claude_bin: str = CLAUDE_BINARY_NAME,
        *,
        auth_token: str,
        log_raw_cli_diagnostics: bool = False,
        log_messaging_error_details: bool = False,
    ):
        """
        Initialize the session manager.

        Args:
            workspace_path: Working directory for CLI processes
            proxy_root_url: Root URL for the local proxy
            allowed_dirs: Directories the CLI is allowed to access
        """
        self.workspace = workspace_path
        self.proxy_root_url = proxy_root_url
        self.allowed_dirs = allowed_dirs or []
        self.claude_bin = claude_bin
        self.auth_token = auth_token
        self._log_raw_cli_diagnostics = log_raw_cli_diagnostics
        self._log_messaging_error_details = log_messaging_error_details

        self._sessions: dict[str, ManagedClaudeSession] = {}
        self._pending_sessions: dict[str, ManagedClaudeSession] = {}
        self._temp_to_real: dict[str, str] = {}
        self._real_to_temp: dict[str, str] = {}
        self._closing_sessions: set[ManagedClaudeSession] = set()
        self._lock = asyncio.Lock()

    def _session_for_id(self, session_id: str) -> ManagedClaudeSession | None:
        lookup_id = self._temp_to_real.get(session_id, session_id)
        session = self._sessions.get(lookup_id)
        if session is not None:
            return session
        return self._pending_sessions.get(lookup_id)

    def _forget_session(self, session: ManagedClaudeSession) -> None:
        pending_ids = [
            session_id
            for session_id, owned in self._pending_sessions.items()
            if owned is session
        ]
        real_ids = [
            session_id
            for session_id, owned in self._sessions.items()
            if owned is session
        ]
        for session_id in pending_ids:
            self._pending_sessions.pop(session_id, None)
        for real_id in real_ids:
            self._sessions.pop(real_id, None)
            temp_id = self._real_to_temp.pop(real_id, None)
            if temp_id is not None:
                self._temp_to_real.pop(temp_id, None)
        self._closing_sessions.discard(session)

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> tuple[ManagedClaudeSession, str, bool]:
        """
        Get an existing session or create a new one.

        Returns:
            Tuple of (session instance, session_id, is_new_session)
        """
        async with self._lock:
            if session_id:
                lookup_id = self._temp_to_real.get(session_id, session_id)

                if lookup_id in self._sessions:
                    session = self._sessions[lookup_id]
                    if session in self._closing_sessions:
                        raise RuntimeError("Managed Claude session is closing.")
                    return session, lookup_id, False
                if lookup_id in self._pending_sessions:
                    session = self._pending_sessions[lookup_id]
                    if session in self._closing_sessions:
                        raise RuntimeError("Managed Claude session is closing.")
                    return session, lookup_id, False

            temp_id = session_id if session_id else f"pending_{uuid.uuid4().hex[:8]}"

            new_session = ManagedClaudeSession(
                workspace_path=self.workspace,
                proxy_root_url=self.proxy_root_url,
                allowed_dirs=self.allowed_dirs,
                claude_bin=self.claude_bin,
                auth_token=self.auth_token,
                log_raw_cli_diagnostics=self._log_raw_cli_diagnostics,
            )
            self._pending_sessions[temp_id] = new_session

            return new_session, temp_id, True

    async def register_real_session_id(
        self, temp_id: str, real_session_id: str
    ) -> bool:
        """Register the real session ID from CLI output."""
        async with self._lock:
            session = self._pending_sessions.get(temp_id)
            if session is None:
                logger.warning(f"Temp session {temp_id} not found")
                return False
            if session in self._closing_sessions:
                logger.warning("Cannot register a closing managed Claude session")
                return False
            existing = self._session_for_id(real_session_id)
            if existing is not None and existing is not session:
                logger.warning(
                    "Cannot register managed Claude session: real ID is already owned"
                )
                return False

            self._pending_sessions.pop(temp_id)
            self._sessions[real_session_id] = session
            self._temp_to_real[temp_id] = real_session_id
            self._real_to_temp[real_session_id] = temp_id

            logger.info(f"Registered session: {temp_id} -> {real_session_id}")
            return True

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session from the manager."""
        async with self._lock:
            session = self._session_for_id(session_id)
            if session is None:
                return False
            self._closing_sessions.add(session)
            stopped = await session.stop()
            if not stopped:
                return False
            self._forget_session(session)
            return True

    async def stop_all(self) -> None:
        """Stop all sessions."""
        async with self._lock:
            all_sessions = list(
                dict.fromkeys(
                    [
                        *self._sessions.values(),
                        *self._pending_sessions.values(),
                        *self._closing_sessions,
                    ]
                )
            )
            self._closing_sessions.update(all_sessions)
            failures = 0
            for session in all_sessions:
                try:
                    stopped = await session.stop()
                except Exception as e:
                    stopped = False
                    if self._log_messaging_error_details:
                        logger.error(
                            "Error stopping session: {}: {}",
                            type(e).__name__,
                            e,
                        )
                    else:
                        logger.error(
                            "Error stopping session: exc_type={}",
                            type(e).__name__,
                        )
                if stopped:
                    self._forget_session(session)
                else:
                    failures += 1

            if failures:
                raise RuntimeError(
                    f"Managed Claude session shutdown failures: {failures}."
                )
            logger.info("All sessions stopped")

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "active_sessions": len(self._sessions),
            "pending_sessions": len(self._pending_sessions),
            "busy_count": sum(1 for s in self._sessions.values() if s.is_busy),
        }
