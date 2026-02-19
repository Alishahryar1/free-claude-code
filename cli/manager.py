"""
CLI Session Manager for Multi-Instance Claude CLI Support

Manages a pool of CLISession instances, each handling one conversation.
This enables true parallel processing where multiple conversations run
simultaneously in separate CLI processes.
"""

import asyncio
import uuid

from loguru import logger

from .session import CLISession


class CLISessionManager:
    """
    Manages multiple CLISession instances for parallel conversation processing.

    Each new conversation gets its own CLISession with its own subprocess.
    Replies to existing conversations reuse the same CLISession instance.
    """

    def __init__(
        self,
        workspace_path: str,
        api_url: str,
        allowed_dirs: list[str] | None = None,
        plans_directory: str | None = None,
    ):
        """
        Initialize the session manager.

        Args:
            workspace_path: Working directory for CLI processes
            api_url: API URL for the proxy
            allowed_dirs: Directories the CLI is allowed to access
            plans_directory: Directory for Claude Code CLI plan files (passed via --settings)
        """
        self.workspace = workspace_path
        self.api_url = api_url
        self.allowed_dirs = allowed_dirs or []
        self.plans_directory = plans_directory

        self._sessions: dict[str, CLISession] = {}
        self._pending_sessions: dict[str, CLISession] = {}
        self._temp_to_real: dict[str, str] = {}
        self._lock = asyncio.Lock()

        logger.info("CLISessionManager initialized")

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> tuple[CLISession, str, bool]:
        """
        Get an existing session or create a new one.

        Returns:
            Tuple of (CLISession instance, session_id, is_new_session)
        """
        async with self._lock:
            if session_id:
                lookup_id = self._temp_to_real.get(session_id, session_id)

                if lookup_id in self._sessions:
                    return self._sessions[lookup_id], lookup_id, False
                if lookup_id in self._pending_sessions:
                    return self._pending_sessions[lookup_id], lookup_id, False

            temp_id = session_id if session_id else f"pending_{uuid.uuid4().hex[:8]}"

            new_session = CLISession(
                workspace_path=self.workspace,
                api_url=self.api_url,
                allowed_dirs=self.allowed_dirs,
                plans_directory=self.plans_directory,
            )
            self._pending_sessions[temp_id] = new_session
            logger.info(f"Created new session: {temp_id}")

            return new_session, temp_id, True

    async def register_real_session_id(
        self, temp_id: str, real_session_id: str
    ) -> bool:
        """Register the real session ID from CLI output."""
        async with self._lock:
            if temp_id not in self._pending_sessions:
                logger.warning(f"Temp session {temp_id} not found")
                return False

            session = self._pending_sessions.pop(temp_id)
            self._sessions[real_session_id] = session
            self._temp_to_real[temp_id] = real_session_id

            logger.info(f"Registered session: {temp_id} -> {real_session_id}")
            return True

    async def get_real_session_id(self, temp_id: str) -> str | None:
        """Get the real session ID for a temporary ID."""
        async with self._lock:
            return self._temp_to_real.get(temp_id)

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session from the manager."""
        async with self._lock:
            if session_id in self._pending_sessions:
                session = self._pending_sessions.pop(session_id)
                await session.stop()
                return True

            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                await session.stop()
                for temp, real in list(self._temp_to_real.items()):
                    if real == session_id:
                        del self._temp_to_real[temp]
                return True

            return False

    async def stop_all(self):
        """Stop all sessions."""
        async with self._lock:
            all_sessions = list(self._sessions.values()) + list(
                self._pending_sessions.values()
            )
            for session in all_sessions:
                try:
                    await session.stop()
                except Exception as e:
                    logger.error(f"Error stopping session: {e}")

            self._sessions.clear()
            self._pending_sessions.clear()
            self._temp_to_real.clear()
            logger.info("All sessions stopped")

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "active_sessions": len(self._sessions),
            "pending_sessions": len(self._pending_sessions),
            "busy_count": sum(1 for s in self._sessions.values() if s.is_busy),
        }
