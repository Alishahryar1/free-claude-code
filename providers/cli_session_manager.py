"""
CLI Session Manager for Multi-Instance Claude CLI Support

Manages a pool of CLISession instances, each handling one conversation.
This enables true parallel processing where multiple conversations run
simultaneously in separate CLI processes.
"""

import asyncio
import uuid
import logging
from typing import Dict, Optional, Tuple, List
from .claude_cli import CLISession

logger = logging.getLogger(__name__)


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
        allowed_dirs: Optional[List[str]] = None,
        max_sessions: int = 10,
    ):
        """
        Initialize the session manager.
        
        Args:
            workspace_path: Working directory for CLI processes
            api_url: API URL for the proxy
            allowed_dirs: Directories the CLI is allowed to access
            max_sessions: Maximum concurrent sessions (prevents resource exhaustion)
        """
        self.workspace = workspace_path
        self.api_url = api_url
        self.allowed_dirs = allowed_dirs or []
        self.max_sessions = max_sessions
        
        # Active sessions: real_session_id -> CLISession
        self._sessions: Dict[str, CLISession] = {}
        
        # Pending sessions: temp_id -> CLISession (before we know real session ID)
        self._pending_sessions: Dict[str, CLISession] = {}
        
        # Mapping: temp_id -> real_session_id (for updating after CLI responds)
        self._temp_to_real: Dict[str, str] = {}
        
        # Lock for thread-safe session management
        self._lock = asyncio.Lock()
        
        logger.info(f"CLISessionManager initialized (max_sessions={max_sessions})")

    async def get_or_create_session(
        self, session_id: Optional[str] = None
    ) -> Tuple[CLISession, str, bool]:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional existing session ID to resume.
                       If None, creates a new session.
        
        Returns:
            Tuple of (CLISession instance, session_id, is_new_session)
            For new sessions, session_id is a temporary ID until CLI assigns real one.
        """
        async with self._lock:
            # Case 1: Resume existing session (active or pending)
            if session_id:
                # Resolve temp_id to real_id if needed
                lookup_id = self._temp_to_real.get(session_id, session_id)
                
                if lookup_id in self._sessions:
                    logger.debug(f"Reusing existing session: {lookup_id}")
                    return self._sessions[lookup_id], lookup_id, False
                if lookup_id in self._pending_sessions:
                    logger.debug(f"Reusing pending session: {lookup_id}")
                    return self._pending_sessions[lookup_id], lookup_id, False
            
            # Case 2: Check if we're at capacity
            total_sessions = len(self._sessions) + len(self._pending_sessions)
            if total_sessions >= self.max_sessions:
                # Find and clean up any idle sessions
                await self._cleanup_idle_sessions_unlocked()
                
                # Re-check after cleanup
                total_sessions = len(self._sessions) + len(self._pending_sessions)
                if total_sessions >= self.max_sessions:
                    logger.warning(f"Max sessions ({self.max_sessions}) reached")
                    raise RuntimeError(
                        f"Maximum concurrent sessions ({self.max_sessions}) reached. "
                        "Please wait for existing conversations to complete."
                    )
            
            # Case 3: Create new session
            # If session_id was provided (but not found), use it as temp_id
            # Otherwise generate a new one
            temp_id = session_id if session_id else f"pending_{uuid.uuid4().hex[:8]}"
            
            new_session = CLISession(
                workspace_path=self.workspace,
                api_url=self.api_url,
                allowed_dirs=self.allowed_dirs,
            )
            self._pending_sessions[temp_id] = new_session
            logger.info(f"Created new session with temp_id: {temp_id}")
            
            return new_session, temp_id, True

    async def register_real_session_id(self, temp_id: str, real_session_id: str) -> bool:
        """
        Called when we learn the real session ID from CLI output.
        Moves session from pending to active.
        
        Args:
            temp_id: The temporary ID we assigned
            real_session_id: The real session ID from Claude CLI
            
        Returns:
            True if registration succeeded, False otherwise
        """
        async with self._lock:
            if temp_id not in self._pending_sessions:
                logger.warning(f"Temp session {temp_id} not found for registration")
                return False
            
            session = self._pending_sessions.pop(temp_id)
            self._sessions[real_session_id] = session
            self._temp_to_real[temp_id] = real_session_id
            
            logger.info(f"Registered session: {temp_id} -> {real_session_id}")
            return True

    async def get_real_session_id(self, temp_id: str) -> Optional[str]:
        """Get the real session ID for a temporary ID."""
        async with self._lock:
            return self._temp_to_real.get(temp_id)

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the manager.
        
        Args:
            session_id: Session ID to remove (can be temp or real)
            
        Returns:
            True if session was removed, False if not found
        """
        async with self._lock:
            # Check pending sessions
            if session_id in self._pending_sessions:
                session = self._pending_sessions.pop(session_id)
                await session.stop()
                logger.info(f"Removed pending session: {session_id}")
                return True
            
            # Check active sessions
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                await session.stop()
                # Clean up temp mapping
                for temp, real in list(self._temp_to_real.items()):
                    if real == session_id:
                        del self._temp_to_real[temp]
                logger.info(f"Removed active session: {session_id}")
                return True
            
            return False

    async def _cleanup_idle_sessions_unlocked(self):
        """
        Clean up sessions that are no longer busy.
        Must be called while holding self._lock.
        """
        idle_sessions = []
        
        for session_id, session in self._sessions.items():
            if not session.is_busy:
                idle_sessions.append(session_id)
        
        for session_id in idle_sessions[:3]:  # Remove up to 3 idle sessions
            session = self._sessions.pop(session_id)
            await session.stop()
            logger.debug(f"Cleaned up idle session: {session_id}")

    async def stop_all(self):
        """Stop all active sessions. Called on shutdown."""
        async with self._lock:
            all_sessions = list(self._sessions.values()) + list(self._pending_sessions.values())
            for session in all_sessions:
                try:
                    await session.stop()
                except Exception as e:
                    logger.error(f"Error stopping session: {e}")
            
            self._sessions.clear()
            self._pending_sessions.clear()
            self._temp_to_real.clear()
            logger.info("All sessions stopped")

    def get_stats(self) -> Dict:
        """Get current session statistics."""
        return {
            "active_sessions": len(self._sessions),
            "pending_sessions": len(self._pending_sessions),
            "max_sessions": self.max_sessions,
            "busy_count": sum(1 for s in self._sessions.values() if s.is_busy),
        }
