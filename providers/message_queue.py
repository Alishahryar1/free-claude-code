"""
Message Queue Manager for Claude CLI Telegram Bot

Handles queuing of messages when Claude is busy processing a request.
Messages are processed one-by-one in order per session.
"""

import asyncio
import logging
from typing import Callable, Awaitable, Dict, Optional, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    """A message waiting to be processed."""
    prompt: str
    chat_id: int
    msg_id: int
    reply_msg_id: int  # The status message to update
    event: any  # Original Telegram event for context


class SessionQueue:
    """Queue for a single session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.queue: asyncio.Queue[QueuedMessage] = asyncio.Queue()
        self.is_processing = False
        self.current_task: Optional[asyncio.Task] = None


class MessageQueueManager:
    """
    Manages per-session message queues.
    
    When a session is busy, new messages are queued and processed
    one-by-one after the current request completes.
    """
    
    def __init__(self):
        self._queues: Dict[str, SessionQueue] = {}
        self._lock = asyncio.Lock()
    
    def _get_or_create_queue(self, session_id: str) -> SessionQueue:
        """Get existing queue or create new one for session."""
        if session_id not in self._queues:
            self._queues[session_id] = SessionQueue(session_id)
        return self._queues[session_id]
    
    def is_session_busy(self, session_id: str) -> bool:
        """Check if a session is currently processing a request."""
        if session_id not in self._queues:
            return False
        return self._queues[session_id].is_processing
    
    async def enqueue(
        self,
        session_id: str,
        message: QueuedMessage,
        processor: Callable[[str, QueuedMessage], Awaitable[None]],
    ) -> bool:
        """
        Add a message to the session's queue.
        
        If the session is not busy, processing starts immediately.
        If busy, the message is queued for later processing.
        
        Args:
            session_id: Claude session ID
            message: The queued message data
            processor: Async function to process the message
            
        Returns:
            True if message was queued (session busy), False if processed immediately
        """
        async with self._lock:
            sq = self._get_or_create_queue(session_id)
            
            if sq.is_processing:
                # Session is busy, queue the message
                await sq.queue.put(message)
                queue_size = sq.queue.qsize()
                logger.info(f"Queued message for session {session_id}, queue size: {queue_size}")
                return True
            else:
                # Session is free, start processing
                sq.is_processing = True
        
        # Process outside the lock
        await self._process_message(session_id, message, processor)
        return False
    
    async def _process_message(
        self,
        session_id: str,
        message: QueuedMessage,
        processor: Callable[[str, QueuedMessage], Awaitable[None]],
    ) -> None:
        """Process a single message and then check the queue."""
        try:
            await processor(session_id, message)
        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}")
        finally:
            # Check if there are more messages in the queue
            await self._process_next(session_id, processor)
    
    async def _process_next(
        self,
        session_id: str,
        processor: Callable[[str, QueuedMessage], Awaitable[None]],
    ) -> None:
        """Process the next message in queue, if any."""
        async with self._lock:
            if session_id not in self._queues:
                return
            
            sq = self._queues[session_id]
            
            if sq.queue.empty():
                # No more messages, mark session as free
                sq.is_processing = False
                logger.debug(f"Session {session_id} queue empty, marking as free")
                return
            
            # Get next message
            try:
                next_msg = sq.queue.get_nowait()
                logger.info(f"Processing next queued message for session {session_id}")
            except asyncio.QueueEmpty:
                sq.is_processing = False
                return
        
        # Process next message (outside lock)
        await self._process_message(session_id, next_msg, processor)
    
    def get_queue_size(self, session_id: str) -> int:
        """Get the number of messages waiting in a session's queue."""
        if session_id not in self._queues:
            return 0
        return self._queues[session_id].queue.qsize()
    
    def cancel_session(self, session_id: str) -> int:
        """
        Cancel all queued messages for a session.
        
        Returns:
            Number of messages that were cancelled
        """
        if session_id not in self._queues:
            return 0
        
        sq = self._queues[session_id]
        cancelled = 0
        
        while not sq.queue.empty():
            try:
                sq.queue.get_nowait()
                cancelled += 1
            except asyncio.QueueEmpty:
                break
        
        sq.is_processing = False
        logger.info(f"Cancelled {cancelled} queued messages for session {session_id}")
        return cancelled
