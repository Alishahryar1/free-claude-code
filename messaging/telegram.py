"""
Telegram Platform Adapter

Implements MessagingPlatform for Telegram using Telethon.
"""

import asyncio
import logging
import os
from typing import Callable, Awaitable, Optional, Any, Dict

from .base import MessagingPlatform
from .models import IncomingMessage

logger = logging.getLogger(__name__)

# Optional import - Telethon may not be installed
try:
    from telethon import TelegramClient, events, errors

    TELETHON_AVAILABLE = True
except ImportError:
    TelegramClient = None
    events = None
    errors = None
    TELETHON_AVAILABLE = False


class TelegramPlatform(MessagingPlatform):
    """
    Telegram messaging platform adapter.

    Uses Telethon for Telegram API access.
    Designed for personal use (sending messages to yourself).
    """

    name = "telegram"

    def __init__(
        self,
        api_id: Optional[str] = None,
        api_hash: Optional[str] = None,
        allowed_user_id: Optional[str] = None,
        session_path: str = "claude_bot.session",
    ):
        if not TELETHON_AVAILABLE:
            raise ImportError(
                "Telethon is required for Telegram support. Install with: pip install telethon"
            )

        self.api_id = api_id or os.getenv("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.getenv("TELEGRAM_API_HASH")
        self.allowed_user_id = allowed_user_id or os.getenv("ALLOWED_TELEGRAM_USER_ID")
        self.session_path = session_path

        self._client: Optional[TelegramClient] = None
        self._message_handler: Optional[
            Callable[[IncomingMessage], Awaitable[None]]
        ] = None
        self._connected = False
        # Cache entity objects to avoid flood wait errors
        self._entity_cache: Dict[str, Any] = {}

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.api_id or not self.api_hash:
            raise ValueError("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")

        self._client = TelegramClient(
            self.session_path,
            int(self.api_id),
            self.api_hash,
        )

        # Register event handler
        @self._client.on(events.NewMessage())
        async def on_new_message(event):
            await self._handle_event(event)

        await self._client.start()
        self._connected = True

        # Run in background
        asyncio.create_task(self._client.run_until_disconnected())

        # Send startup notification
        try:
            me_entity = await self._client.get_input_entity("me")
            await self._client.send_message(
                me_entity, "🚀 **Claude Code Proxy is online!**"
            )
        except Exception as e:
            logger.warning(f"Could not send startup message: {e}")

        logger.info("Telegram platform started")

    async def stop(self) -> None:
        """Disconnect from Telegram."""
        if self._client:
            await self._client.disconnect()
        self._connected = False
        logger.info("Telegram platform stopped")

    async def _get_entity(self, chat_id: str) -> Any:
        """Get entity object for a chat_id, using cache to avoid flood wait errors."""
        if chat_id in self._entity_cache:
            return self._entity_cache[chat_id]
        
        # Get entity and cache it
        entity = await self._client.get_input_entity(peer=int(chat_id))
        self._entity_cache[chat_id] = entity
        return entity

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> str:
        """Send a message to a chat."""
        if not self._client:
            raise RuntimeError("Telegram client not connected")

        try:
            entity = await self._get_entity(chat_id)
            msg = await self._client.send_message(
                entity,
                text,
                reply_to=int(reply_to) if reply_to else None,
                parse_mode=parse_mode,
            )
            return str(msg.id)
        except errors.FloodWaitError as e:
            logger.error(f"Telegram flood wait: {e.seconds}s")
            raise

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        """Edit an existing message."""
        if not self._client:
            raise RuntimeError("Telegram client not connected")

        try:
            entity = await self._get_entity(chat_id)
            await self._client.edit_message(
                entity,
                int(message_id),
                text,
                parse_mode=parse_mode,
            )
        except errors.FloodWaitError as e:
            logger.error(f"Telegram flood wait on edit: {e.seconds}s")
            raise
        except errors.MessageNotModifiedError:
            # Message content unchanged, ignore
            pass

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler callback."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Check if connected to Telegram."""
        return self._connected and self._client is not None

    async def _handle_event(self, event: Any) -> None:
        """Handle incoming Telegram event."""
        # Security check
        if self.allowed_user_id:
            if str(event.sender_id) != str(self.allowed_user_id).strip():
                logger.debug(
                    f"Ignored message from unauthorized user: {event.sender_id}"
                )
                return

        if not event.text:
            return

        if not self._message_handler:
            logger.warning("No message handler registered")
            return

        # Convert to platform-agnostic message
        incoming = IncomingMessage(
            text=event.text,
            chat_id=str(event.chat_id),
            user_id=str(event.sender_id),
            message_id=str(event.id),
            platform="telegram",
            reply_to_message_id=str(event.reply_to_msg_id)
            if event.reply_to_msg_id
            else None,
            raw_event=event,
        )

        try:
            await self._message_handler(incoming)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await self.send_message(
                    incoming.chat_id,
                    f"❌ **Error:** {str(e)[:200]}",
                    reply_to=incoming.message_id,
                )
            except Exception:
                pass
