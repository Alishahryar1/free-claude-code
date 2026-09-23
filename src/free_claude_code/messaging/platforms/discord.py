"""Discord messaging runtime."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from free_claude_code.core.diagnostics import format_user_error_preview

from ..limiter import MessagingRateLimiter
from ..models import IncomingMessage, MessageScope
from ..rendering.discord_markdown import format_status_discord
from ..voice import Transcriber, VoiceCancellationResult
from .discord_inbound import (
    discord_text_message_from_event,
    discord_voice_request_from_event,
    get_audio_attachment,
    parse_allowed_channels,
)
from .discord_io import DiscordMessenger
from .ports import InboundMessageHandler
from .voice_flow import VoiceNoteFlow

_discord_module: Any = None
try:
    import discord as _discord_import

    _discord_module = _discord_import
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False


def _get_discord() -> Any:
    """Return the discord module or raise a setup error."""
    if not DISCORD_AVAILABLE or _discord_module is None:
        raise ImportError(
            "discord.py is required. Install with: pip install discord.py"
        )
    return _discord_module


if DISCORD_AVAILABLE and _discord_module is not None:
    _discord = _discord_module

    class _DiscordClient(_discord.Client):
        """Internal Discord client that forwards events to the runtime."""

        def __init__(
            self,
            runtime: DiscordRuntime,
            intents: _discord.Intents,
        ) -> None:
            super().__init__(intents=intents)
            self._runtime = runtime

        async def on_ready(self) -> None:
            self._runtime._mark_connected()

        async def on_message(self, message: Any) -> None:
            await self._runtime._handle_client_message(message)
else:
    _DiscordClient = None


class DiscordRuntime:
    """Owns Discord SDK lifecycle and inbound event handoff."""

    name = "discord"

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_channel_ids: str | None = None,
        *,
        limiter: MessagingRateLimiter,
        transcriber: Transcriber | None,
        log_raw_messaging_content: bool = False,
        log_api_error_tracebacks: bool = False,
    ) -> None:
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py is required. Install with: pip install discord.py"
            )

        self.bot_token = bot_token
        self.allowed_channel_ids = parse_allowed_channels(allowed_channel_ids)
        if not self.bot_token:
            logger.warning("DISCORD_BOT_TOKEN not set")

        discord = _get_discord()
        intents = discord.Intents.default()
        intents.message_content = True

        assert _DiscordClient is not None
        self._client = _DiscordClient(self, intents)
        self._message_handler: InboundMessageHandler | None = None
        self._connected = False
        self._accepting_messages = False
        self._ready = asyncio.Event()
        self._inbound_tasks: set[asyncio.Task[Any]] = set()
        self._limiter = limiter
        self._start_task: asyncio.Task[None] | None = None
        self.outbound = DiscordMessenger(
            get_client=lambda: self._client,
            get_discord=_get_discord,
            limiter=limiter,
        )
        self._voice_flow = VoiceNoteFlow(
            transcriber=transcriber,
            log_raw_messaging_content=log_raw_messaging_content,
            log_api_error_tracebacks=log_api_error_tracebacks,
        )
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks

    async def _handle_client_message(self, message: Any) -> None:
        """Adapter entry point used by the internal Discord client."""
        if not self._accepting_messages:
            return
        task = asyncio.current_task()
        if task is not None:
            self._inbound_tasks.add(task)
        try:
            if self._accepting_messages:
                await self._on_discord_message(message)
        finally:
            if task is not None:
                self._inbound_tasks.discard(task)

    def _mark_connected(self) -> None:
        """Publish Discord readiness while this runtime accepts ingress."""
        if not self._accepting_messages:
            return
        self._connected = True
        self._ready.set()
        logger.info("Discord platform connected")

    async def cancel_pending_voice(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None:
        """Cancel a pending voice transcription."""
        return await self._voice_flow.cancel_pending_voice(scope, reply_id)

    async def cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel every pending voice transcription and handoff."""
        return await self._voice_flow.cancel_all_pending_voices()

    async def cancel_pending_voices_in_scope(
        self,
        scope: MessageScope,
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel pending voice transcriptions belonging to one chat."""
        return await self._voice_flow.cancel_pending_voices_in_scope(scope)

    async def _handle_voice_note(
        self, message: Any, attachment: Any, channel_id: str
    ) -> bool:
        """Handle a Discord audio attachment."""
        return await self._voice_flow.handle(
            discord_voice_request_from_event(message, attachment, channel_id),
            message_handler=self._message_handler,
            queue_send_message=self.outbound.queue_send_message,
            queue_delete_messages=self.outbound.queue_delete_messages,
        )

    async def _on_discord_message(self, message: Any) -> None:
        """Handle incoming Discord messages."""
        if message.author.bot:
            return

        channel_id = str(message.channel.id)
        if not self.allowed_channel_ids or channel_id not in self.allowed_channel_ids:
            return

        if not message.content:
            audio_att = get_audio_attachment(message)
            if audio_att:
                await self._handle_voice_note(message, audio_att, channel_id)
            return

        incoming = discord_text_message_from_event(
            message,
            log_raw_messaging_content=self._log_raw_messaging_content,
        )
        if self._message_handler is None:
            return

        try:
            await self._message_handler(incoming)
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.error("Error handling message: {}", e)
            else:
                logger.error("Error handling message: exc_type={}", type(e).__name__)
            with contextlib.suppress(Exception):
                await self.outbound.send_message(
                    channel_id,
                    format_status_discord("Error:", format_user_error_preview(e)),
                    reply_to=str(message.id),
                )

    async def start(self) -> None:
        """Initialize and connect to Discord."""
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN is required")

        self._limiter.start()
        self._accepting_messages = True
        self._ready.clear()

        self._start_task = asyncio.create_task(
            self._client.start(self.bot_token),
            name="discord-client-start",
        )
        self._start_task.add_done_callback(self._observe_client_exit)
        ready_task = asyncio.create_task(
            self._ready.wait(),
            name="discord-client-ready",
        )
        try:
            done, _pending = await asyncio.wait(
                (self._start_task, ready_task),
                timeout=30.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise RuntimeError("Discord client failed to connect within timeout")
            if self._start_task in done:
                await self._start_task
                raise RuntimeError("Discord client stopped before becoming ready")
            if self._start_task.done():
                await self._start_task
                raise RuntimeError("Discord client stopped unexpectedly")
        finally:
            ready_task.cancel()
            await asyncio.gather(ready_task, return_exceptions=True)

        logger.info("Discord platform started")

    def _observe_client_exit(self, task: asyncio.Task[None]) -> None:
        """Observe the long-lived Discord client task and publish lost readiness."""
        if task.cancelled():
            exception: BaseException | None = None
        else:
            exception = task.exception()

        was_connected = self._connected
        if not self._accepting_messages:
            return

        self._connected = False
        self._ready.clear()
        if not was_connected:
            return

        if exception is None:
            logger.error("Discord client stopped unexpectedly")
        elif self._log_api_error_tracebacks:
            logger.error("Discord client stopped unexpectedly: {}", exception)
        else:
            logger.error(
                "Discord client stopped unexpectedly: exc_type={}",
                type(exception).__name__,
            )

    async def quiesce(self) -> None:
        """Stop Discord ingress and drain active SDK handlers."""
        self._accepting_messages = False
        try:
            if not self._client.is_closed():
                await self._client.close()
        finally:
            try:
                await self._drain_start_task()
            finally:
                try:
                    await self._drain_inbound_tasks()
                finally:
                    self._connected = False
                    self._ready.clear()

    async def close(self) -> None:
        """Close Discord delivery resources after ingress is quiescent."""
        try:
            await self.outbound.close()
        finally:
            await self._limiter.shutdown()
            logger.info("Discord platform closed")

    async def _drain_start_task(self) -> None:
        task = self._start_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await asyncio.gather(task, return_exceptions=True)
        finally:
            if task.done() and self._start_task is task:
                self._start_task = None

    async def _drain_inbound_tasks(self) -> None:
        tasks = tuple(self._inbound_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        """Register the workflow callback for inbound messages."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Return whether Discord startup completed."""
        return self._connected
