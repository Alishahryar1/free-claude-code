"""Messaging platform ports used by the customer-facing workflow."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..models import IncomingMessage, MessageScope
from ..voice import VoiceCancellationResult

InboundMessageHandler = Callable[[IncomingMessage], Awaitable[None]]


@runtime_checkable
class MessagingRuntime(Protocol):
    """Owns ingress and delivery lifecycle for one messaging platform."""

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

    async def quiesce(self) -> None: ...

    async def close(self) -> None: ...

    def on_message(self, handler: InboundMessageHandler) -> None: ...

    @property
    def is_connected(self) -> bool: ...


@runtime_checkable
class OutboundMessenger(Protocol):
    """Owns queued outbound platform delivery."""

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None: ...

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None: ...

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None: ...

    def fire_and_forget(self, task: Awaitable[Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class MessagingStartupNotice:
    """One clearable customer notice declared by a platform composition."""

    chat_id: str
    transport_label: str


@runtime_checkable
class VoiceCancellation(Protocol):
    """Optional voice-note cancellation boundary used by stop/clear flows."""

    async def cancel_pending_voice(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None: ...

    async def cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]: ...

    async def cancel_pending_voices_in_scope(
        self,
        scope: MessageScope,
    ) -> tuple[VoiceCancellationResult, ...]: ...


@dataclass(frozen=True, slots=True)
class MessagingPlatformComponents:
    """Runtime/outbound bundle for one configured messaging platform."""

    name: str
    runtime: MessagingRuntime
    outbound: OutboundMessenger
    voice_cancellation: VoiceCancellation | None = None
    startup_notice: MessagingStartupNotice | None = None
