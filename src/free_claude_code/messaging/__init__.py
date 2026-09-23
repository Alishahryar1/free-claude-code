"""Platform-agnostic messaging layer."""

from .managed_protocols import (
    ManagedClaudeSessionManagerProtocol,
    ManagedClaudeSessionProtocol,
)
from .models import IncomingMessage, MessageScope
from .platforms.ports import OutboundMessenger

__all__ = [
    "IncomingMessage",
    "ManagedClaudeSessionManagerProtocol",
    "ManagedClaudeSessionProtocol",
    "MessageScope",
    "OutboundMessenger",
]
