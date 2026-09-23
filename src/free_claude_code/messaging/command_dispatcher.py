"""Command parsing and dispatch for messaging handlers."""

from .command_context import MessagingCommandContext
from .commands import handle_clear_command, handle_stats_command, handle_stop_command
from .models import IncomingMessage

_COMMAND_HANDLERS = {
    "/clear": handle_clear_command,
    "/stop": handle_stop_command,
    "/stats": handle_stats_command,
}


def parse_command_base(text: str | None) -> str:
    """Return the slash command without bot mention suffix."""
    parts = (text or "").strip().split()
    cmd = parts[0] if parts else ""
    return cmd.split("@", 1)[0] if cmd else ""


async def dispatch_command(
    context: MessagingCommandContext,
    incoming: IncomingMessage,
    command_base: str,
) -> bool:
    """Dispatch a known command and return whether it was handled."""
    command = _COMMAND_HANDLERS.get(command_base)
    if command is None:
        return False
    await command(context, incoming)
    return True
