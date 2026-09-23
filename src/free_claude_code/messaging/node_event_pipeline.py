"""CLI event handling for a single queued node (transcript + session + errors)."""

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from free_claude_code.core.trace import trace_event

from .cli_event_constants import TRANSCRIPT_EVENT_TYPES, get_status_for_event
from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .safe_diagnostics import text_len_hint
from .transcript import TranscriptBuffer
from .trees import NodeClaim

RecordSession = Callable[[str], Awaitable[None]]
CompleteClaim = Callable[[str | None], Awaitable[None]]
FailClaim = Callable[[str, str], Awaitable[None]]


async def handle_session_info_event(
    event_data: dict[str, Any],
    claim: NodeClaim,
    captured_session_id: str | None,
    temp_session_id: str | None,
    *,
    cli_manager: ManagedClaudeSessionManagerProtocol,
    record_session: RecordSession,
) -> tuple[str | None, str | None]:
    """Handle session_info event; return updated (captured_session_id, temp_session_id)."""
    if event_data.get("type") != "session_info":
        return captured_session_id, temp_session_id

    real_session_id = event_data.get("session_id")
    if not real_session_id or not temp_session_id:
        return captured_session_id, temp_session_id

    registered = await cli_manager.register_real_session_id(
        temp_session_id,
        real_session_id,
    )
    if not registered:
        raise RuntimeError("Managed Claude session registration failed.")
    trace_event(
        stage="claude_cli",
        event="claude_cli.session.registered",
        source="claude_cli",
        node_id=claim.node.node_id,
        temp_session_id=temp_session_id,
        real_session_id=real_session_id,
        tree_root_id=claim.identity.root_id,
    )
    await record_session(real_session_id)

    return real_session_id, None


async def process_parsed_cli_event(
    parsed: dict[str, Any],
    transcript: TranscriptBuffer,
    update_ui: Callable[..., Awaitable[None]],
    last_status: str | None,
    had_transcript_events: bool,
    claim: NodeClaim,
    captured_session_id: str | None,
    *,
    format_status: Callable[..., str],
    complete_claim: CompleteClaim,
    fail_claim: FailClaim,
    log_messaging_error_details: bool = False,
) -> tuple[str | None, bool]:
    """Process a single parsed CLI event. Returns (last_status, had_transcript_events)."""
    ptype = parsed.get("type") or ""

    if ptype in TRANSCRIPT_EVENT_TYPES:
        transcript.apply(parsed)
        had_transcript_events = True

    status = get_status_for_event(ptype, parsed, format_status)
    if status is not None:
        await update_ui(status)
        last_status = status
    elif ptype == "block_stop":
        await update_ui(last_status, force=True)
    elif ptype == "complete":
        if parsed.get("status") != "success":
            return last_status, had_transcript_events
        if not had_transcript_events:
            transcript.apply({"type": "text_chunk", "text": "Done."})
        trace_event(
            stage="claude_cli",
            event="turn.completed",
            source="cli_event",
            node_id=claim.node.node_id,
            claude_session_id=captured_session_id,
        )
        await update_ui(format_status("✅", "Complete"), force=True)
        await complete_claim(captured_session_id)
    elif ptype == "error":
        error_msg = parsed.get("message", "Unknown error")
        em = error_msg if isinstance(error_msg, str) else str(error_msg)
        trace_event(
            stage="claude_cli",
            event="turn.failed",
            source="cli_event",
            node_id=claim.node.node_id,
            claude_session_id=captured_session_id,
            cli_error_message=em,
        )
        if log_messaging_error_details:
            logger.error("HANDLER: Error event received: {}", error_msg)
        else:
            logger.error(
                "HANDLER: Error event received: message_chars={}",
                text_len_hint(em),
            )
        await update_ui(format_status("❌", "Error"), force=True)
        await fail_claim(em, "Parent task failed")

    return last_status, had_transcript_events
