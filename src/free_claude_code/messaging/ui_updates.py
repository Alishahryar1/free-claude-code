"""Throttled platform UI updates driven by transcript rendering."""

import time
from collections.abc import Callable

from loguru import logger

from .platforms.ports import OutboundMessenger
from .safe_diagnostics import format_exception_for_log
from .transcript import RenderCtx, TranscriptBuffer


class ThrottledTranscriptEditor:
    """Rate-limited status message edits from a growing transcript."""

    def __init__(
        self,
        *,
        outbound: OutboundMessenger,
        parse_mode: str | None,
        get_limit_chars: Callable[[], int],
        transcript: TranscriptBuffer,
        render_ctx: RenderCtx,
        node_id: str,
        chat_id: str,
        status_msg_id: str,
        debug_platform_edits: bool,
        log_messaging_error_details: bool = False,
    ) -> None:
        self._outbound = outbound
        self._parse_mode = parse_mode
        self._get_limit_chars = get_limit_chars
        self._transcript = transcript
        self._render_ctx = render_ctx
        self._node_id = node_id
        self._chat_id = chat_id
        self._status_msg_id = status_msg_id
        self._debug_platform_edits = debug_platform_edits
        self._log_messaging_error_details = log_messaging_error_details
        self._last_ui_update = 0.0
        self._last_displayed_text: str | None = None
        self._last_status: str | None = None

    @property
    def last_status(self) -> str | None:
        return self._last_status

    async def update(self, status: str | None = None, *, force: bool = False) -> None:
        """Render transcript + optional status line and edit the platform message."""
        now = time.time()
        if not force and now - self._last_ui_update < 1.0:
            return

        self._last_ui_update = now
        if status is not None:
            self._last_status = status
        try:
            display = self._transcript.render(
                self._render_ctx,
                limit_chars=self._get_limit_chars(),
                status=status,
            )
        except Exception as e:
            logger.warning(
                "Transcript render failed for node {}: {}",
                self._node_id,
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )
            return
        if display and display != self._last_displayed_text:
            logger.debug(
                "PLATFORM_EDIT: node_id={} chat_id={} msg_id={} force={} status={!r} chars={}",
                self._node_id,
                self._chat_id,
                self._status_msg_id,
                bool(force),
                status,
                len(display),
            )
            if self._debug_platform_edits:
                logger.debug("PLATFORM_EDIT_TEXT:\n{}", display)
            self._last_displayed_text = display
            try:
                await self._outbound.queue_edit_message(
                    self._chat_id,
                    self._status_msg_id,
                    display,
                    parse_mode=self._parse_mode,
                )
            except Exception as e:
                logger.warning(
                    "Failed to update platform for node {}: {}",
                    self._node_id,
                    format_exception_for_log(
                        e, log_full_message=self._log_messaging_error_details
                    ),
                )
