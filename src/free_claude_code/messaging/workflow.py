"""Messaging workflow coordinator for Discord and Telegram prompts."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from loguru import logger

from free_claude_code.core.trace import trace_event

from .command_context import ReplyClearResult, StopOutcome
from .command_dispatcher import parse_command_base
from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .models import AdmissionToken, IncomingMessage, MessageScope
from .node_runner import MessagingNodeRunner
from .platforms.ports import (
    MessagingStartupNotice,
    OutboundMessenger,
    VoiceCancellation,
)
from .rendering.profiles import build_rendering_profile
from .safe_diagnostics import format_exception_for_log
from .session import SessionStore
from .transcript import RenderCtx
from .trees import (
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    ConversationSnapshot,
    FailureResult,
    NodeUiTarget,
    QueueDecision,
    ReplyTarget,
    TreeQueueManager,
)
from .turn_intake import MessagingTurnIntake
from .voice import VoiceCancellationResult


async def _finish_owned_operation[T](
    operation: Coroutine[Any, Any, T],
    *,
    name: str,
) -> T:
    """Finish owned work, preserving its failures before caller cancellation."""
    task = asyncio.create_task(operation, name=name)
    current = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if current is not None and current.cancelling():
                cancellation = cancellation or exc
        except Exception:
            break

    # Task.result() raises the owned operation's failure. Caller cancellation
    # is restored only after successful completion.
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


def _stop_outcome(
    voices: tuple[VoiceCancellationResult, ...],
    cancellation: CancellationResult,
) -> StopOutcome:
    """Summarize distinct stopped owners and whether existing status UI covers them."""
    status_coverage: dict[tuple[MessageScope, str], bool] = {}
    for voice in voices:
        key = (voice.scope, voice.voice_message_id)
        status_coverage[key] = (
            status_coverage.get(key, False) or voice.status_message_id is not None
        )
    for effect in cancellation.effects:
        status_coverage[(effect.node.scope, effect.node.node_id)] = True
    return StopOutcome(
        cancelled_count=len(status_coverage),
        status_feedback_scopes=frozenset(
            scope for (scope, _owner_id), covered in status_coverage.items() if covered
        ),
        fallback_required=any(not covered for covered in status_coverage.values()),
    )


class MessagingWorkflow:
    """Own messaging state transitions and their external side effects."""

    def __init__(
        self,
        outbound: OutboundMessenger,
        cli_manager: ManagedClaudeSessionManagerProtocol,
        session_store: SessionStore,
        *,
        platform_name: str | None = None,
        voice_cancellation: VoiceCancellation | None = None,
        debug_platform_edits: bool = False,
        debug_subagent_stack: bool = False,
        log_raw_cli_diagnostics: bool = False,
        log_messaging_error_details: bool = False,
    ) -> None:
        self.platform_name = platform_name or "messaging"
        self.outbound = outbound
        self.voice_cancellation = voice_cancellation
        self.cli_manager = cli_manager
        self.session_store = session_store
        self._log_messaging_error_details = log_messaging_error_details
        self._rendering_profile = build_rendering_profile(self.platform_name)
        self._state_lock = asyncio.Lock()
        self._stop_generation = 0
        self._clear_generations: dict[MessageScope, int] = {}
        self._pending_restored_status_targets: tuple[NodeUiTarget, ...] = ()

        self._tree_queue: TreeQueueManager
        self.node_runner = MessagingNodeRunner(
            platform_name=self.platform_name,
            outbound=outbound,
            cli_manager=cli_manager,
            session_store=session_store,
            get_tree_queue=lambda: self._tree_queue,
            format_status=self.format_status,
            get_parse_mode=self._parse_mode,
            get_render_ctx=self.get_render_ctx,
            get_limit_chars=self._get_limit_chars,
            debug_platform_edits=debug_platform_edits,
            debug_subagent_stack=debug_subagent_stack,
            log_raw_cli_diagnostics=log_raw_cli_diagnostics,
            log_messaging_error_details=log_messaging_error_details,
        )
        self.turn_intake = MessagingTurnIntake(
            platform_name=self.platform_name,
            outbound=outbound,
            session_store=session_store,
            command_context=self,
            resolve_reply=self.resolve_reply,
            admit_turn=self._admit_turn_if_current,
            format_status=self.format_status,
            get_parse_mode=self._parse_mode,
            record_outgoing_message=self.record_outgoing_message,
        )
        self._tree_queue = self._build_tree_queue()

    def _build_tree_queue(
        self, snapshot: ConversationSnapshot | None = None
    ) -> TreeQueueManager:
        if snapshot is None:
            return TreeQueueManager(
                self.node_runner.process_node,
                queue_update_callback=self.turn_intake.update_queue_positions,
                node_started_callback=self.turn_intake.mark_node_processing,
                unexpected_failure_callback=self._apply_unexpected_failure,
                log_messaging_error_details=self._log_messaging_error_details,
            )
        return TreeQueueManager.from_snapshot(
            snapshot,
            self.node_runner.process_node,
            queue_update_callback=self.turn_intake.update_queue_positions,
            node_started_callback=self.turn_intake.mark_node_processing,
            unexpected_failure_callback=self._apply_unexpected_failure,
            log_messaging_error_details=self._log_messaging_error_details,
        )

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        return self._rendering_profile.format_status(emoji, label, suffix)

    def _parse_mode(self) -> str | None:
        return self._rendering_profile.parse_mode

    def get_render_ctx(self) -> RenderCtx:
        return self._rendering_profile.render_ctx

    def _get_limit_chars(self) -> int:
        return self._rendering_profile.limit_chars

    @property
    def tree_queue(self) -> TreeQueueManager:
        """Expose the manager facade for diagnostics and smoke tests."""
        return self._tree_queue

    def restore(self) -> None:
        """Restore and reconcile persisted conversations before platform start."""
        snapshot = self.session_store.load_conversation_snapshot()
        if snapshot.is_empty:
            return
        logger.info("Restoring {} conversation trees...", len(snapshot.trees))
        self._tree_queue = self._build_tree_queue(snapshot)
        normalized = self._tree_queue.restored_snapshot
        if normalized is not None and normalized != snapshot:
            self.session_store.save_conversation_snapshot(normalized)
        self._pending_restored_status_targets = self._tree_queue.restored_stale_targets

    async def repair_restored_statuses(self) -> None:
        """Replace stale queued/processing UI after delivery becomes available."""
        targets = self._pending_restored_status_targets
        self._pending_restored_status_targets = ()
        for target in targets:
            if self.platform_name != "messaging" and (
                target.scope.platform != self.platform_name
            ):
                continue
            try:
                await self.outbound.queue_edit_message(
                    target.scope.chat_id,
                    target.status_message_id,
                    self.format_status("❌", "Interrupted by server restart"),
                    parse_mode=self._parse_mode(),
                    fire_and_forget=False,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to repair restored status for node {}: {}",
                    target.node_id,
                    type(exc).__name__,
                )

    async def publish_startup_notice(self, notice: MessagingStartupNotice) -> None:
        """Publish one notice, then transfer its receipt to clear ownership."""
        scope = MessageScope(platform=self.platform_name, chat_id=notice.chat_id)
        async with self._state_lock:
            clear_generation = self._clear_generations.get(scope, 0)

        try:
            message_id = await self.outbound.queue_send_message(
                notice.chat_id,
                self.format_status(
                    "🚀",
                    "Claude Code Proxy is online!",
                    f"({notice.transport_label})",
                ),
                parse_mode=self._parse_mode(),
                fire_and_forget=False,
            )
        except Exception as exc:
            logger.warning(
                "Could not publish messaging startup notice: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )
            return

        if not message_id:
            return

        publisher_task = asyncio.current_task()
        await _finish_owned_operation(
            self._finalize_startup_notice(
                notice,
                message_id,
                clear_generation,
                publisher_task,
            ),
            name="messaging-finalize-startup-notice",
        )

    async def _finalize_startup_notice(
        self,
        notice: MessagingStartupNotice,
        message_id: str,
        clear_generation: int,
        publisher_task: asyncio.Task[Any] | None,
    ) -> None:
        """Commit one delivery receipt or compensate it outside the state lock."""
        async with self._state_lock:
            must_discard = (
                publisher_task is not None and publisher_task.cancelling() > 0
            ) or clear_generation != self._clear_generations.get(
                MessageScope(platform=self.platform_name, chat_id=notice.chat_id), 0
            )
            if not must_discard:
                must_discard = not self.record_outgoing_message(
                    self.platform_name,
                    notice.chat_id,
                    message_id,
                    "startup",
                )

        if must_discard:
            await self._discard_startup_notice(notice.chat_id, message_id)

    async def _discard_startup_notice(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete an uncommitted notice or retain its ID for a later clear."""
        try:
            await self.outbound.queue_delete_messages(
                chat_id,
                [message_id],
                fire_and_forget=False,
            )
        except Exception as exc:
            logger.warning(
                "Could not discard messaging startup notice: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )
            async with self._state_lock:
                tracked = self.record_outgoing_message(
                    self.platform_name,
                    chat_id,
                    message_id,
                    "startup",
                )
            if not tracked:
                logger.warning(
                    "Messaging startup notice could neither be deleted nor tracked"
                )
            return

        async with self._state_lock:
            self.forget_tracked_message_ids(
                self.platform_name,
                chat_id,
                {message_id},
            )

    async def close(self) -> None:
        """Finish every owned task and durable write before releasing delivery."""
        await self.stop_all_tasks()
        await self._tree_queue.wait_idle()
        self.session_store.flush_pending_save()

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """Handle one platform message."""
        trace_event(
            stage="ingress",
            event="turn.received",
            source=self.platform_name,
            chat_id=incoming.chat_id,
            platform_message_id=incoming.message_id,
            reply_to_message_id=incoming.reply_to_message_id,
            thread_id=incoming.message_thread_id,
            message_text=incoming.text or "",
        )
        with logger.contextualize(
            chat_id=incoming.chat_id,
            node_id=incoming.message_id,
        ):
            is_standalone_clear = (
                parse_command_base(incoming.text) == "/clear"
                and not incoming.is_reply()
            )
            async with self._state_lock:
                admission_token = AdmissionToken(
                    stop_generation=self._stop_generation,
                    clear_generation=self._clear_generations.get(incoming.scope, 0),
                )
                if not is_standalone_clear:
                    self._record_incoming_message(incoming)
            try:
                await self.turn_intake.handle_message(
                    incoming,
                    admission_token=admission_token,
                )
            except BaseException:
                if is_standalone_clear:
                    async with self._state_lock:
                        self._record_incoming_message(incoming)
                raise

    async def resolve_reply(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> ReplyTarget | None:
        return await self._tree_queue.resolve_reply(scope, reference_id)

    async def _admit_turn_if_current(
        self,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_reference_id: str | None,
        admission_token: AdmissionToken,
    ) -> QueueDecision | None:
        return await _finish_owned_operation(
            self._admit_if_current(
                incoming,
                status_message_id,
                parent_reference_id,
                admission_token,
            ),
            name=f"messaging-admit-{incoming.message_id}",
        )

    async def _admit_if_current(
        self,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_reference_id: str | None,
        admission_token: AdmissionToken,
    ) -> QueueDecision | None:
        """Commit admission and its exact snapshot as one owned transaction."""
        async with self._state_lock:
            current_token = AdmissionToken(
                stop_generation=self._stop_generation,
                clear_generation=self._clear_generations.get(incoming.scope, 0),
            )
            if admission_token != current_token:
                return None
            decision = await self._tree_queue.admit(
                incoming,
                status_message_id,
                parent_reference_id=parent_reference_id,
            )
            if decision.snapshot is not None:
                self.session_store.save_tree_snapshot(decision.snapshot)
            return decision

    def get_tree_count(self) -> int:
        return self._tree_queue.get_tree_count()

    async def stop_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> StopOutcome:
        """Stop the exact voice/tree owner of one replied-to message."""
        return await _finish_owned_operation(
            self._stop_reply(scope, reply_id),
            name=f"messaging-stop-reply-{reply_id}",
        )

    async def _stop_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> StopOutcome:
        voice_result = await self._cancel_pending_voice(scope, reply_id)
        if voice_result is not None:
            self.render_voice_stopped(voice_result)

        async with self._state_lock:
            node_id = await self._tree_queue.resolve_node_id(scope, reply_id)
            if node_id is None:
                voices = (voice_result,) if voice_result is not None else ()
                return _stop_outcome(voices, CancellationResult())
            result = await self._tree_queue.cancel_node(
                scope,
                node_id,
                reason=CancellationReason.STOP,
            )
            self._apply_cancellation_result(result)

        voices = (voice_result,) if voice_result is not None else ()
        return _stop_outcome(voices, result)

    async def clear_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> ReplyClearResult | None:
        """Clear the exact voice/tree owner of one replied-to message."""
        return await _finish_owned_operation(
            self._clear_reply(scope, reply_id),
            name=f"messaging-clear-reply-{reply_id}",
        )

    async def _clear_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> ReplyClearResult | None:
        voice_result = await self._cancel_pending_voice(scope, reply_id)

        async with self._state_lock:
            subtree = await self._tree_queue.remove_message_subtree(
                scope,
                reply_id,
                reason=CancellationReason.CLEAR,
            )
            if not subtree.tree_matched and voice_result is None:
                return None
            self._save_cancellation_snapshots(subtree.cancellation)
            if subtree.removed_tree_identity is not None:
                self.session_store.remove_tree_snapshot(subtree.removed_tree_identity)

        delete_message_ids = set(subtree.delete_message_ids)
        if voice_result is not None:
            delete_message_ids.update(voice_result.delete_message_ids)
        return ReplyClearResult(
            delete_message_ids=frozenset(delete_message_ids),
            tree_matched=subtree.tree_matched,
        )

    async def stop_all_tasks(self) -> StopOutcome:
        """Stop every pending and active messaging task."""
        return await _finish_owned_operation(
            self._stop_all_tasks(),
            name="messaging-stop-all",
        )

    async def _stop_all_tasks(self) -> StopOutcome:
        voice_results = await self._cancel_all_pending_voices()
        for voice in voice_results:
            self.render_voice_stopped(voice)
        async with self._state_lock:
            self._stop_generation += 1
            logger.info("Cancelling tree queue tasks...")
            result = await self._tree_queue.cancel_all(reason=CancellationReason.STOP)
            logger.info("Cancelled {} nodes", len(result.effects))
            self._apply_cancellation_result(result)
            logger.info("Stopping all CLI sessions...")
            await self.cli_manager.stop_all()
            return _stop_outcome(voice_results, result)

    async def clear_chat(self, platform: str, chat_id: str) -> frozenset[str]:
        """Clear FCC state atomically with respect to later turn admission."""
        return await _finish_owned_operation(
            self._clear_chat(platform, chat_id),
            name=f"messaging-clear-{platform}-{chat_id}",
        )

    async def _clear_chat(
        self,
        platform: str,
        chat_id: str,
    ) -> frozenset[str]:
        """Reset one chat's FCC state and return all tracked deletion IDs."""
        clear_scope = MessageScope(platform=platform, chat_id=chat_id)
        voice_results = await self._cancel_pending_voices_in_scope(clear_scope)
        async with self._state_lock:
            delete_message_ids: set[str] = set()
            for voice in voice_results:
                delete_message_ids.update(voice.delete_message_ids)
            delete_message_ids.update(
                self.session_store.get_tracked_message_ids_for_chat(platform, chat_id)
            )

            delete_message_ids.update(
                await self._tree_queue.get_message_ids_for_chat(platform, chat_id)
            )
            # All fallible/cancellable reads precede the commit boundary. Once
            # the scope generation advances, state removal is one-way work.
            self._clear_generations[clear_scope] = (
                self._clear_generations.get(clear_scope, 0) + 1
            )
            failures: list[Exception] = []
            try:
                await self._tree_queue.clear_scope(
                    clear_scope,
                    reason=CancellationReason.CLEAR,
                )
            except Exception as exc:
                failures.append(exc)
            try:
                self.session_store.clear_scope(clear_scope)
            except Exception as exc:
                failures.append(exc)
                logger.warning(
                    "Failed to persist final cleared session state: {}",
                    type(exc).__name__,
                )
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise ExceptionGroup("Chat clear failed", failures)
            return frozenset(delete_message_ids)

    async def _cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]:
        cancellation = self.voice_cancellation
        if cancellation is None:
            return ()
        return await cancellation.cancel_all_pending_voices()

    async def _cancel_pending_voices_in_scope(
        self,
        scope: MessageScope,
    ) -> tuple[VoiceCancellationResult, ...]:
        cancellation = self.voice_cancellation
        if cancellation is None:
            return ()
        return await cancellation.cancel_pending_voices_in_scope(scope)

    async def _cancel_pending_voice(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> VoiceCancellationResult | None:
        cancellation = self.voice_cancellation
        if cancellation is None:
            return None
        return await cancellation.cancel_pending_voice(scope, reply_id)

    def render_voice_stopped(self, result: VoiceCancellationResult) -> None:
        """Publish terminal UI for a voice cancelled before tree ownership."""
        if result.status_message_id is None:
            return
        self.outbound.fire_and_forget(
            self.outbound.queue_edit_message(
                result.scope.chat_id,
                result.status_message_id,
                self.format_status("⏹", "Stopped."),
                parse_mode=self._parse_mode(),
            )
        )

    def forget_tracked_message_ids(
        self,
        platform: str,
        chat_id: str,
        message_ids: set[str],
    ) -> None:
        try:
            self.session_store.forget_tracked_message_ids(
                platform, chat_id, message_ids
            )
        except Exception as exc:
            logger.warning(
                "Failed to update managed-message log after clear: {}",
                type(exc).__name__,
            )

    def record_outgoing_message(
        self,
        platform: str,
        chat_id: str,
        msg_id: str | None,
        kind: str,
    ) -> bool:
        """Record an outgoing message ID for /clear and report ownership."""
        if not msg_id:
            return False
        try:
            self.session_store.record_message_id(
                platform,
                chat_id,
                str(msg_id),
                "out",
                kind,
            )
        except Exception as exc:
            logger.debug(
                "Failed to record message_id: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )
            return False
        return True

    def _record_incoming_message(self, incoming: IncomingMessage) -> bool:
        """Record an inbound prompt, voice note, or command for standalone clear."""
        command = parse_command_base(incoming.text)
        kind = (
            "command"
            if command.startswith("/")
            else "voice"
            if incoming.status_message_id is not None
            else "prompt"
        )
        try:
            self.session_store.record_message_id(
                incoming.platform,
                incoming.chat_id,
                str(incoming.message_id),
                "in",
                kind,
            )
        except Exception as exc:
            logger.debug(
                "Failed to record managed inbound message_id: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )
            return False
        return True

    def _save_cancellation_snapshots(self, result: CancellationResult) -> None:
        """Persist transition snapshots without publishing cancellation UI."""
        for snapshot in result.snapshots:
            self.session_store.save_tree_snapshot(snapshot)

    def _apply_cancellation_result(self, result: CancellationResult) -> None:
        """Apply detached UI and persistence effects from one transition."""
        for effect in result.effects:
            if effect.ui_owner is CancellationUiOwner.WORKFLOW:
                self.outbound.fire_and_forget(
                    self.outbound.queue_edit_message(
                        effect.node.scope.chat_id,
                        effect.node.status_message_id,
                        self.format_status("⏹", "Stopped."),
                        parse_mode=self._parse_mode(),
                    )
                )
        self._save_cancellation_snapshots(result)

    def _apply_unexpected_failure(self, result: FailureResult) -> None:
        """Persist and render a failure that escaped the total node runner."""
        if result.snapshot is not None:
            self.session_store.save_tree_snapshot(result.snapshot)
        for target in result.affected:
            self.outbound.fire_and_forget(
                self.outbound.queue_edit_message(
                    target.scope.chat_id,
                    target.status_message_id,
                    self.format_status("💥", "Task Failed"),
                    parse_mode=self._parse_mode(),
                )
            )


__all__ = ["MessagingWorkflow"]
