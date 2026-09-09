"""Server-owned coding sessions, independent of HTTP and browser lifetimes."""

import asyncio
import uuid
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

from loguru import logger

from free_claude_code.application.session_events import (
    EventPublisher,
    EventSubscription,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .models import (
    ACTIVE_RUN_STATUSES,
    CodeCatalog,
    CodeConflictError,
    CodeDetail,
    CodeItem,
    CodeMode,
    CodeNotFoundError,
    CodePage,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
    CodeValidationError,
    HarnessEvent,
    ItemUpdate,
    NativeHistoryMissing,
    NativeThread,
    RunStatus,
    now_ms,
)
from .ports import CodeStore, HarnessConnection, HarnessFactory, HarnessSelection


@dataclass(slots=True)
class _Owner:
    session: CodeSession
    run: CodeRun | None
    items: dict[str, CodeItem]
    prompts: dict[str, CodePrompt]
    runs: dict[str, CodeRun]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    connection: HarnessConnection | None = None
    generation: str | None = None
    version: int = 0
    sequence: int = 0
    job: asyncio.Task[None] | None = None
    flush_task: asyncio.Task[None] | None = None
    interrupt_task: asyncio.Task[None] | None = None
    interrupt_for: str | None = None
    loaded_thread_id: str | None = None
    dirty: set[str] = field(default_factory=set)
    known_turns: set[str] = field(default_factory=set)
    review_generations: dict[str, str] = field(default_factory=dict)
    dirty_characters: int = 0
    storage_failed: bool = False
    failure_task: asyncio.Task[None] | None = None
    deleted: bool = False

    @property
    def busy(self) -> bool:
        return self.run is not None and self.run.status in ACTIVE_RUN_STATUSES

    @property
    def pending(self) -> bool:
        return any(
            prompt.status in {"pending", "answering"}
            for prompt in self.prompts.values()
        )

    @property
    def active_review_ids(self) -> tuple[str, ...]:
        return tuple(
            item_id
            for item_id, generation in self.review_generations.items()
            if generation == self.generation
        )


class CodeService:
    def __init__(self, store: CodeStore, harness: HarnessFactory) -> None:
        self._store = store
        self._harness = harness
        self._owners: dict[str, _Owner] = {}
        self._load_lock = asyncio.Lock()
        self._events = EventPublisher()
        self.epoch = str(uuid.uuid4())
        self._commands: set[asyncio.Task] = set()
        self._jobs: set[asyncio.Task] = set()
        self._accepting = False
        self._started = False
        self._message: str | None = "Code sessions is starting."
        self._close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._started:
            return
        try:
            await self._store.start()
            self._started = True
            pending = await self._store.pending_deletions()
        except Exception as exc:
            await self._store.close()
            self._started = False
            self._message = _error_message(exc)
            return
        except BaseException:
            await self._store.close()
            raise
        self._accepting = True
        self._message = None
        for session in pending:
            owner = await self._owner(session.id)
            owner.finished = asyncio.Event()
            owner.job = self._job(self._delete_native(owner, reconcile=True))

    def availability(self) -> tuple[bool, str | None]:
        if not self._accepting:
            return False, self._message
        return self._harness.availability()

    def catalog(self) -> CodeCatalog:
        return self._harness.catalog()

    def begin_shutdown(self) -> None:
        self._accepting = False
        self._message = "Code sessions is stopping."
        self._events.disconnect_subscribers()

    async def close(self) -> None:
        self.begin_shutdown()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="fcc-code-close")
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._commands:
            await asyncio.gather(*tuple(self._commands), return_exceptions=True)
        owners = tuple(self._owners.values())
        for owner in owners:
            if owner.busy and owner.run is not None and not owner.storage_failed:
                try:
                    await self._stop(owner.session.id, owner.run.id, shutting_down=True)
                except Exception:
                    await self._close_connection(owner)
                    self._storage_failure(owner)
        jobs = [owner.job for owner in owners if owner.job is not None]
        if jobs:
            await asyncio.wait(jobs, timeout=5)
        await asyncio.gather(*(self._close_connection(owner) for owner in owners))
        for owner in owners:
            async with owner.lock:
                if owner.busy and not owner.storage_failed:
                    try:
                        await self._finish_locked(
                            owner,
                            "interrupted",
                            "FCC stopped before this turn finished.",
                        )
                    except Exception:
                        self._storage_failure(owner)
        if self._jobs:
            await asyncio.gather(*tuple(self._jobs), return_exceptions=True)
        self._events.close()
        await self._store.close()
        self._started = False
        self._message = "Code sessions is stopped."

    async def _command[T](self, work: Coroutine[object, object, T]) -> T:
        task = asyncio.create_task(work)
        self._commands.add(task)
        task.add_done_callback(self._command_done)
        return await asyncio.shield(task)

    def _command_done(self, task: asyncio.Task) -> None:
        self._commands.discard(task)
        if not task.cancelled():
            task.exception()

    def _job(self, work: Coroutine[object, object, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(work)
        self._jobs.add(task)
        task.add_done_callback(self._job_done)
        return task

    def _job_done(self, task: asyncio.Task) -> None:
        self._jobs.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning(
                "Code session task failed: exc_type={}", type(error).__name__
            )

    def _require_available(self) -> None:
        if not self._accepting:
            raise CodeUnavailableError(self._message or "Code sessions is unavailable.")

    async def _owner(self, session_id: str) -> _Owner:
        _validate_id(session_id)
        async with self._load_lock:
            owner = self._owners.get(session_id)
            if owner is None:
                session = await self._store.get_session(session_id)
                items = await self._store.items(session_id, None, None)
                runs = await self._store.runs(session_id)
                owner = _Owner(
                    session,
                    await self._store.latest_run(session_id),
                    {item.id: item for item in items},
                    {
                        prompt.id: prompt
                        for prompt in await self._store.prompts(session_id)
                    },
                    {run.id: run for run in runs},
                    sequence=max((item.sequence for item in items), default=0),
                    known_turns={
                        run.native_turn_id for run in runs if run.native_turn_id
                    },
                )
                if owner.run and owner.run.native_turn_id:
                    owner.known_turns.add(owner.run.native_turn_id)
                if not owner.busy:
                    owner.finished.set()
                self._owners[session_id] = owner
        if owner.deleted:
            raise CodeNotFoundError("Code session was deleted.")
        return owner

    async def create_session(self, session_id: str, cwd: str) -> CodeSession:
        return await self._command(self._create(session_id, cwd))

    async def _create(self, session_id: str, cwd: str) -> CodeSession:
        self._require_available()
        _validate_id(session_id)
        try:
            folder = Path(cwd).expanduser().resolve(strict=True)
            if not folder.is_dir():
                raise ValueError
        except OSError, ValueError, RuntimeError:
            raise CodeValidationError(
                "Choose an existing folder on the FCC computer."
            ) from None
        session = await self._store.create(
            CodeSession(
                id=session_id, cwd=str(folder), model=self.catalog().default_model
            )
        )
        owner = await self._owner(session.id)
        async with owner.lock:
            self._publish(owner, "session.updated")
        return session

    async def list_sessions(
        self, cursor: tuple[int, str] | None = None, limit: int = 25, query: str = ""
    ) -> CodePage:
        return await self._store.list_sessions(cursor, max(1, min(limit, 25)), query)

    async def subscribe(self) -> tuple[EventSubscription, JsonObject]:
        self._require_available()
        subscription = self._events.subscribe()
        summaries = [
            self._summary(owner)
            for owner in tuple(self._owners.values())
            if not owner.deleted
            and (
                owner.busy
                or owner.pending
                or owner.active_review_ids
                or owner.session.status != "ready"
            )
        ]
        return subscription, {
            "epoch": self.epoch,
            "cursor": subscription.cursor,
            "sessions": summaries,
        }

    @property
    def cursor(self) -> int:
        return self._events.cursor

    async def get_detail(
        self,
        session_id: str,
        *,
        before: tuple[int, int] | None = None,
        include_item_ids: Sequence[str] = (),
    ) -> CodeDetail:
        owner = await self._owner(session_id)
        async with owner.lock:
            self._check_owner(owner)
            await self._flush_locked(owner)
            page_before = before
            if owner.busy and owner.run:
                active_start = (owner.run.ordinal, 0)
                page_before = min(before, active_start) if before else active_start
            page = await self._store.item_page(session_id, page_before, 50)
            included_ids = {*owner.active_review_ids, *include_item_ids}
            extra = [
                item
                for item in owner.items.values()
                if (owner.busy and owner.run and item.run_id == owner.run.id)
                or item.id in included_ids
                or (
                    item.kind == "prompt"
                    and owner.prompts[item.id].status in {"pending", "answering"}
                )
            ]
            selected = sorted(
                {item.id: item for item in [*page.items, *extra]}.values(),
                key=lambda item: (owner.runs[item.run_id].ordinal, item.sequence),
            )
            prompt_ids = {item.id for item in selected if item.kind == "prompt"}
            prompts = tuple(
                prompt for prompt in owner.prompts.values() if prompt.id in prompt_ids
            )
            return CodeDetail(
                owner.session,
                owner.run,
                tuple(selected),
                prompts,
                self.epoch,
                owner.version,
                self.cursor,
                page.next_before,
                tuple(
                    {
                        run.id: run
                        for run in (
                            *page.runs,
                            *(owner.runs[item.run_id] for item in extra),
                            *((owner.run,) if owner.run else ()),
                        )
                    }.values()
                ),
                tuple(
                    prompt.id
                    for prompt in owner.prompts.values()
                    if prompt.status in {"pending", "answering"}
                ),
                owner.active_review_ids,
            )

    async def update_settings(
        self, session_id: str, revision: int, changes: JsonObject
    ) -> CodeSession:
        return await self._command(self._update_settings(session_id, revision, changes))

    async def _update_settings(
        self, session_id: str, revision: int, changes: JsonObject
    ) -> CodeSession:
        if not changes or changes.keys() - {
            "title",
            "model",
            "reasoning_effort",
            "mode",
        }:
            raise CodeValidationError(
                "Choose a title, model, effort or mode to update."
            )
        owner = await self._owner(session_id)
        async with owner.lock:
            self._editable(owner, revision)
            updates: dict[str, object] = dict(changes)
            if "title" in changes:
                title = changes["title"]
                if not isinstance(title, str) or not title.strip() or len(title) > 200:
                    raise CodeValidationError(
                        "Enter a title of at most 200 characters."
                    )
                updates.update(title=title.strip(), auto_title=False)
            if changes.keys() & {"model", "reasoning_effort", "mode"} and (
                owner.busy or owner.pending
            ):
                raise CodeConflictError(
                    "Finish this turn and its prompts before changing settings."
                )
            if "mode" in changes and changes["mode"] not in get_args(
                CodeMode.__value__
            ):
                raise CodeValidationError("Choose an available permission mode.")
            if changes.keys() & {"model", "reasoning_effort"}:
                model = changes.get("model", owner.session.model)
                entry = next(
                    (entry for entry in self.catalog().models if entry.id == model),
                    None,
                )
                if entry is None:
                    raise CodeValidationError(
                        "This model is unavailable. Choose another model."
                    )
                effort = changes.get("reasoning_effort", owner.session.reasoning_effort)
                if (
                    "reasoning_effort" not in changes
                    and model != owner.session.model
                    and effort not in entry.reasoning_efforts
                ):
                    effort = None
                if effort is not None and effort not in entry.reasoning_efforts:
                    raise CodeValidationError(
                        "This effort is unavailable. Choose another effort."
                    )
                updates.update(model=model, reasoning_effort=effort)
            session = await self._store.update_settings(
                _revision(owner.session, **updates), revision
            )
            owner.session = session
            self._publish(owner, "session.updated")
            return session

    async def send(
        self,
        session_id: str,
        operation_id: str,
        revision: int,
        text: str,
        *,
        expected_epoch: str,
    ) -> CodeRun:
        return await self._command(
            self._send(session_id, operation_id, revision, text, expected_epoch)
        )

    async def _send(
        self,
        session_id: str,
        operation_id: str,
        revision: int,
        text: str,
        expected_epoch: str,
    ) -> CodeRun:
        _validate_id(operation_id)
        if not text.strip() or len(text) > 1_000_000:
            raise CodeValidationError(
                "Enter a message of at most 1,000,000 characters."
            )
        owner = await self._owner(session_id)
        async with owner.lock:
            previous = await self._store.get_run(session_id, operation_id)
            if previous:
                if previous.session_id != session_id or previous.text != text:
                    raise CodeConflictError(
                        "This Send ID was already used for a different message."
                    )
                return previous
            if expected_epoch != self.epoch:
                raise CodeConflictError(
                    "FCC restarted. Your draft has been kept; send it when you are ready."
                )
            self._editable(owner, revision)
            if owner.busy or owner.pending:
                raise CodeConflictError(
                    "This session is busy. Your draft has been kept."
                )
            selection = self._harness.prepare(
                owner.session.model, owner.session.reasoning_effort, owner.session.mode
            )
            run = CodeRun(
                id=operation_id,
                session_id=session_id,
                text=text,
                model=selection.model,
                reasoning_effort=selection.reasoning_effort,
                mode=selection.mode,
            )
            title = (
                " ".join(text.split())[:80]
                if owner.session.auto_title
                else owner.session.title
            )
            session = _revision(
                owner.session, title=title, auto_title=False, error=None
            )
            item = CodeItem(
                id=operation_id,
                session_id=session_id,
                sequence=owner.sequence + 1,
                run_id=operation_id,
                kind="user",
                text=text,
                complete=True,
            )
            session, run = await self._store.admit_run(session, run, item, revision)
            owner.session, owner.run = session, run
            owner.runs[run.id] = run
            owner.items[item.id] = item
            owner.sequence = item.sequence
            owner.finished = asyncio.Event()
            self._publish(owner, "run.updated", items=[item.model_dump(mode="json")])
            owner.job = self._job(self._work(owner, run.id, selection))
            return run

    async def stop(self, session_id: str, operation_id: str) -> CodeRun:
        return await self._command(self._stop(session_id, operation_id))

    async def _stop(
        self, session_id: str, operation_id: str, *, shutting_down: bool = False
    ) -> CodeRun:
        _validate_id(operation_id)
        owner = await self._owner(session_id)
        async with owner.lock:
            run = await self._store.get_run(session_id, operation_id)
            if run is None or run.session_id != session_id:
                raise CodeNotFoundError("Code turn not found.")
            if owner.run is None or owner.run.id != operation_id or not owner.busy:
                return run
            if not shutting_down:
                self._require_available()
            if not owner.run.stop_requested:
                run = owner.run.model_copy(
                    update={"stop_requested": True, "status": "stopping"}
                )
                await self._persist(owner, owner.session, run=run)
                owner.run = run
                self._publish(owner, "run.updated")
            self._schedule_interrupt(owner)
            return owner.run

    async def answer(
        self, session_id: str, prompt_id: str, response_id: str, answer: JsonObject
    ) -> CodePrompt:
        return await self._command(
            self._answer(session_id, prompt_id, response_id, answer)
        )

    async def _answer(
        self, session_id: str, prompt_id: str, response_id: str, answer: JsonObject
    ) -> CodePrompt:
        _validate_id(response_id)
        owner = await self._owner(session_id)
        async with owner.lock:
            self._require_available()
            prompt = owner.prompts.get(prompt_id)
            if prompt is None:
                raise CodeNotFoundError("This prompt no longer exists.")
            if prompt.response_id == response_id:
                return prompt
            connection = owner.connection
            if (
                prompt.status != "pending"
                or connection is None
                or prompt.generation != owner.generation
            ):
                raise CodeConflictError(
                    "This prompt was already answered or is no longer active."
                )
            response = connection.prepare_answer(prompt.request_id, answer)
            claimed = await self._store.claim_prompt(
                session_id, prompt_id, response_id, connection.generation
            )
            owner.prompts[prompt_id] = claimed
            self._publish_prompt(owner, claimed)
            self._job(self._deliver_answer(owner, claimed, response))
            return claimed

    async def delete_session(
        self, session_id: str, revision: int
    ) -> CodeSession | None:
        return await self._command(self._delete(session_id, revision))

    async def _delete(self, session_id: str, revision: int) -> CodeSession | None:
        self._require_available()
        if await self._store.is_deleted(session_id):
            return None
        owner = await self._owner(session_id)
        async with owner.lock:
            self._check_owner(owner)
            if owner.session.status != "ready":
                if owner.session.status == "delete_uncertain" and (
                    owner.job is None or owner.job.done()
                ):
                    if revision != owner.session.revision:
                        raise CodeConflictError(
                            "This session changed. Refresh its state and try again."
                        )
                    owner.finished = asyncio.Event()
                    owner.job = self._job(self._delete_native(owner, reconcile=True))
                return owner.session
            self._editable(owner, revision)
            if owner.busy or owner.pending:
                raise CodeConflictError(
                    "Stop the session and finish its prompts before deleting it."
                )
            session = _revision(owner.session, status="deleting", error=None)
            await self._persist(owner, session)
            owner.session = session
            owner.finished = asyncio.Event()
            self._publish(owner, "session.updated")
            owner.job = self._job(self._delete_native(owner))
            return session

    async def wait_idle(self, session_id: str) -> None:
        """Wait for the currently owned operation, including its durable settlement."""
        owner = self._owners.get(session_id)
        if owner is not None:
            await owner.finished.wait()

    async def _work(
        self, owner: _Owner, run_id: str, selection: HarnessSelection
    ) -> None:
        finished = owner.finished
        try:
            connection = owner.connection
            if connection is not None and not connection.supports(selection):
                await self._close_connection(owner)
                connection = None
            if connection is None:
                connection = await selection.open(
                    owner.session.cwd, lambda event: self._event(owner, event)
                )
                owner.connection = connection
                owner.generation = connection.generation
            thread_id = owner.session.native_thread_id
            if thread_id is None:
                async with owner.lock:
                    if owner.run and owner.run.stop_requested:
                        await self._finish_locked(owner, "interrupted")
                        return
                native = await connection.create_thread()
            elif owner.loaded_thread_id != thread_id:
                try:
                    native = await connection.resume_thread(thread_id)
                except NativeHistoryMissing:
                    if owner.session.native_may_have_input:
                        raise
                    native = await connection.create_thread()
            else:
                native = None
            async with owner.lock:
                if native is not None:
                    session = owner.session.model_copy(
                        update={
                            "native_thread_id": native.id,
                            "native_permission_defaults": owner.session.native_permission_defaults
                            if owner.session.native_permission_defaults is not None
                            else native.permission_defaults,
                        }
                    )
                    await self._persist(owner, session)
                    owner.session = session
                    owner.loaded_thread_id = native.id
                    await self._recover_locked(owner, native)
                run = owner.run
                if run is None or run.id != run_id or not owner.busy:
                    return
                if run.stop_requested or not self._accepting:
                    await self._finish_locked(owner, "interrupted")
                    return
                defaults = owner.session.native_permission_defaults
                if defaults is None:
                    raise CodeUnavailableError(
                        "The harness did not return its permission settings. Your input was not sent."
                    )
                run = run.model_copy(update={"submission_started": True})
                session = owner.session.model_copy(
                    update={"native_may_have_input": True}
                )
                await self._persist(owner, session, run=run)
                owner.session, owner.run = session, run
            turn_id = await connection.start_turn(run.text, selection, run.id, defaults)
            async with owner.lock:
                if owner.run is None or owner.run.id != run_id or not owner.busy:
                    return
                if owner.run.native_turn_id not in {None, turn_id}:
                    raise CodeUnavailableError(
                        "Codex returned a different turn identity."
                    )
                updated = owner.run.model_copy(
                    update={
                        "native_turn_id": turn_id,
                        "status": "stopping" if owner.run.stop_requested else "running",
                    }
                )
                await self._persist(owner, owner.session, run=updated)
                owner.run = updated
                owner.known_turns.add(turn_id)
                self._publish(owner, "run.updated")
                self._schedule_interrupt(owner)
            await finished.wait()
        except asyncio.CancelledError:
            await self._fail(
                owner,
                run_id,
                "Code session initialization was interrupted.",
                interrupted=True,
            )
            raise
        except Exception as exc:
            await self._fail(owner, run_id, _error_message(exc))
        finally:
            if not self._accepting or owner.storage_failed:
                await self._close_connection(owner)

    async def _recover_locked(self, owner: _Owner, native: NativeThread) -> None:
        mapped: dict[str, CodeRun] = {}
        for turn in native.turns:
            identities = {
                item.client_id
                for item in turn.items
                if item.kind == "user" and item.client_id
            }
            candidates = [
                run
                for run in owner.runs.values()
                if run.native_turn_id == turn.id or run.id in identities
            ]
            if len(candidates) > 1 or any(
                identity not in owner.runs for identity in identities
            ):
                raise CodeUnavailableError(
                    "Codex history contains conflicting turn identities. Saved history was retained."
                )
            if candidates:
                run = candidates[0]
                if (
                    not run.submission_started
                    or run.native_turn_id not in {None, turn.id}
                    or run.id in {value.id for value in mapped.values()}
                ):
                    raise CodeUnavailableError(
                        "Codex history could not be matched to its saved turns."
                    )
                mapped[turn.id] = run
        unmatched = [turn for turn in native.turns if turn.id not in mapped]
        pending = [
            run
            for run in owner.runs.values()
            if run.submission_started
            and run.native_turn_id is None
            and run.id not in {value.id for value in mapped.values()}
        ]
        if unmatched:
            if len(unmatched) != 1 or len(pending) != 1:
                raise CodeUnavailableError(
                    "Codex history could not be matched to its saved turns. No input was resent."
                )
            mapped[unmatched[0].id] = pending[0]
        for turn in native.turns:
            previous = mapped[turn.id]
            # Restart/connection loss has already settled interrupted work. Recover
            # source content and missing diagnostics without rewriting that outcome.
            run = previous.model_copy(
                update={
                    "native_turn_id": turn.id,
                    "error": previous.error or turn.error,
                    "error_details": previous.error_details or turn.error_details,
                }
            )
            for item in turn.items:
                self._update_item(owner, item, run, historical=True)
            items = tuple(owner.items[item_id] for item_id in owner.dirty)
            await self._persist(owner, owner.session, run=run, items=items)
            owner.dirty.clear()
            owner.dirty_characters = 0
            owner.known_turns.add(turn.id)
            if owner.run and owner.run.id == run.id:
                owner.run = run
            self._publish(
                owner,
                "run.updated",
                runs=[run.model_dump(mode="json")],
                items=[item.model_dump(mode="json") for item in items],
            )

    def _schedule_interrupt(self, owner: _Owner) -> None:
        run, connection = owner.run, owner.connection
        if (
            run is None
            or connection is None
            or not run.stop_requested
            or not run.native_turn_id
            or not owner.busy
            or owner.interrupt_for == run.id
        ):
            return
        owner.interrupt_for = run.id
        owner.interrupt_task = self._job(
            self._interrupt(owner, connection, run, owner.finished)
        )

    async def _interrupt(
        self,
        owner: _Owner,
        connection: HarnessConnection,
        run: CodeRun,
        finished: asyncio.Event,
    ) -> None:
        assert run.native_turn_id is not None
        try:
            await connection.interrupt(run.native_turn_id)
            async with asyncio.timeout(5):
                await finished.wait()
            return
        except Exception:
            pass
        async with owner.lock:
            if (
                owner.connection is not connection
                or owner.run is None
                or owner.run.id != run.id
                or not owner.busy
            ):
                return
            self._detach_connection_locked(owner)
        await connection.close()
        async with owner.lock:
            if owner.run and owner.run.id == run.id and owner.busy:
                await self._finish_locked(
                    owner,
                    "interrupted",
                    "Codex did not stop normally; its session process was closed.",
                )

    async def _deliver_answer(
        self, owner: _Owner, prompt: CodePrompt, response: JsonObject
    ) -> None:
        connection = owner.connection
        if connection is None:
            return
        try:
            async with owner.lock:
                current = owner.prompts.get(prompt.id)
                if (
                    current is None
                    or current.status != "answering"
                    or owner.generation != prompt.generation
                ):
                    return
            await connection.respond(prompt.request_id, response)
            async with owner.lock:
                current = owner.prompts.get(prompt.id)
                if (
                    current is not None
                    and current.status == "answering"
                    and owner.generation == prompt.generation
                ):
                    resolved = current.model_copy(update={"status": "resolved"})
                    await self._persist(owner, owner.session, prompts=(resolved,))
                    owner.prompts[prompt.id] = resolved
                    self._publish_prompt(owner, resolved)
        except CodeConflictError:
            async with owner.lock:
                current = owner.prompts.get(prompt.id)
                if current and current.status == "answering":
                    expired = current.model_copy(update={"status": "expired"})
                    await self._persist(owner, owner.session, prompts=(expired,))
                    owner.prompts[prompt.id] = expired
                    self._publish_prompt(owner, expired)
        except Exception as exc:
            if owner.run and owner.busy:
                await self._fail(
                    owner,
                    owner.run.id,
                    "The native answer could not be confirmed. It was not resent. "
                    + _error_message(exc),
                )
            else:
                await self._close_connection(owner)

    async def _event(self, owner: _Owner, event: HarnessEvent) -> None:
        try:
            async with owner.lock:
                if (
                    owner.deleted
                    or event.generation != owner.generation
                    or event.thread_id not in {None, owner.session.native_thread_id}
                ):
                    return
                if owner.session.status != "ready":
                    return
                run = owner.run
                matches = (
                    run is not None
                    and owner.busy
                    and run.submission_started
                    and event.turn_id is not None
                    and (
                        run.native_turn_id == event.turn_id
                        or (
                            run.native_turn_id is None
                            and event.kind == "turn_started"
                            and event.turn_id not in owner.known_turns
                        )
                    )
                )
                if (
                    event.kind in {"turn_started", "turn_completed"}
                    and matches
                    and run is not None
                ):
                    updated = run.model_copy(
                        update={
                            "native_turn_id": event.turn_id,
                            "status": "stopping" if run.stop_requested else "running",
                        }
                    )
                    await self._persist(owner, owner.session, run=updated)
                    owner.run = updated
                    if event.turn_id:
                        owner.known_turns.add(event.turn_id)
                    if event.kind == "turn_completed":
                        await self._finish_locked(
                            owner, event.status, event.message, event.error_details
                        )
                    else:
                        self._publish(owner, "run.updated")
                        self._schedule_interrupt(owner)
                elif (
                    event.kind == "item"
                    and event.item is not None
                    and event.item.kind == "subagent_auto_review"
                    and run is not None
                ):
                    existing = self._native_item(owner, event.item)
                    if (
                        existing is not None
                        and existing.complete
                        and not event.item.complete
                    ):
                        return
                    target = owner.runs[existing.run_id] if existing else run
                    item = self._update_item(owner, event.item, target)
                    previous_generation = owner.review_generations.get(item.id)
                    if item.complete:
                        owner.review_generations.pop(item.id, None)
                    else:
                        owner.review_generations[item.id] = event.generation
                    if previous_generation != owner.review_generations.get(item.id):
                        # Reannounce the saved row when its liveness changes.
                        owner.dirty.add(item.id)
                    await self._flush_locked(owner)
                elif (
                    event.kind == "item"
                    and event.item is not None
                    and matches
                    and run is not None
                ):
                    self._update_item(owner, event.item, run)
                    if event.item.complete or owner.dirty_characters >= 4096:
                        await self._flush_locked(owner)
                    elif owner.flush_task is None or owner.flush_task.done():
                        owner.flush_task = self._job(self._flush_later(owner))
                elif event.kind == "prompt" and event.prompt is not None:
                    request = event.prompt
                    if request.turn_id is not None and not matches:
                        return
                    if any(
                        prompt.generation == event.generation
                        and prompt.request_id == request.request_id
                        for prompt in owner.prompts.values()
                    ):
                        return
                    await self._flush_locked(owner)
                    prompt = CodePrompt(
                        id=str(uuid.uuid4()),
                        session_id=owner.session.id,
                        generation=event.generation,
                        request_id=request.request_id,
                        native_turn_id=request.turn_id,
                        native_item_id=request.item_id,
                        kind=request.kind,
                        form=request.form,
                        raw=request.raw,
                    )
                    if run is None:
                        raise CodeUnavailableError(
                            "Codex returned a prompt without a saved turn."
                        )
                    item = CodeItem(
                        id=prompt.id,
                        session_id=owner.session.id,
                        run_id=run.id,
                        sequence=owner.sequence + 1,
                        kind="prompt",
                        complete=True,
                    )
                    await self._persist(
                        owner, owner.session, items=(item,), prompts=(prompt,)
                    )
                    owner.items[item.id] = item
                    owner.sequence = item.sequence
                    owner.prompts[prompt.id] = prompt
                    self._publish_prompt(owner, prompt)
                elif event.kind == "resolved":
                    for prompt in tuple(owner.prompts.values()):
                        if (
                            prompt.generation == event.generation
                            and prompt.request_id == event.request_id
                            and prompt.status in {"pending", "answering"}
                        ):
                            resolved = prompt.model_copy(update={"status": "resolved"})
                            await self._persist(
                                owner, owner.session, prompts=(resolved,)
                            )
                            owner.prompts[prompt.id] = resolved
                            self._publish_prompt(owner, resolved)
                elif event.kind == "error" and matches:
                    self._publish(
                        owner,
                        "run.notice",
                        message=event.message or "Codex reported an error.",
                        will_retry=event.will_retry,
                    )
                elif event.kind == "notice" and event.message and run is not None:
                    await self._flush_locked(owner)
                    item = CodeItem(
                        id=str(uuid.uuid4()),
                        session_id=owner.session.id,
                        sequence=owner.sequence + 1,
                        run_id=run.id,
                        kind="notice",
                        title="Notice",
                        text=event.message,
                        complete=True,
                        raw=event.raw,
                    )
                    await self._persist(owner, owner.session, items=(item,))
                    owner.items[item.id] = item
                    owner.sequence = item.sequence
                    self._publish(
                        owner, "item.updated", item=item.model_dump(mode="json")
                    )
                elif event.kind == "closed":
                    self._detach_connection_locked(owner)
                    if owner.busy:
                        await self._finish_locked(
                            owner,
                            "interrupted" if run and run.stop_requested else "failed",
                            event.message or "The Codex process ended.",
                        )
                    await self._expire_prompts_locked(owner)
        except Exception as exc:
            if owner.run and owner.busy:
                # Never wait for process teardown from its event dispatcher.
                self._job(self._fail(owner, owner.run.id, _error_message(exc)))
            else:
                logger.warning(
                    "Code event could not be saved: exc_type={}", type(exc).__name__
                )

    @staticmethod
    def _native_item(owner: _Owner, update: ItemUpdate) -> CodeItem | None:
        return next(
            (
                item
                for item in owner.items.values()
                if item.native_turn_id == update.turn_id
                and item.native_item_id == update.item_id
            ),
            None,
        )

    def _update_item(
        self,
        owner: _Owner,
        update: ItemUpdate,
        run: CodeRun,
        *,
        historical: bool = False,
    ) -> CodeItem:
        if update.kind != "subagent_auto_review":
            owner.known_turns.add(update.turn_id)
        existing = self._native_item(owner, update)
        if existing is None and update.kind == "user":
            if update.client_id is not None and update.client_id != run.id:
                raise CodeUnavailableError(
                    "Codex returned a different message identity."
                )
            existing = owner.items.get(run.id)
        if existing is not None and existing.run_id != run.id:
            raise CodeUnavailableError(
                "Codex returned output belonging to another turn."
            )
        if existing is None:
            owner.sequence += 1
        preserve = bool(historical and existing and existing.complete)
        raw = _merge_source(existing.raw if existing else {}, update.raw, preserve)
        text = update.text
        detail = update.detail
        if historical and existing:
            text = (existing.text or text) if preserve else (text or existing.text)
            detail = (
                (existing.detail or detail) if preserve else (detail or existing.detail)
            )
        item = CodeItem(
            id=existing.id if existing else str(uuid.uuid4()),
            session_id=owner.session.id,
            sequence=existing.sequence if existing else owner.sequence,
            run_id=run.id,
            native_turn_id=update.turn_id,
            native_item_id=update.item_id,
            kind=update.kind,
            title=update.title or (existing.title if existing else ""),
            text=text,
            detail=detail,
            raw=raw,
            complete=update.complete or bool(existing and existing.complete),
        )
        if item == existing:
            return existing
        owner.items[item.id] = item
        owner.dirty.add(item.id)
        owner.dirty_characters += abs(
            len(item.text) - (len(existing.text) if existing else 0)
        ) + abs(len(item.detail) - (len(existing.detail) if existing else 0))
        owner.version += 1
        return item

    async def _flush_later(self, owner: _Owner) -> None:
        await asyncio.sleep(0.25)
        try:
            async with owner.lock:
                if not owner.deleted and not owner.storage_failed:
                    await self._flush_locked(owner)
        except Exception as exc:
            if owner.run:
                await self._fail(owner, owner.run.id, _error_message(exc))

    async def _flush_locked(self, owner: _Owner) -> None:
        if not owner.dirty:
            return
        items = tuple(owner.items[item_id] for item_id in owner.dirty)
        await self._persist(owner, owner.session, items=items)
        owner.dirty.clear()
        owner.dirty_characters = 0
        for item in sorted(items, key=lambda value: value.sequence):
            self._publish(
                owner,
                "item.updated",
                item=item.model_dump(mode="json"),
                runs=[owner.runs[item.run_id].model_dump(mode="json")],
            )

    async def _finish_locked(
        self,
        owner: _Owner,
        status: RunStatus,
        message: str | None = None,
        error_details: JsonObject | None = None,
    ) -> None:
        if owner.run is None or not owner.busy:
            return
        run = owner.run.model_copy(
            update={
                "status": status,
                "error": message,
                "error_details": error_details or {},
                "finished_at": now_ms(),
            }
        )
        session = _revision(owner.session, error=None)
        prompts = tuple(
            prompt.model_copy(update={"status": "expired"})
            for prompt in owner.prompts.values()
            if prompt.status in {"pending", "answering"}
            and prompt.native_turn_id is not None
            and prompt.native_turn_id == run.native_turn_id
        )
        items = tuple(owner.items[item_id] for item_id in owner.dirty)
        await self._persist(owner, session, run=run, items=items, prompts=prompts)
        owner.session, owner.run = session, run
        owner.prompts.update((prompt.id, prompt) for prompt in prompts)
        owner.dirty.clear()
        owner.dirty_characters = 0
        self._publish(
            owner,
            "run.updated",
            items=[item.model_dump(mode="json") for item in items],
            prompts=[prompt.model_dump(mode="json") for prompt in prompts],
        )
        owner.finished.set()

    async def _expire_prompts_locked(self, owner: _Owner) -> None:
        prompts = tuple(
            prompt.model_copy(update={"status": "expired"})
            for prompt in owner.prompts.values()
            if prompt.status in {"pending", "answering"}
        )
        if not prompts:
            return
        await self._persist(owner, owner.session, prompts=prompts)
        for prompt in prompts:
            owner.prompts[prompt.id] = prompt
            self._publish_prompt(owner, prompt)

    async def _persist(
        self,
        owner: _Owner,
        session: CodeSession,
        *,
        run: CodeRun | None = None,
        items: Sequence[CodeItem] = (),
        prompts: Sequence[CodePrompt] = (),
    ) -> None:
        if owner.storage_failed:
            raise CodeUnavailableError(
                "Code storage failed. Restart FCC to restore saved history."
            )
        try:
            await self._store.save_progress(
                session, owner.session.revision, run=run, items=items, prompts=prompts
            )
            if run is not None:
                owner.runs[run.id] = run
        except CodeConflictError, CodeNotFoundError:
            raise
        except Exception:
            owner.storage_failed = True
            if owner.failure_task is None:
                owner.failure_task = self._job(self._halt_storage(owner))
            raise

    async def _halt_storage(self, owner: _Owner) -> None:
        await self._close_connection(owner)
        self._storage_failure(owner)

    def _detach_connection_locked(self, owner: _Owner) -> HarnessConnection | None:
        connection = owner.connection
        owner.connection = None
        owner.generation = None
        owner.loaded_thread_id = None
        if owner.review_generations:
            owner.review_generations.clear()
            if not owner.deleted:
                self._publish(owner, "session.updated")
        return connection

    async def _close_connection(self, owner: _Owner) -> None:
        async with owner.lock:
            connection = self._detach_connection_locked(owner)
        if connection is not None:
            await connection.close()
        async with owner.lock:
            if not owner.deleted and not owner.storage_failed:
                try:
                    if (
                        connection
                        and connection.thread_id
                        and owner.session.native_thread_id is None
                    ):
                        session = owner.session.model_copy(
                            update={"native_thread_id": connection.thread_id}
                        )
                        await self._persist(owner, session)
                        owner.session = session
                    await self._expire_prompts_locked(owner)
                except Exception:
                    self._storage_failure(owner)

    async def _fail(
        self, owner: _Owner, run_id: str, message: str, *, interrupted: bool = False
    ) -> None:
        async with owner.lock:
            if owner.run is None or owner.run.id != run_id or not owner.busy:
                return
        try:
            await self._close_connection(owner)
            async with owner.lock:
                if owner.run and owner.run.id == run_id and not owner.storage_failed:
                    status = (
                        "interrupted"
                        if interrupted or owner.run.stop_requested
                        else "failed"
                    )
                    await self._finish_locked(owner, status, message)
        except Exception:
            self._storage_failure(owner)

    def _storage_failure(self, owner: _Owner) -> None:
        owner.storage_failed = True
        owner.finished.set()
        self._publish(
            owner,
            "session.notice",
            message="Code storage failed. This session was stopped; restart FCC to restore saved history.",
        )

    async def _delete_native(self, owner: _Owner, *, reconcile: bool = False) -> None:
        thread_id = owner.session.native_thread_id
        native_complete = False
        try:
            if thread_id is not None:
                connection = owner.connection
                if connection is None:
                    connection = await self._harness.open_history(
                        owner.session.cwd, lambda event: self._event(owner, event)
                    )
                    owner.connection, owner.generation = (
                        connection,
                        connection.generation,
                    )
                try:
                    if reconcile:
                        await connection.read_thread(thread_id)
                        raise CodeConflictError(
                            "The native conversation still exists. You can retry deletion."
                        )
                    await connection.delete_thread(thread_id)
                except NativeHistoryMissing:
                    pass
            native_complete = True
            await self._close_connection(owner)
            async with owner.lock:
                await self._store.delete(owner.session.id)
                owner.deleted = True
                owner.items.clear()
                owner.prompts.clear()
                owner.run = None
                owner.runs.clear()
                self._publish(owner, "session.deleted")
            async with self._load_lock:
                if self._owners.get(owner.session.id) is owner:
                    del self._owners[owner.session.id]
        except Exception as exc:
            await self._close_connection(owner)
            status = (
                "ready" if isinstance(exc, CodeConflictError) else "delete_uncertain"
            )
            async with owner.lock:
                session = _revision(
                    owner.session, status=status, error=_error_message(exc)
                )
                try:
                    await self._persist(owner, session)
                except Exception:
                    self._storage_failure(owner)
                    return
                owner.session = session
                self._publish(owner, "session.updated")
            if status == "delete_uncertain" and not reconcile and not native_complete:
                await self._delete_native(owner, reconcile=True)
        finally:
            owner.finished.set()

    def _check_owner(self, owner: _Owner) -> None:
        if owner.deleted:
            raise CodeNotFoundError("Code session was deleted.")
        if owner.storage_failed:
            raise CodeUnavailableError(
                "Code storage failed. Restart FCC to restore saved history."
            )

    def _editable(self, owner: _Owner, revision: int) -> None:
        self._require_available()
        self._check_owner(owner)
        if owner.session.status != "ready" or owner.session.revision != revision:
            raise CodeConflictError(
                "This session changed. Refresh its state and try again."
            )

    def _summary(self, owner: _Owner) -> JsonObject:
        return {
            "session_id": owner.session.id,
            "session": owner.session.model_dump(mode="json"),
            "run": owner.run.model_dump(mode="json") if owner.run else None,
            "active_review_ids": list(owner.active_review_ids),
            "version": owner.version,
            "epoch": self.epoch,
        }

    def _publish(self, owner: _Owner, event: str, **data: JsonValue) -> None:
        owner.version += 1
        payload = self._summary(owner)
        for key, value in data.items():
            payload[key] = value
        self._events.publish(event, payload)

    def _publish_prompt(self, owner: _Owner, prompt: CodePrompt) -> None:
        item = owner.items[prompt.id]
        self._publish(
            owner,
            "prompt.updated",
            prompt=prompt.model_dump(mode="json"),
            item=item.model_dump(mode="json"),
            runs=[owner.runs[item.run_id].model_dump(mode="json")],
        )


def _validate_id(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
        if str(parsed) != value or parsed.version != 4:
            raise ValueError
    except ValueError, AttributeError:
        raise CodeValidationError("Invalid session or command ID.") from None


def _merge_source(
    previous: JsonObject, current: JsonObject, preserve: bool
) -> JsonObject:
    merged = dict(previous)
    for key, value in current.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_source(existing, value, preserve)
        elif not preserve or existing in (None, "", [], {}):
            merged[key] = value
    return merged


def _revision(session: CodeSession, **updates: object) -> CodeSession:
    return session.model_copy(
        update={"revision": session.revision + 1, "updated_at": now_ms(), **updates}
    )


def _error_message(error: Exception) -> str:
    if isinstance(
        error,
        CodeUnavailableError
        | CodeValidationError
        | CodeConflictError
        | CodeNotFoundError,
    ):
        return str(error)
    return f"The Code session could not continue ({type(error).__name__})."
