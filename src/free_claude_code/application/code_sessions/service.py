"""Server-owned coding sessions, independent of HTTP and browser lifetimes."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.readiness import InitializationWait
from free_claude_code.application.session_events import (
    EventPublisher,
    EventSubscription,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .models import (
    CodeCatalog,
    CodeConflictError,
    CodeDetail,
    CodeHarness,
    CodeNotFoundError,
    CodePage,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
    CodeValidationError,
    HarnessCapabilities,
    HarnessEvent,
    HarnessId,
    ItemUpdate,
    NativeHistoryMissing,
    NativeThread,
    RunStatus,
)
from .ports import CodeStore, HarnessConnection, HarnessFactory, HarnessSelection
from .state import SessionProgress, SessionState


@dataclass(slots=True)
class _SessionGate:
    on_idle: Callable[[], None]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0

    async def __aenter__(self) -> _SessionGate:
        self.users += 1
        try:
            await self.lock.acquire()
        except BaseException:
            self.users -= 1
            self.on_idle()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        self.lock.release()
        self.users -= 1
        self.on_idle()


@dataclass(slots=True)
class _SessionRuntime:
    state: SessionState
    lock: _SessionGate
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    connection: HarnessConnection | None = None
    generation: str | None = None
    setup_task: asyncio.Task[None] | None = None
    capabilities: HarnessCapabilities | None = None
    job: asyncio.Task[None] | None = None
    flush_task: asyncio.Task[None] | None = None
    interrupt_task: asyncio.Task[None] | None = None
    interrupt_for: str | None = None
    loaded_thread_id: str | None = None
    storage_failed: bool = False
    failure_task: asyncio.Task[None] | None = None
    deleted: bool = False


class CodeService:
    def __init__(
        self, store: CodeStore, harnesses: Mapping[HarnessId, HarnessFactory]
    ) -> None:
        self._store = store
        self._harnesses = dict(harnesses)
        self._owners: dict[str, _SessionRuntime] = {}
        self._gates: dict[str, _SessionGate] = {}
        self._version = 0
        self._events = EventPublisher()
        self.epoch = str(uuid.uuid4())
        self._commands: set[asyncio.Task] = set()
        self._jobs: set[asyncio.Task] = set()
        self._accepting = False
        self._started = False
        self._stopping = False
        self._start_task: asyncio.Task[None] | None = None
        self._storage_state = "starting"
        self._message: str | None = "Code sessions is starting."
        self._close_task: asyncio.Task[None] | None = None

    def _ensure_start(self) -> asyncio.Task[None]:
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start())
        return self._start_task

    async def start(self) -> None:
        await asyncio.shield(self._ensure_start())

    def storage_status(self) -> JsonObject:
        return {"state": self._storage_state, "message": self._message}

    async def _wait_for_store(self) -> None:
        if self._started:
            return
        if self._stopping:
            raise CodeUnavailableError("Code sessions is stopping.")
        try:
            await InitializationWait().wait(self._ensure_start())
        except ApplicationUnavailableError as exc:
            raise CodeUnavailableError(exc.message) from None
        self._require_available()

    async def _start(self) -> None:
        if self._started:
            return
        try:
            await self._store.start()
            pending = await self._store.pending_deletions()
            self._started = True
        except Exception as exc:
            await self._store.close()
            self._started = False
            self._message = _error_message(exc)
            self._storage_state = "failed"
            return
        except BaseException:
            await self._store.close()
            raise
        if self._stopping:
            return
        self._accepting = True
        self._storage_state = "ready"
        self._message = None
        for session in pending:
            owner = await self._owner(session.id)
            owner.finished = asyncio.Event()
            owner.job = self._job(self._delete_native(owner, reconcile=True))

    def availability(self) -> tuple[bool, str | None]:
        if not self._accepting:
            return False, self._message
        return True, None

    def _factory(self, harness: HarnessId) -> HarnessFactory:
        factory = self._harnesses.get(harness)
        if factory is None:
            raise CodeValidationError("Choose an available harness.")
        return factory

    def _require_harness(self, harness: HarnessId) -> HarnessFactory:
        self._require_available()
        factory = self._factory(harness)
        available, message = factory.availability()
        if not available:
            raise CodeUnavailableError(message or f"{factory.name} is unavailable.")
        return factory

    def catalog(self, harness: HarnessId = "codex") -> CodeCatalog:
        return self._factory(harness).catalog()

    def harnesses(self) -> tuple[CodeHarness, ...]:
        result = []
        for factory in self._harnesses.values():
            available, message = factory.availability()
            catalog = factory.catalog()
            result.append(
                CodeHarness(
                    id=factory.id,
                    name=factory.name,
                    available=available,
                    message=message,
                    prepare_on_open=factory.prepare_on_open,
                    modes=factory.modes,
                    default_model=catalog.default_model,
                    models=catalog.models,
                )
            )
        return tuple(result)

    def begin_shutdown(self) -> None:
        self._stopping = True
        self._storage_state = "stopping"
        self._accepting = False
        self._message = "Code sessions is stopping."
        self._events.disconnect_subscribers()

    async def close(self) -> None:
        self.begin_shutdown()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="fcc-code-close")
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._start_task is not None:
            if not self._start_task.done():
                self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        owners = tuple(self._owners.values())
        for owner in owners:
            if owner.setup_task is not None and not owner.setup_task.done():
                owner.setup_task.cancel()
        if self._commands:
            await asyncio.gather(*tuple(self._commands), return_exceptions=True)
        owners = tuple(self._owners.values())
        for owner in owners:
            if (
                owner.state.busy
                and owner.state.run is not None
                and not owner.storage_failed
            ):
                try:
                    await self._stop(
                        owner.state.session.id, owner.state.run.id, shutting_down=True
                    )
                except Exception:
                    await self._close_connection(owner)
                    self._storage_failure(owner)
        jobs = [owner.job for owner in owners if owner.job is not None]
        if jobs:
            await asyncio.wait(jobs, timeout=5)
        await asyncio.gather(*(self._close_connection(owner) for owner in owners))
        for owner in owners:
            async with self._locked(owner):
                if owner.state.busy and not owner.storage_failed:
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

    def _gate(self, session_id: str) -> _SessionGate:
        if session_id not in self._gates:
            self._gates[session_id] = _SessionGate(
                lambda: self._discard_gate(session_id)
            )
        return self._gates[session_id]

    def _discard_gate(self, session_id: str) -> None:
        gate = self._gates.get(session_id)
        if gate is not None and not gate.users and session_id not in self._owners:
            del self._gates[session_id]

    @asynccontextmanager
    async def _locked(self, owner: _SessionRuntime) -> AsyncIterator[None]:
        async with owner.lock:
            try:
                yield
            finally:
                if not owner.storage_failed:
                    owner.state.prune()

    async def _owner(self, session_id: str) -> _SessionRuntime:
        await self._wait_for_store()
        _validate_id(session_id)
        async with self._gate(session_id) as gate:
            owner = self._owners.get(session_id)
            if owner is None:
                seed = await self._store.execution_seed(session_id)
                owner = _SessionRuntime(
                    SessionState(
                        seed.session,
                        seed.run,
                        seed.items,
                        seed.prompts,
                        seed.runs,
                        sequence=seed.sequence,
                    ),
                    gate,
                )
                if not owner.state.busy:
                    owner.finished.set()
                self._owners[session_id] = owner
        if owner.deleted:
            raise CodeNotFoundError("Code session was deleted.")
        return owner

    async def create_session(
        self, session_id: str, cwd: str, harness: HarnessId = "codex"
    ) -> CodeSession:
        return await self._command(self._create(session_id, cwd, harness))

    async def _create(
        self, session_id: str, cwd: str, harness: HarnessId
    ) -> CodeSession:
        await self._wait_for_store()
        self._require_harness(harness)
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
                id=session_id,
                cwd=str(folder),
                model=self.catalog(harness).default_model,
                harness=harness,
            )
        )
        owner = await self._owner(session.id)
        async with self._locked(owner):
            self._publish(owner, "session.updated")
        return session

    async def list_sessions(
        self, cursor: tuple[int, str] | None = None, limit: int = 25, query: str = ""
    ) -> CodePage:
        await self._wait_for_store()
        return await self._store.list_sessions(cursor, max(1, min(limit, 25)), query)

    async def subscribe(self) -> tuple[EventSubscription, JsonObject]:
        await self._wait_for_store()
        self._require_available()
        subscription = self._events.subscribe()
        summaries = [
            self._summary(owner)
            for owner in tuple(self._owners.values())
            if not owner.deleted
            and (
                owner.state.busy
                or owner.state.pending
                or owner.state.active_review_ids(owner.generation)
                or owner.state.session.status != "ready"
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
        await self._wait_for_store()
        _validate_id(session_id)
        async with self._gate(session_id):
            owner = self._owners.get(session_id)
            active_reviews: tuple[str, ...] = ()
            if owner is not None:
                self._check_owner(owner)
                await self._flush_locked(owner)
                active_reviews = owner.state.active_review_ids(owner.generation)
                owner.state.prune()
            snapshot = await self._store.read_history(
                session_id,
                before,
                (*include_item_ids, *active_reviews),
            )
            return CodeDetail(
                snapshot.session,
                snapshot.run,
                snapshot.items,
                snapshot.prompts,
                self.epoch,
                self._version,
                self.cursor,
                snapshot.next_before,
                snapshot.runs,
                snapshot.active_prompt_ids,
                active_reviews,
                owner.capabilities if owner is not None else None,
            )

    async def update_settings(
        self, session_id: str, revision: int, changes: JsonObject
    ) -> CodeSession:
        return await self._command(self._update_settings(session_id, revision, changes))

    async def _update_settings(
        self, session_id: str, revision: int, changes: JsonObject
    ) -> CodeSession:
        SessionState.validate_settings_fields(changes)
        owner = await self._owner(session_id)
        async with self._locked(owner):
            self._editable(owner, revision)
            if (
                changes.keys() & {"model", "reasoning_effort", "mode"}
                and owner.setup_task is not None
                and not owner.setup_task.done()
            ):
                raise CodeConflictError(
                    "This session is initializing. Your draft has been kept."
                )
            updates = owner.state.settings_updates(changes)
            self._validate_mode(owner, changes)
            catalog = (
                self._session_catalog(owner)
                if changes.keys() & {"model", "reasoning_effort"}
                else None
            )
            session = await self._store.update_settings(
                owner.state.prepare_settings(updates, catalog), revision
            )
            owner.state.accept_session(session)
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
        if owner.setup_task is not None and not owner.setup_task.done():
            await asyncio.shield(owner.setup_task)
        async with self._locked(owner):
            previous = await self._store.get_run(session_id, operation_id)
            if previous:
                owner.state.check_receipt(previous, text)
                return previous
            if expected_epoch != self.epoch:
                raise CodeConflictError(
                    "FCC restarted. Your draft has been kept; send it when you are ready."
                )
            self._editable(owner, revision)
            owner.state.check_can_send()
            selected = owner.state.session
        factory = self._require_harness(selected.harness)
        selection = await factory.prepare(
            selected.model, selected.reasoning_effort, selected.mode
        )
        async with self._locked(owner):
            previous = await self._store.get_run(session_id, operation_id)
            if previous:
                owner.state.check_receipt(previous, text)
                return previous
            if expected_epoch != self.epoch:
                raise CodeConflictError(
                    "FCC restarted. Your draft has been kept; send it when you are ready."
                )
            self._editable(owner, revision)
            owner.state.check_can_send()
            session, run, item = owner.state.prepare_admission(
                operation_id,
                text,
                selection.model,
                selection.reasoning_effort,
                selection.mode,
            )
            session, run = await self._store.admit_run(session, run, item, revision)
            owner.state.accept_admission(session, run, item)
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
        async with self._locked(owner):
            receipt = await self._store.get_run(session_id, operation_id)
            if receipt is None or receipt.session_id != session_id:
                raise CodeNotFoundError("Code turn not found.")
            run = owner.state.active_run(operation_id)
            if run is None:
                return receipt
            if not shutting_down:
                self._require_available()
            progress = owner.state.request_stop()
            if progress is not None:
                await self._commit_progress(owner, progress)
                self._publish(owner, "run.updated")
            self._schedule_interrupt(owner)
            return owner.state.runs[run.id]

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
        async with self._locked(owner):
            self._require_available()
            connection = owner.connection
            if prompt_id not in owner.state.prompts:
                saved = await self._store.get_prompt(session_id, prompt_id)
                if saved is not None:
                    owner.state.accept_prompt(saved)
            prompt = owner.state.answerable_prompt(
                prompt_id,
                response_id,
                owner.generation if connection is not None else None,
            )
            if prompt.response_id == response_id:
                return prompt
            assert connection is not None
            response = connection.prepare_answer(prompt.request_id, answer)
            claimed = await self._store.claim_prompt(
                session_id, prompt_id, response_id, connection.generation
            )
            owner.state.accept_prompt(claimed)
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
        async with self._locked(owner):
            self._check_owner(owner)
            if owner.state.session.status != "ready":
                if owner.state.session.status == "delete_uncertain" and (
                    owner.job is None or owner.job.done()
                ):
                    owner.state.check_revision(revision)
                    owner.finished = asyncio.Event()
                    owner.job = self._job(self._delete_native(owner, reconcile=True))
                return owner.state.session
            self._editable(owner, revision)
            await self._commit_progress(owner, owner.state.prepare_delete())
            owner.finished = asyncio.Event()
            self._publish(owner, "session.updated")
            owner.job = self._job(self._delete_native(owner))
            return owner.state.session

    async def wait_idle(self, session_id: str) -> None:
        """Wait for the currently owned operation, including its durable settlement."""
        owner = self._owners.get(session_id)
        if owner is not None:
            await owner.finished.wait()

    def _session_catalog(self, owner: _SessionRuntime) -> CodeCatalog:
        session = owner.state.session
        catalog = self.catalog(session.harness)
        capabilities = owner.capabilities
        if capabilities is None or not self._factory(session.harness).prepare_on_open:
            return catalog
        return CodeCatalog(
            catalog.default_model,
            tuple(
                model.model_copy(
                    update={"reasoning_efforts": capabilities.reasoning_efforts}
                )
                if model.id == capabilities.model
                else model
                for model in catalog.models
            ),
        )

    def _validate_mode(self, owner: _SessionRuntime, changes: JsonObject) -> None:
        if "mode" not in changes:
            return
        modes = self._factory(owner.state.session.harness).modes
        if owner.capabilities is not None:
            modes = owner.capabilities.modes
        if changes["mode"] not in {option.id for option in modes}:
            raise CodeValidationError(
                "This permission mode is unavailable for this harness/model."
            )

    async def prepare_session(
        self, session_id: str, revision: int, *, expected_epoch: str
    ) -> CodeDetail:
        return await self._command(
            self._prepare_session(session_id, revision, expected_epoch)
        )

    async def _prepare_session(
        self, session_id: str, revision: int, epoch: str
    ) -> CodeDetail:
        owner = await self._owner(session_id)
        async with self._locked(owner):
            self._editable(owner, revision)
            if epoch != self.epoch:
                raise CodeConflictError(
                    "FCC restarted. Refresh the session before preparing it."
                )
            selected = owner.state.session
            busy = owner.state.busy or owner.state.pending
        if not busy:
            factory = self._require_harness(selected.harness)
            selection = await factory.prepare(
                selected.model, selected.reasoning_effort, selected.mode
            )
            await self._ensure_prepared(owner, selection, expected_revision=revision)
        return await self.get_detail(session_id)

    async def _ensure_prepared(
        self,
        owner: _SessionRuntime,
        selection: HarnessSelection,
        *,
        expected_revision: int | None = None,
    ) -> None:
        async with self._locked(owner):
            self._check_owner(owner)
            if expected_revision is not None:
                owner.state.check_revision(expected_revision)
            if not self._accepting or owner.state.session.status != "ready":
                raise CodeUnavailableError("This session is closing.")
            current = owner.state.session
            if (
                current.model != selection.model
                or current.mode != selection.mode
                or (
                    current.reasoning_effort is not None
                    and current.reasoning_effort != selection.reasoning_effort
                )
            ):
                raise CodeConflictError("The selection changed during preparation.")
            if (
                owner.connection is not None
                and owner.connection.supports(selection)
                and owner.loaded_thread_id is not None
            ):
                return
            task = owner.setup_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._prepare_connection(owner, selection), name="fcc-code-prepare"
                )
                owner.setup_task = task
        await asyncio.shield(task)

    async def _prepare_connection(
        self, owner: _SessionRuntime, selection: HarnessSelection
    ) -> None:
        try:
            async with asyncio.timeout(30):
                if owner.connection is not None and not owner.connection.supports(
                    selection
                ):
                    await self._close_connection(owner)
                if owner.connection is None:
                    connection = await selection.open(
                        owner.state.session.cwd, lambda event: self._event(owner, event)
                    )
                    async with self._locked(owner):
                        owner.connection = connection
                        owner.generation = connection.generation
                else:
                    connection = owner.connection
                thread_id = owner.state.session.native_thread_id
                if thread_id is None:
                    native = await connection.create_thread()
                elif owner.loaded_thread_id != thread_id:
                    runs = await self._store.runs(owner.state.session.id)
                    try:
                        native = await connection.resume_thread(
                            thread_id,
                            submitted_run_ids=frozenset(
                                run.id for run in runs if run.submission_started
                            ),
                        )
                    except NativeHistoryMissing:
                        if owner.state.session.native_may_have_input:
                            raise
                        native = await connection.create_thread()
                else:
                    native = None
                async with self._locked(owner):
                    if (
                        not self._accepting
                        or owner.deleted
                        or owner.state.session.status != "ready"
                    ):
                        raise CodeUnavailableError(
                            "This session closed during preparation."
                        )
                    if native is not None:
                        await self._commit_progress(
                            owner,
                            owner.state.attach_thread(
                                native.id, native.permission_defaults
                            ),
                        )
                        owner.loaded_thread_id = native.id
                        await self._recover_locked(owner, native)
                    owner.capabilities = connection.capabilities(selection)
                    self._publish(
                        owner,
                        "session.capabilities",
                        capabilities=owner.capabilities.model_dump(
                            mode="json", exclude={"configuration_key"}
                        ),
                    )
        except BaseException:
            await self._close_connection(owner)
            raise

    async def _work(
        self, owner: _SessionRuntime, run_id: str, selection: HarnessSelection
    ) -> None:
        finished = owner.finished
        try:
            await self._ensure_prepared(owner, selection)
            connection = owner.connection
            if connection is None:
                raise CodeUnavailableError(
                    "The harness connection closed during preparation."
                )
            async with self._locked(owner):
                run = owner.state.active_run(run_id)
                if run is None:
                    return
                if run.stop_requested or not self._accepting:
                    await self._finish_locked(owner, "interrupted")
                    return
                progress = owner.state.prepare_submission()
                await self._commit_progress(owner, progress)
                run, defaults = (
                    progress.run,
                    progress.session.native_permission_defaults,
                )
                assert run is not None
            turn_id = await connection.start_turn(run.text, selection, run.id, defaults)
            async with self._locked(owner):
                if owner.state.active_run(run_id) is None:
                    return
                await self._commit_progress(owner, owner.state.bind_turn(turn_id))
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

    async def _recover_locked(
        self, owner: _SessionRuntime, native: NativeThread
    ) -> None:
        if native.history_notice:
            self._publish(owner, "session.notice", message=native.history_notice)
        saved_runs = await self._store.runs(owner.state.session.id)
        for turn, run in owner.state.match_history(native, saved_runs):
            owner.state.remember_items(
                await self._store.run_items(owner.state.session.id, run.id),
                (run,),
            )
            for item in turn.items:
                self._update_item(owner, item, run, historical=True)
            items = owner.state.pending_items
            await self._commit_progress(
                owner, SessionProgress(owner.state.session, run=run, items=items)
            )
            owner.state.acknowledge_flush()
            self._publish(
                owner,
                "run.updated",
                runs=[run.model_dump(mode="json")],
                items=[item.model_dump(mode="json") for item in items],
            )
            owner.state.prune()

    def _schedule_interrupt(self, owner: _SessionRuntime) -> None:
        run, connection = owner.state.run, owner.connection
        if (
            run is None
            or connection is None
            or not run.stop_requested
            or not run.native_turn_id
            or not owner.state.busy
            or owner.interrupt_for == run.id
        ):
            return
        owner.interrupt_for = run.id
        owner.interrupt_task = self._job(
            self._interrupt(owner, connection, run, owner.finished)
        )

    async def _interrupt(
        self,
        owner: _SessionRuntime,
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
        async with self._locked(owner):
            if (
                owner.connection is not connection
                or owner.state.run is None
                or owner.state.run.id != run.id
                or not owner.state.busy
            ):
                return
            self._detach_connection_locked(owner)
        await connection.close()
        async with self._locked(owner):
            if owner.state.run and owner.state.run.id == run.id and owner.state.busy:
                await self._finish_locked(
                    owner,
                    "interrupted",
                    "The harness did not stop normally; its session process was closed.",
                )

    async def _deliver_answer(
        self, owner: _SessionRuntime, prompt: CodePrompt, response: JsonObject
    ) -> None:
        connection = owner.connection
        if connection is None:
            return
        try:
            async with self._locked(owner):
                if (
                    owner.generation != prompt.generation
                    or owner.state.pending_answer(prompt.id, prompt.generation) is None
                ):
                    return
            await connection.respond(prompt.request_id, response)
            async with self._locked(owner):
                if owner.generation == prompt.generation:
                    resolved = owner.state.resolve_answer(prompt.id, prompt.generation)
                    if resolved is not None:
                        await self._commit_progress(
                            owner,
                            SessionProgress(owner.state.session, prompts=(resolved,)),
                        )
                        self._publish_prompt(owner, resolved)
        except CodeConflictError:
            async with self._locked(owner):
                expired = owner.state.expire_answer(prompt.id)
                if expired is not None:
                    await self._commit_progress(
                        owner, SessionProgress(owner.state.session, prompts=(expired,))
                    )
                    self._publish_prompt(owner, expired)
        except Exception as exc:
            if owner.state.run and owner.state.busy:
                await self._fail(
                    owner,
                    owner.state.run.id,
                    "The native answer could not be confirmed. It was not resent. "
                    + _error_message(exc),
                )
            else:
                await self._close_connection(owner)

    async def _event(self, owner: _SessionRuntime, event: HarnessEvent) -> None:
        try:
            async with self._locked(owner):
                if (
                    owner.deleted
                    or event.generation != owner.generation
                    or not owner.state.accepts_event(event)
                ):
                    return
                if event.kind == "context_usage":
                    progress = owner.state.context_usage(event.context_used_tokens)
                    if progress is not None:
                        await self._commit_progress(owner, progress)
                        self._publish(owner, "session.updated")
                    return
                run = owner.state.run
                known_turn = False
                if (
                    event.kind == "turn_started"
                    and event.turn_id is not None
                    and run is not None
                    and run.native_turn_id is None
                ):
                    known_turn = await self._store.has_native_turn(
                        owner.state.session.id, event.turn_id
                    )
                matches = owner.state.matches_turn(event, known_turn=known_turn)
                if (
                    event.kind in {"turn_started", "turn_completed"}
                    and matches
                    and run is not None
                ):
                    assert event.turn_id is not None
                    await self._commit_progress(
                        owner, owner.state.bind_turn(event.turn_id)
                    )
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
                    await self._remember_native_item(owner, event.item)
                    item, changed = owner.state.stage_review(
                        event.item, event.generation
                    )
                    if item is None:
                        return
                    if changed:
                        self._version += 1
                    await self._flush_locked(owner)
                elif (
                    event.kind == "item"
                    and event.item is not None
                    and matches
                    and run is not None
                ):
                    await self._remember_native_item(owner, event.item)
                    self._update_item(owner, event.item, run)
                    if event.item.complete or owner.state.dirty_characters >= 4096:
                        await self._flush_locked(owner)
                    elif owner.flush_task is None or owner.flush_task.done():
                        owner.flush_task = self._job(self._flush_later(owner))
                elif (
                    event.kind == "item"
                    and event.item is not None
                    and event.item.complete
                ):
                    saved = await self._store.get_native_item(
                        owner.state.session.id, event.item.turn_id, event.item.item_id
                    )
                    if (
                        saved is not None
                        and saved.kind == "tool"
                        and not saved.complete
                    ):
                        previous_run = await self._store.get_run(
                            owner.state.session.id, saved.run_id
                        )
                        if previous_run is not None:
                            owner.state.remember_items((saved,), (previous_run,))
                            self._update_item(owner, event.item, previous_run)
                            await self._flush_locked(owner)
                elif event.kind == "prompt" and event.prompt is not None:
                    request = event.prompt
                    if request.turn_id is not None and not matches:
                        return
                    if owner.state.has_prompt(
                        request, event.generation
                    ) or await self._store.has_prompt_request(
                        owner.state.session.id,
                        event.generation,
                        request.request_id,
                    ):
                        return
                    await self._flush_locked(owner)
                    progress = owner.state.prepare_prompt(request, event.generation)
                    await self._commit_progress(owner, progress)
                    self._publish_prompt(owner, progress.prompts[0])
                elif event.kind == "resolved":
                    for resolved in owner.state.resolve_request(
                        event.request_id, event.generation
                    ):
                        await self._commit_progress(
                            owner,
                            SessionProgress(owner.state.session, prompts=(resolved,)),
                        )
                        self._publish_prompt(owner, resolved)
                elif run is not None and (
                    (event.kind == "error" and matches)
                    or (event.kind == "notice" and event.message)
                ):
                    await self._flush_locked(owner)
                    item = owner.state.prepare_notice(event)
                    await self._commit_progress(
                        owner, SessionProgress(owner.state.session, items=(item,))
                    )
                    self._publish(
                        owner, "item.updated", item=item.model_dump(mode="json")
                    )
                elif event.kind == "session_notice":
                    self._publish(owner, "session.notice", message=event.message or "")
                elif event.kind == "closed":
                    self._detach_connection_locked(owner)
                    if owner.state.busy:
                        await self._finish_locked(
                            owner,
                            "interrupted" if run and run.stop_requested else "failed",
                            event.message or "The harness process ended.",
                        )
                    await self._expire_prompts_locked(owner)
        except Exception as exc:
            if owner.state.run and owner.state.busy:
                # Teardown must not wait for this native event dispatcher from inside
                # its own callback. A separate owned job closes the connection.
                self._job(self._fail(owner, owner.state.run.id, _error_message(exc)))
            else:
                logger.warning(
                    "Code session event could not be saved: exc_type={}",
                    type(exc).__name__,
                )

    async def _remember_native_item(
        self, owner: _SessionRuntime, update: ItemUpdate
    ) -> None:
        if owner.state.native_item(update) is not None:
            return
        saved = await self._store.get_native_item(
            owner.state.session.id, update.turn_id, update.item_id
        )
        if saved is not None:
            run = owner.state.runs.get(saved.run_id)
            if run is None:
                run = await self._store.get_run(owner.state.session.id, saved.run_id)
            assert run is not None
            owner.state.remember_items((saved,), (run,))

    def _update_item(
        self,
        owner: _SessionRuntime,
        update: ItemUpdate,
        run: CodeRun,
        *,
        historical: bool = False,
    ) -> None:
        _, changed = owner.state.stage_item(update, run, historical=historical)
        if changed:
            self._version += 1

    async def _flush_later(self, owner: _SessionRuntime) -> None:
        await asyncio.sleep(0.25)
        try:
            async with self._locked(owner):
                if not owner.deleted and not owner.storage_failed:
                    await self._flush_locked(owner)
        except Exception as exc:
            if owner.state.run:
                await self._fail(owner, owner.state.run.id, _error_message(exc))

    async def _flush_locked(self, owner: _SessionRuntime) -> None:
        items = owner.state.pending_items
        if not items:
            return
        await self._commit_progress(
            owner, SessionProgress(owner.state.session, items=items)
        )
        owner.state.acknowledge_flush()
        for item in sorted(items, key=lambda item: item.sequence):
            self._publish(
                owner,
                "item.updated",
                item=item.model_dump(mode="json"),
                runs=[owner.state.runs[item.run_id].model_dump(mode="json")],
            )

    async def _finish_locked(
        self,
        owner: _SessionRuntime,
        status: RunStatus,
        message: str | None = None,
        error_details: JsonObject | None = None,
    ) -> None:
        progress = owner.state.prepare_finish(status, message, error_details)
        if progress is None:
            return
        await self._commit_progress(owner, progress)
        owner.state.acknowledge_flush()
        self._publish(
            owner,
            "run.updated",
            items=[item.model_dump(mode="json") for item in progress.items],
            prompts=[prompt.model_dump(mode="json") for prompt in progress.prompts],
        )
        owner.finished.set()

    async def _expire_prompts_locked(self, owner: _SessionRuntime) -> None:
        prompts = owner.state.expire_prompts()
        if not prompts:
            return
        await self._commit_progress(
            owner, SessionProgress(owner.state.session, prompts=prompts)
        )
        for prompt in prompts:
            self._publish_prompt(owner, prompt)

    async def _commit_progress(
        self, owner: _SessionRuntime, progress: SessionProgress
    ) -> None:
        if owner.storage_failed:
            raise CodeUnavailableError(
                "Code storage failed. Restart FCC to restore saved history."
            )
        try:
            await self._store.save_progress(
                progress.session,
                owner.state.session.revision,
                run=progress.run,
                items=progress.items,
                prompts=progress.prompts,
            )
        except CodeConflictError, CodeNotFoundError:
            raise
        except Exception:
            owner.storage_failed = True
            if owner.failure_task is None:
                owner.failure_task = self._job(self._halt_storage(owner))
            raise
        owner.state.apply_progress(progress)

    async def _halt_storage(self, owner: _SessionRuntime) -> None:
        await self._close_connection(owner)
        self._storage_failure(owner)

    def _detach_connection_locked(
        self, owner: _SessionRuntime
    ) -> HarnessConnection | None:
        connection = owner.connection
        owner.connection = None
        owner.generation = None
        owner.loaded_thread_id = None
        owner.capabilities = None
        if owner.state.clear_review_liveness() and not owner.deleted:
            self._publish(owner, "session.updated")
        return connection

    async def _close_connection(self, owner: _SessionRuntime) -> None:
        async with self._locked(owner):
            connection = self._detach_connection_locked(owner)
        if connection is not None:
            await connection.close()
        async with self._locked(owner):
            if not owner.deleted and not owner.storage_failed:
                try:
                    if (
                        connection
                        and connection.thread_id
                        and owner.state.session.native_thread_id is None
                    ):
                        await self._commit_progress(
                            owner, owner.state.attach_thread(connection.thread_id)
                        )
                    await self._expire_prompts_locked(owner)
                except Exception:
                    self._storage_failure(owner)

    async def _fail(
        self,
        owner: _SessionRuntime,
        run_id: str,
        message: str,
        *,
        interrupted: bool = False,
    ) -> None:
        async with self._locked(owner):
            if (
                owner.state.run is None
                or owner.state.run.id != run_id
                or not owner.state.busy
            ):
                return
        try:
            await self._close_connection(owner)
            async with self._locked(owner):
                if (
                    owner.state.run
                    and owner.state.run.id == run_id
                    and not owner.storage_failed
                ):
                    status = (
                        "interrupted"
                        if interrupted or owner.state.run.stop_requested
                        else "failed"
                    )
                    await self._finish_locked(owner, status, message)
        except Exception:
            self._storage_failure(owner)

    def _storage_failure(self, owner: _SessionRuntime) -> None:
        owner.storage_failed = True
        owner.finished.set()
        self._publish(
            owner,
            "session.notice",
            message="Code storage failed. This session was stopped; restart FCC to restore saved history.",
        )

    async def _delete_native(
        self, owner: _SessionRuntime, *, reconcile: bool = False
    ) -> None:
        native_complete = False
        try:
            if owner.setup_task is not None and not owner.setup_task.done():
                owner.setup_task.cancel()
                await asyncio.gather(owner.setup_task, return_exceptions=True)
            await self._close_connection(owner)
            thread_id = owner.state.session.native_thread_id
            if thread_id is not None:
                history = await self._factory(owner.state.session.harness).open_history(
                    owner.state.session.cwd, lambda event: self._event(owner, event)
                )
                try:
                    if reconcile:
                        await history.read_thread(thread_id)
                        raise CodeConflictError(
                            "The native conversation still exists. You can retry deletion."
                        )
                    await history.delete_thread(thread_id)
                except NativeHistoryMissing:
                    pass
                finally:
                    await history.close()
            native_complete = True
            await self._close_connection(owner)
            async with self._locked(owner):
                await self._store.delete(owner.state.session.id)
                owner.deleted = True
                owner.state.discard_deleted_history()
                self._publish(owner, "session.deleted")
                if self._owners.get(owner.state.session.id) is owner:
                    del self._owners[owner.state.session.id]
        except Exception as exc:
            await self._close_connection(owner)
            status = (
                "ready" if isinstance(exc, CodeConflictError) else "delete_uncertain"
            )
            async with self._locked(owner):
                progress = owner.state.deletion_failed(status, _error_message(exc))
                try:
                    await self._commit_progress(owner, progress)
                except Exception:
                    self._storage_failure(owner)
                    return
                self._publish(owner, "session.updated")
            if status == "delete_uncertain" and not reconcile and not native_complete:
                await self._delete_native(owner, reconcile=True)
        finally:
            owner.finished.set()

    def _check_owner(self, owner: _SessionRuntime) -> None:
        if owner.deleted:
            raise CodeNotFoundError("Code session was deleted.")
        if owner.storage_failed:
            raise CodeUnavailableError(
                "Code storage failed. Restart FCC to restore saved history."
            )

    def _editable(self, owner: _SessionRuntime, revision: int) -> None:
        self._require_available()
        self._check_owner(owner)
        owner.state.check_editable(revision)

    def _summary(self, owner: _SessionRuntime) -> JsonObject:
        return {
            "session_id": owner.state.session.id,
            "session": owner.state.session.model_dump(mode="json"),
            "run": owner.state.run.model_dump(mode="json") if owner.state.run else None,
            "active_review_ids": list(owner.state.active_review_ids(owner.generation)),
            "version": self._version,
            "epoch": self.epoch,
        }

    def _publish(self, owner: _SessionRuntime, event: str, **data: JsonValue) -> None:
        self._version += 1
        payload = self._summary(owner)
        for key, value in data.items():
            payload[key] = value
        self._events.publish(event, payload)

    def _publish_prompt(self, owner: _SessionRuntime, prompt: CodePrompt) -> None:
        item = owner.state.items[prompt.id]
        self._publish(
            owner,
            "prompt.updated",
            prompt=prompt.model_dump(mode="json"),
            item=item.model_dump(mode="json"),
            runs=[owner.state.runs[item.run_id].model_dump(mode="json")],
        )


def _validate_id(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
        if str(parsed) != value or parsed.version != 4:
            raise ValueError
    except ValueError, AttributeError:
        raise CodeValidationError("Invalid session or command ID.") from None


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
