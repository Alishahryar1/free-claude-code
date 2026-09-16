"""Conversation rules and staged transcript values, without runtime I/O."""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, get_args

from free_claude_code.core.json_types import JsonObject

from .models import (
    ACTIVE_RUN_STATUSES,
    CodeCatalog,
    CodeConflictError,
    CodeItem,
    CodeMode,
    CodeNotFoundError,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
    CodeValidationError,
    HarnessEvent,
    ItemUpdate,
    NativeThread,
    NativeTurn,
    PromptRequest,
    RunStatus,
    now_ms,
)


@dataclass(frozen=True, slots=True)
class SessionProgress:
    """Candidate records saved together by the existing progress transaction."""

    session: CodeSession
    run: CodeRun | None = None
    items: tuple[CodeItem, ...] = ()
    prompts: tuple[CodePrompt, ...] = ()


class SessionState:
    def __init__(
        self,
        session: CodeSession,
        run: CodeRun | None,
        items: Sequence[CodeItem],
        prompts: Sequence[CodePrompt],
        runs: Sequence[CodeRun],
    ) -> None:
        self._session = session
        self._run = run
        self._items = {item.id: item for item in items}
        self._prompts = {prompt.id: prompt for prompt in prompts}
        self._runs = {run.id: run for run in runs}
        self._sequence = max((item.sequence for item in items), default=0)
        self._known_turns = {run.native_turn_id for run in runs if run.native_turn_id}
        if run and run.native_turn_id:
            self._known_turns.add(run.native_turn_id)
        self._dirty: set[str] = set()
        self._dirty_characters = 0
        self._review_generations: dict[str, str] = {}

    @property
    def session(self) -> CodeSession:
        return self._session

    @property
    def run(self) -> CodeRun | None:
        return self._run

    @property
    def items(self) -> Mapping[str, CodeItem]:
        return MappingProxyType(self._items)

    @property
    def prompts(self) -> Mapping[str, CodePrompt]:
        return MappingProxyType(self._prompts)

    @property
    def runs(self) -> Mapping[str, CodeRun]:
        return MappingProxyType(self._runs)

    @property
    def busy(self) -> bool:
        return self._run is not None and self._run.status in ACTIVE_RUN_STATUSES

    @property
    def pending(self) -> bool:
        return any(
            prompt.status in {"pending", "answering"}
            for prompt in self._prompts.values()
        )

    @property
    def pending_items(self) -> tuple[CodeItem, ...]:
        return tuple(self._items[item_id] for item_id in self._dirty)

    @property
    def dirty_characters(self) -> int:
        return self._dirty_characters

    def active_review_ids(self, generation: str | None) -> tuple[str, ...]:
        return tuple(
            item_id
            for item_id, observed in self._review_generations.items()
            if observed == generation
        )

    def clear_review_liveness(self) -> bool:
        changed = bool(self._review_generations)
        self._review_generations.clear()
        return changed

    def accept_session(self, session: CodeSession) -> None:
        self._session = session

    def accept_admission(
        self, session: CodeSession, run: CodeRun, item: CodeItem
    ) -> None:
        self._session, self._run = session, run
        self._runs[run.id] = run
        self._items[item.id] = item
        self._sequence = item.sequence

    def accept_prompt(self, prompt: CodePrompt) -> None:
        self._prompts[prompt.id] = prompt

    def apply_progress(self, progress: SessionProgress) -> None:
        """Install acknowledged records; pending transcript writes stay explicit."""
        self._session = progress.session
        if (run := progress.run) is not None:
            self._runs[run.id] = run
            if self._run is not None and self._run.id == run.id:
                self._run = run
            if run.native_turn_id:
                self._known_turns.add(run.native_turn_id)
        for item in progress.items:
            self._items[item.id] = item
            self._sequence = max(self._sequence, item.sequence)
        self._prompts.update((prompt.id, prompt) for prompt in progress.prompts)

    def acknowledge_flush(self) -> None:
        """The caller holds serialization through saving the complete pending set."""
        self._dirty.clear()
        self._dirty_characters = 0

    def discard_deleted_history(self) -> None:
        self._items.clear()
        self._prompts.clear()
        self._run = None
        self._runs.clear()

    def check_revision(self, revision: int) -> None:
        if self._session.revision != revision:
            raise CodeConflictError(
                "This session changed. Refresh its state and try again."
            )

    def check_editable(self, revision: int) -> None:
        if self._session.status != "ready":
            raise CodeConflictError(
                "This session changed. Refresh its state and try again."
            )
        self.check_revision(revision)

    @staticmethod
    def validate_settings_fields(changes: JsonObject) -> None:
        if not changes or changes.keys() - {
            "title",
            "model",
            "reasoning_effort",
            "mode",
        }:
            raise CodeValidationError(
                "Choose a title, model, effort or mode to update."
            )

    def settings_updates(self, changes: JsonObject) -> dict[str, object]:
        """Validate local rules before the application obtains model inventory."""
        updates: dict[str, object] = dict(changes)
        if "title" in changes:
            title = changes["title"]
            if not isinstance(title, str) or not title.strip() or len(title) > 200:
                raise CodeValidationError("Enter a title of at most 200 characters.")
            updates.update(title=title.strip(), auto_title=False)
        if changes.keys() & {"model", "reasoning_effort", "mode"} and (
            self.busy or self.pending
        ):
            raise CodeConflictError(
                "Finish this turn and its prompts before changing settings."
            )
        if "mode" in changes and changes["mode"] not in get_args(CodeMode.__value__):
            raise CodeValidationError("Choose an available permission mode.")
        return updates

    def prepare_settings(
        self, updates: dict[str, object], catalog: CodeCatalog | None
    ) -> CodeSession:
        updates = dict(updates)
        if updates.keys() & {"model", "reasoning_effort"}:
            assert catalog is not None
            model = updates.get("model", self._session.model)
            entry = next((entry for entry in catalog.models if entry.id == model), None)
            if entry is None:
                raise CodeValidationError(
                    "This model is unavailable. Choose another model."
                )
            effort = updates.get("reasoning_effort", self._session.reasoning_effort)
            if (
                "reasoning_effort" not in updates
                and model != self._session.model
                and effort not in entry.reasoning_efforts
            ):
                effort = None
            if effort is not None and effort not in entry.reasoning_efforts:
                raise CodeValidationError(
                    "This effort is unavailable. Choose another effort."
                )
            updates.update(model=model, reasoning_effort=effort)
        return _revision(self._session, **updates)

    def check_receipt(self, previous: CodeRun, text: str) -> None:
        if previous.session_id != self._session.id or previous.text != text:
            raise CodeConflictError(
                "This Send ID was already used for a different message."
            )

    def check_can_send(self) -> None:
        if self.busy or self.pending:
            raise CodeConflictError("This session is busy. Your draft has been kept.")

    def prepare_admission(
        self,
        operation_id: str,
        text: str,
        model: str,
        reasoning_effort: str | None,
        mode: CodeMode,
    ) -> tuple[CodeSession, CodeRun, CodeItem]:
        self.check_can_send()
        run = CodeRun(
            id=operation_id,
            session_id=self._session.id,
            text=text,
            model=model,
            reasoning_effort=reasoning_effort,
            mode=mode,
        )
        title = (
            " ".join(text.split())[:80]
            if self._session.auto_title
            else self._session.title
        )
        session = _revision(self._session, title=title, auto_title=False, error=None)
        item = CodeItem(
            id=operation_id,
            session_id=self._session.id,
            sequence=self._sequence + 1,
            run_id=operation_id,
            kind="user",
            text=text,
            complete=True,
        )
        return session, run, item

    def active_run(self, run_id: str) -> CodeRun | None:
        return self._run if self._run and self._run.id == run_id and self.busy else None

    def request_stop(self) -> SessionProgress | None:
        run = self._run
        if run is None or run.stop_requested:
            return None
        return SessionProgress(
            self._session,
            run=run.model_copy(update={"stop_requested": True, "status": "stopping"}),
        )

    def attach_thread(
        self, thread_id: str, permission_defaults: JsonObject | None = None
    ) -> SessionProgress:
        return SessionProgress(
            self._session.model_copy(
                update={
                    "native_thread_id": thread_id,
                    "native_permission_defaults": self._session.native_permission_defaults
                    if self._session.native_permission_defaults is not None
                    else permission_defaults,
                }
            )
        )

    def prepare_submission(self) -> SessionProgress:
        assert self._run is not None
        if self._session.native_permission_defaults is None:
            raise CodeUnavailableError(
                "The harness did not return its permission settings. Your input was not sent."
            )
        return SessionProgress(
            self._session.model_copy(update={"native_may_have_input": True}),
            run=self._run.model_copy(update={"submission_started": True}),
        )

    def bind_turn(self, turn_id: str) -> SessionProgress:
        run = self._run
        assert run is not None
        if run.native_turn_id not in {None, turn_id}:
            raise CodeUnavailableError("Codex returned a different turn identity.")
        return SessionProgress(
            self._session,
            run=run.model_copy(
                update={
                    "native_turn_id": turn_id,
                    "status": "stopping" if run.stop_requested else "running",
                }
            ),
        )

    def accepts_event(self, event: HarnessEvent) -> bool:
        return self._session.status == "ready" and event.thread_id in {
            None,
            self._session.native_thread_id,
        }

    def matches_turn(self, event: HarnessEvent) -> bool:
        run = self._run
        return bool(
            run is not None
            and self.busy
            and run.submission_started
            and event.turn_id is not None
            and (
                run.native_turn_id == event.turn_id
                or (
                    run.native_turn_id is None
                    and event.kind == "turn_started"
                    and event.turn_id not in self._known_turns
                )
            )
        )

    def context_usage(self, used_tokens: int | None) -> SessionProgress | None:
        if used_tokens is None or used_tokens == self._session.context_used_tokens:
            return None
        return SessionProgress(
            self._session.model_copy(update={"context_used_tokens": used_tokens})
        )

    def match_history(
        self, native: NativeThread
    ) -> tuple[tuple[NativeTurn, CodeRun], ...]:
        mapped: dict[str, CodeRun] = {}
        for turn in native.turns:
            identities = {
                item.client_id
                for item in turn.items
                if item.kind == "user" and item.client_id
            }
            candidates = [
                run
                for run in self._runs.values()
                if run.native_turn_id == turn.id or run.id in identities
            ]
            if len(candidates) > 1 or any(
                identity not in self._runs for identity in identities
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
            for run in self._runs.values()
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
        # Recovery fills content and diagnostics, preserving outcomes already settled
        # by restart or connection loss.
        return tuple(
            (
                turn,
                mapped[turn.id].model_copy(
                    update={
                        "native_turn_id": turn.id,
                        "error": mapped[turn.id].error or turn.error,
                        "error_details": mapped[turn.id].error_details
                        or turn.error_details,
                    }
                ),
            )
            for turn in native.turns
        )

    def _native_item(self, update: ItemUpdate) -> CodeItem | None:
        return next(
            (
                item
                for item in self._items.values()
                if item.native_turn_id == update.turn_id
                and item.native_item_id == update.item_id
            ),
            None,
        )

    def stage_item(
        self, update: ItemUpdate, run: CodeRun, *, historical: bool = False
    ) -> tuple[CodeItem, bool]:
        if update.kind != "subagent_auto_review":
            self._known_turns.add(update.turn_id)
        existing = self._native_item(update)
        if existing is None and update.kind == "user":
            if update.client_id is not None and update.client_id != run.id:
                raise CodeUnavailableError(
                    "Codex returned a different message identity."
                )
            existing = self._items.get(run.id)
        if existing is not None and existing.run_id != run.id:
            raise CodeUnavailableError(
                "Codex returned output belonging to another turn."
            )
        if existing is None:
            self._sequence += 1
        preserve = bool(historical and existing and existing.complete)
        raw = _merge_source(existing.raw if existing else {}, update.raw, preserve)
        text, detail = update.text, update.detail
        if historical and existing:
            text = (existing.text or text) if preserve else (text or existing.text)
            detail = (
                (existing.detail or detail) if preserve else (detail or existing.detail)
            )
        item = CodeItem(
            id=existing.id if existing else str(uuid.uuid4()),
            session_id=self._session.id,
            sequence=existing.sequence if existing else self._sequence,
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
            return existing, False
        self._items[item.id] = item
        self._dirty.add(item.id)
        self._dirty_characters += abs(
            len(item.text) - (len(existing.text) if existing else 0)
        ) + abs(len(item.detail) - (len(existing.detail) if existing else 0))
        return item, True

    def stage_review(
        self, update: ItemUpdate, generation: str
    ) -> tuple[CodeItem | None, bool]:
        if self._run is None:
            return None, False
        existing = self._native_item(update)
        if existing is not None and existing.complete and not update.complete:
            return None, False
        target = self._runs[existing.run_id] if existing else self._run
        item, changed = self.stage_item(update, target)
        previous = self._review_generations.get(item.id)
        if item.complete:
            self._review_generations.pop(item.id, None)
        else:
            self._review_generations[item.id] = generation
        if previous != self._review_generations.get(item.id):
            # The saved row is reannounced when its connection liveness changes.
            self._dirty.add(item.id)
        return item, changed

    def has_prompt(self, request: PromptRequest, generation: str) -> bool:
        return any(
            prompt.generation == generation and prompt.request_id == request.request_id
            for prompt in self._prompts.values()
        )

    def prepare_prompt(
        self, request: PromptRequest, generation: str
    ) -> SessionProgress:
        if self._run is None:
            raise CodeUnavailableError("Codex returned a prompt without a saved turn.")
        prompt = CodePrompt(
            id=str(uuid.uuid4()),
            session_id=self._session.id,
            generation=generation,
            request_id=request.request_id,
            native_turn_id=request.turn_id,
            native_item_id=request.item_id,
            kind=request.kind,
            form=request.form,
            raw=request.raw,
        )
        item = CodeItem(
            id=prompt.id,
            session_id=self._session.id,
            run_id=self._run.id,
            sequence=self._sequence + 1,
            kind="prompt",
            complete=True,
        )
        return SessionProgress(self._session, items=(item,), prompts=(prompt,))

    def answerable_prompt(
        self, prompt_id: str, response_id: str, generation: str | None
    ) -> CodePrompt:
        prompt = self._prompts.get(prompt_id)
        if prompt is None:
            raise CodeNotFoundError("This prompt no longer exists.")
        if prompt.response_id == response_id:
            return prompt
        if prompt.status != "pending" or prompt.generation != generation:
            raise CodeConflictError(
                "This prompt was already answered or is no longer active."
            )
        return prompt

    def pending_answer(
        self, prompt_id: str, generation: str | None
    ) -> CodePrompt | None:
        current = self._prompts.get(prompt_id)
        return (
            current
            if current is not None
            and current.status == "answering"
            and current.generation == generation
            else None
        )

    def resolve_answer(
        self, prompt_id: str, generation: str | None
    ) -> CodePrompt | None:
        current = self.pending_answer(prompt_id, generation)
        return current.model_copy(update={"status": "resolved"}) if current else None

    def expire_answer(self, prompt_id: str) -> CodePrompt | None:
        current = self._prompts.get(prompt_id)
        return (
            current.model_copy(update={"status": "expired"})
            if current and current.status == "answering"
            else None
        )

    def resolve_request(
        self, request_id: str | int | None, generation: str
    ) -> tuple[CodePrompt, ...]:
        return tuple(
            prompt.model_copy(update={"status": "resolved"})
            for prompt in self._prompts.values()
            if prompt.generation == generation
            and prompt.request_id == request_id
            and prompt.status in {"pending", "answering"}
        )

    def expire_prompts(self) -> tuple[CodePrompt, ...]:
        return tuple(
            prompt.model_copy(update={"status": "expired"})
            for prompt in self._prompts.values()
            if prompt.status in {"pending", "answering"}
        )

    def prepare_notice(self, event: HarnessEvent) -> CodeItem:
        assert self._run is not None
        message = event.message or "Codex reported an error."
        if event.kind == "error" and event.will_retry:
            message = f"Retrying… {message}"
        return CodeItem(
            id=str(uuid.uuid4()),
            session_id=self._session.id,
            sequence=self._sequence + 1,
            run_id=self._run.id,
            kind="notice",
            title="Notice",
            text=message,
            complete=True,
            raw=event.raw,
        )

    def prepare_finish(
        self,
        status: RunStatus,
        message: str | None = None,
        error_details: JsonObject | None = None,
    ) -> SessionProgress | None:
        if self._run is None or not self.busy:
            return None
        run = self._run.model_copy(
            update={
                "status": status,
                "error": message,
                "error_details": error_details or {},
                "finished_at": now_ms(),
            }
        )
        prompts = tuple(
            prompt.model_copy(update={"status": "expired"})
            for prompt in self._prompts.values()
            if prompt.status in {"pending", "answering"}
            and prompt.native_turn_id is not None
            and prompt.native_turn_id == run.native_turn_id
        )
        return SessionProgress(
            _revision(self._session, error=None),
            run=run,
            items=self.pending_items,
            prompts=prompts,
        )

    def prepare_delete(self) -> SessionProgress:
        if self.busy or self.pending:
            raise CodeConflictError(
                "Stop the session and finish its prompts before deleting it."
            )
        return SessionProgress(_revision(self._session, status="deleting", error=None))

    def deletion_failed(
        self, status: Literal["ready", "delete_uncertain"], message: str
    ) -> SessionProgress:
        return SessionProgress(_revision(self._session, status=status, error=message))


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
