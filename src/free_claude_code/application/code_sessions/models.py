"""FCC-owned coding conversation state and harness-neutral events."""

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from free_claude_code.core.json_types import JsonObject

type RunStatus = Literal[
    "preparing", "running", "stopping", "completed", "interrupted", "failed"
]
ACTIVE_RUN_STATUSES = frozenset({"preparing", "running", "stopping"})
type CodeMode = Literal["config", "ask", "auto_review", "full_access"]


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CodeSession(Record):
    id: str
    cwd: str
    model: str
    reasoning_effort: str | None = None
    mode: CodeMode = "config"
    harness: Literal["codex"] = "codex"
    title: str = "New code session"
    auto_title: bool = True
    native_thread_id: str | None = None
    native_permission_defaults: JsonObject | None = None
    native_may_have_input: bool = False
    revision: int = 1
    status: Literal["ready", "deleting", "delete_uncertain"] = "ready"
    error: str | None = None
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)


class CodeRun(Record):
    id: str
    session_id: str
    text: str
    model: str
    reasoning_effort: str | None = None
    mode: CodeMode = "config"
    ordinal: int = 0
    status: RunStatus = "preparing"
    submission_started: bool = False
    native_turn_id: str | None = None
    stop_requested: bool = False
    error: str | None = None
    error_details: JsonObject = Field(default_factory=dict)
    created_at: int = Field(default_factory=now_ms)
    finished_at: int | None = None


class CodeItem(Record):
    id: str
    session_id: str
    sequence: int
    run_id: str
    native_turn_id: str | None = None
    native_item_id: str | None = None
    kind: str
    title: str = ""
    text: str = ""
    detail: str = ""
    complete: bool = False
    raw: JsonObject = Field(default_factory=dict)


class CodePrompt(Record):
    """Form and response state for the prompt CodeItem with the same session/id."""

    id: str
    session_id: str
    generation: str
    request_id: str | int
    native_turn_id: str | None = None
    native_item_id: str | None = None
    kind: str
    form: JsonObject
    raw: JsonObject
    status: Literal["pending", "answering", "resolved", "expired"] = "pending"
    response_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CodeDetail:
    session: CodeSession
    run: CodeRun | None
    items: tuple[CodeItem, ...]
    prompts: tuple[CodePrompt, ...]
    epoch: str
    version: int
    cursor: int
    next_before: tuple[int, int] | None = None
    runs: tuple[CodeRun, ...] = ()
    active_prompt_ids: tuple[str, ...] = ()
    active_review_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodeItemPage:
    items: tuple[CodeItem, ...]
    runs: tuple[CodeRun, ...]
    next_before: tuple[int, int] | None = None


class CodeModel(Record):
    id: str
    display_name: str
    provider_id: str
    model_name: str
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class CodeCatalog:
    default_model: str
    models: tuple[CodeModel, ...]


@dataclass(frozen=True, slots=True)
class CodePage:
    sessions: tuple[CodeSession, ...]
    next_cursor: tuple[int, str] | None = None


@dataclass(frozen=True, slots=True)
class ItemUpdate:
    turn_id: str
    item_id: str
    kind: str
    text: str = ""
    title: str = ""
    detail: str = ""
    raw: JsonObject = field(default_factory=dict)
    complete: bool = False
    client_id: str | None = None


@dataclass(frozen=True, slots=True)
class PromptRequest:
    request_id: str | int
    kind: str
    form: JsonObject
    raw: JsonObject
    turn_id: str | None = None
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    generation: str
    thread_id: str | None
    kind: Literal[
        "turn_started",
        "turn_completed",
        "item",
        "prompt",
        "resolved",
        "error",
        "notice",
        "closed",
    ]
    turn_id: str | None = None
    item: ItemUpdate | None = None
    prompt: PromptRequest | None = None
    request_id: str | int | None = None
    status: RunStatus = "completed"
    message: str | None = None
    will_retry: bool = False
    error_details: JsonObject = field(default_factory=dict)
    raw: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NativeTurn:
    id: str
    items: tuple[ItemUpdate, ...] = ()
    status: RunStatus = "completed"
    error: str | None = None
    error_details: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NativeThread:
    id: str
    turns: tuple[NativeTurn, ...] = ()
    permission_defaults: JsonObject | None = None


class CodeError(Exception):
    """A user-facing failure of the coding session application."""


class CodeConflictError(CodeError):
    pass


class CodeNotFoundError(CodeError):
    pass


class CodeValidationError(CodeError):
    pass


class CodeUnavailableError(CodeError):
    pass


class NativeHistoryMissing(CodeNotFoundError):
    """The registered conversation no longer exists in native storage."""
