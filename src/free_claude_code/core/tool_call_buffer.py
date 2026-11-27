"""Bounded source-order retention, independent of protocols and presentation."""

from collections import deque
from collections.abc import Hashable, Iterator
from dataclasses import dataclass

from .failures import ExecutionFailure, FailureKind
from .stream_recovery import SourceRecoverySnapshot, StreamFailureContext

MAX_PENDING_BYTES = 16 * 1024 * 1024
MAX_PENDING_RECORDS = 65_536


class ToolBufferLimitError(ExecutionFailure):
    def __init__(
        self,
        *,
        kind: FailureKind = FailureKind.UPSTREAM,
        status_code: int = 502,
        message: str = "Buffered response exceeded the pending output limit.",
        retryable: bool = False,
        stream_context: StreamFailureContext | None = None,
    ) -> None:
        super().__init__(
            kind,
            status_code,
            message,
            retryable,
            stream_context
            or StreamFailureContext(
                SourceRecoverySnapshot(unsafe_reason="pending_output_limit")
            ),
        )


@dataclass(slots=True)
class RetentionBudget:
    max_bytes: int = MAX_PENDING_BYTES
    max_records: int = MAX_PENDING_RECORDS
    bytes: int = 0
    records: int = 0

    def add(self, size: int, records: int = 1) -> None:
        if (
            self.bytes + size > self.max_bytes
            or self.records + records > self.max_records
        ):
            raise ToolBufferLimitError()
        self.bytes += size
        self.records += records

    def release(self, size: int, records: int = 1) -> None:
        self.bytes -= size
        self.records -= records


@dataclass(slots=True)
class _Slot[T]:
    value: T
    key: Hashable | None = None
    ready: bool = False
    discarded: bool = False
    size: int = 0
    records: int = 1


class OrderedToolCallBuffer[T]:
    """Only a ready prefix can leave; failure drops unresolved placeholders."""

    def __init__(self, budget: RetentionBudget | None = None) -> None:
        self.budget = budget if budget is not None else RetentionBudget()
        self._queue: deque[_Slot[T]] = deque()
        self._pending: dict[Hashable, _Slot[T]] = {}

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def reserve(self, key: Hashable, value: T, *, size: int = 0) -> None:
        if key in self._pending:
            raise ValueError("Tool slot is already reserved")
        self.budget.add(size)
        slot = _Slot(value, key=key, size=size)
        self._pending[key] = slot
        self._queue.append(slot)

    def append(self, value: T, *, size: int = 0) -> None:
        self.budget.add(size)
        self._queue.append(_Slot(value, ready=True, size=size))

    def grow(self, key: Hashable, size: int, records: int = 1) -> None:
        slot = self._pending[key]
        self.budget.add(size, records)
        slot.size += size
        slot.records += records

    def value(self, key: Hashable) -> T:
        return self._pending[key].value

    def complete(self, key: Hashable) -> None:
        self._pending.pop(key).ready = True

    def discard(self, key: Hashable) -> None:
        slot = self._pending.pop(key)
        slot.ready = slot.discarded = True

    def discard_pending(self) -> None:
        for key in tuple(self._pending):
            self.discard(key)

    def drain(self) -> Iterator[T]:
        while self._queue and self._queue[0].ready:
            slot = self._queue.popleft()
            self.budget.release(slot.size, slot.records)
            if not slot.discarded:
                yield slot.value

    def clear(self) -> None:
        for slot in self._queue:
            self.budget.release(slot.size, slot.records)
        self._queue.clear()
        self._pending.clear()
