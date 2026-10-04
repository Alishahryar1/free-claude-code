"""Resolve provider tool identities within one Chat response attempt."""

from dataclasses import dataclass

from free_claude_code.providers.failure_policy import RetryableToolProtocolError


class AmbiguousToolCallIdentityError(RetryableToolProtocolError):
    """A fragment cannot be assigned to exactly one existing tool call."""


@dataclass(slots=True)
class _ToolCallIdentity:
    index: int | None = None
    tool_id: str | None = None


class ToolCallIdentityResolver:
    """Keep upstream indexes and IDs separate from locally allocated slots."""

    def __init__(self) -> None:
        self._calls: list[_ToolCallIdentity] = []

    def allocate(self) -> int:
        """Reserve a new logical slot, including for already-completed calls."""
        slot = len(self._calls)
        self._calls.append(_ToolCallIdentity())
        return slot

    def resolve(self, raw_index: object, raw_id: object) -> int:
        index = raw_index if isinstance(raw_index, int) and raw_index >= 0 else None
        tool_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else None

        if index is not None and tool_id is not None:
            matches = [
                slot
                for slot, call in enumerate(self._calls)
                if call.index == index and call.tool_id == tool_id
            ]
            if not matches:
                matches = [
                    slot
                    for slot, call in enumerate(self._calls)
                    if call.index in (None, index) and call.tool_id in (None, tool_id)
                ]
        elif index is not None:
            matches = [
                slot for slot, call in enumerate(self._calls) if call.index == index
            ]
            if not matches:
                matches = [
                    slot for slot, call in enumerate(self._calls) if call.index is None
                ]
        elif tool_id is not None:
            matches = [
                slot for slot, call in enumerate(self._calls) if call.tool_id == tool_id
            ]
            if not matches:
                matches = [
                    slot
                    for slot, call in enumerate(self._calls)
                    if call.tool_id is None
                ]
        else:
            matches = list(range(len(self._calls)))

        if len(matches) > 1:
            raise AmbiguousToolCallIdentityError(
                "Upstream tool-call fragment has ambiguous identity."
            )
        slot = matches[0] if matches else self.allocate()
        call = self._calls[slot]
        if index is not None:
            call.index = index
        if tool_id is not None:
            call.tool_id = tool_id
        return slot

    def tool_id(self, slot: int) -> str | None:
        return self._calls[slot].tool_id

    def order_key(self, slot: int) -> tuple[int, int]:
        return self._calls[slot].index or 0, slot
