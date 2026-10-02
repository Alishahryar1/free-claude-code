"""Public writer operations used by source-to-Messages adapters."""

from typing import Any, Protocol

from free_claude_code.core.stream_events import StreamEvent


class MessagesWriter(Protocol):
    def allocate_block_index(self) -> int: ...
    def accept_event(self, event: StreamEvent) -> list[StreamEvent]: ...
    def start_message(self) -> list[StreamEvent]: ...
    def responses_terminal(
        self, response: dict[str, Any], *, incomplete: bool
    ) -> None: ...
