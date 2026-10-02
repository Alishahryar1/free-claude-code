"""Public writer operations used by source-to-Responses adapters."""

from typing import Protocol

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_events import StreamEvent

from .messages_usage import NativeMessagesUsage


class ResponsesWriter(Protocol):
    @property
    def messages_usage(self) -> NativeMessagesUsage: ...
    def allocate_output_index(self) -> int: ...
    def accept_event(self, event: StreamEvent) -> list[StreamEvent]: ...
    def start_response(self) -> list[StreamEvent]: ...
    def complete_response(
        self, *, incomplete: bool, usage: JsonObject | None
    ) -> None: ...
