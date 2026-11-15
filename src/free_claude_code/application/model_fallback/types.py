"""The shapes a caller hands the model fallback sequence."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Literal

from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import ContinuationSeed

from ..routing import ProviderModelCandidate

WireApi = Literal["messages", "responses", "gemini", "chat"]
CandidateStreamOpener = Callable[
    [int, ProviderModelCandidate, ContinuationSeed | None],
    Awaitable[AsyncIterator[StreamOutput]],
]
