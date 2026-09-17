"""A pinned provider route and private execution progress, without HTTP ownership."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from free_claude_code.core.openai_responses import OpenAIResponsesRequest

type ResponsesEgress = Literal["chat", "messages", "responses"]


class ResponsesTurn(Protocol):
    def __call__(
        self, request: OpenAIResponsesRequest, /, *, input_tokens: int
    ) -> AsyncIterator[str]: ...


@dataclass(frozen=True, slots=True)
class ResponsesBinding:
    egress: ResponsesEgress
    stream: ResponsesTurn


@dataclass(frozen=True, slots=True)
class ExecutionProgress:
    """Internal activity; never serialized as a client frame."""

    phase: Literal["provider", "local"] = "provider"


type ExecutionChunk = str | ExecutionProgress
