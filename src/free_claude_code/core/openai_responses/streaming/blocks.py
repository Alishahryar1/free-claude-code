"""Block state for OpenAI Responses streaming assembly."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class TextBlockState:
    index: int
    output_index: int
    item_id: str
    text_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningBlockState:
    index: int
    output_index: int
    item_id: str
    text_parts: list[str] = field(default_factory=list)
    encrypted_content: str | None = None


@dataclass(slots=True)
class ToolBlockState:
    index: int
    output_index: int
    item_id: str
    call_id: str
    kind: Literal["function", "custom"]
    name: str
    namespace: str | None = None
    argument_parts: list[str] = field(default_factory=list)


BlockState = TextBlockState | ReasoningBlockState | ToolBlockState
