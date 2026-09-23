"""Streaming parser for provider-emitted thinking tags."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Type of content chunk."""

    TEXT = "text"
    THINKING = "thinking"


@dataclass
class ContentChunk:
    """A chunk of parsed content."""

    type: ContentType
    content: str


class ThinkTagParser:
    """
    Streaming parser for ``<think>...</think>`` tags.

    Handles partial tags at chunk boundaries by buffering.
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self._buffer: str = ""
        self._in_think_tag: bool = False

    @property
    def in_think_mode(self) -> bool:
        """Whether currently inside a think tag."""
        return self._in_think_tag

    def feed(self, content: str) -> Iterator[ContentChunk]:
        """Feed content and yield parsed chunks."""
        self._buffer += content

        while self._buffer:
            prev_len = len(self._buffer)
            if not self._in_think_tag:
                chunk = self._parse_outside_think()
            else:
                chunk = self._parse_inside_think()

            if chunk:
                yield chunk
            elif len(self._buffer) == prev_len:
                break

    def _parse_outside_think(self) -> ContentChunk | None:
        """Parse content outside think tags."""
        think_start = self._buffer.find(self.OPEN_TAG)
        orphan_close = self._buffer.find(self.CLOSE_TAG)

        if orphan_close != -1 and (think_start == -1 or orphan_close < think_start):
            pre_orphan = self._buffer[:orphan_close]
            self._buffer = self._buffer[orphan_close + len(self.CLOSE_TAG) :]
            if pre_orphan:
                return ContentChunk(ContentType.TEXT, pre_orphan)
            return None

        if think_start == -1:
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1:
                potential_tag = self._buffer[last_bracket:]
                tag_len = len(potential_tag)
                if (
                    tag_len < len(self.OPEN_TAG)
                    and self.OPEN_TAG.startswith(potential_tag)
                ) or (
                    tag_len < len(self.CLOSE_TAG)
                    and self.CLOSE_TAG.startswith(potential_tag)
                ):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ContentChunk(ContentType.TEXT, emit)
                    return None

            emit = self._buffer
            self._buffer = ""
            if emit:
                return ContentChunk(ContentType.TEXT, emit)
            return None

        pre_think = self._buffer[:think_start]
        self._buffer = self._buffer[think_start + len(self.OPEN_TAG) :]
        self._in_think_tag = True
        if pre_think:
            return ContentChunk(ContentType.TEXT, pre_think)
        return None

    def _parse_inside_think(self) -> ContentChunk | None:
        """Parse content inside think tags."""
        think_end = self._buffer.find(self.CLOSE_TAG)

        if think_end == -1:
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1 and len(self._buffer) - last_bracket < len(
                self.CLOSE_TAG
            ):
                potential_tag = self._buffer[last_bracket:]
                if self.CLOSE_TAG.startswith(potential_tag):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ContentChunk(ContentType.THINKING, emit)
                    return None

            emit = self._buffer
            self._buffer = ""
            if emit:
                return ContentChunk(ContentType.THINKING, emit)
            return None

        thinking_content = self._buffer[:think_end]
        self._buffer = self._buffer[think_end + len(self.CLOSE_TAG) :]
        self._in_think_tag = False
        if thinking_content:
            return ContentChunk(ContentType.THINKING, thinking_content)
        return None

    def flush(self) -> ContentChunk | None:
        """Flush any remaining buffered content."""
        if self._buffer:
            chunk_type = (
                ContentType.THINKING if self._in_think_tag else ContentType.TEXT
            )
            content = self._buffer
            self._buffer = ""
            return ContentChunk(chunk_type, content)
        return None
