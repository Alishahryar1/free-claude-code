"""Think tag parser for extracting reasoning content from responses."""

from dataclasses import dataclass
from typing import Optional, Iterator
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
    Streaming parser for <think>...</think> tags.

    Handles partial tags at chunk boundaries by buffering.
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"
    OPEN_TAG_LEN = 7
    CLOSE_TAG_LEN = 8

    def __init__(self):
        self._buffer: str = ""
        self._in_think_tag: bool = False

    @property
    def in_think_mode(self) -> bool:
        """Whether currently inside a think tag."""
        return self._in_think_tag

    def feed(self, content: str) -> Iterator[ContentChunk]:
        """
        Feed content and yield parsed chunks.

        Handles partial tags by buffering content near potential tag boundaries.
        """
        self._buffer += content

        while self._buffer:
            if not self._in_think_tag:
                chunk = self._parse_outside_think()
                if chunk:
                    yield chunk
                else:
                    break
            else:
                chunk = self._parse_inside_think()
                if chunk:
                    yield chunk
                else:
                    break

    def _parse_outside_think(self) -> Optional[ContentChunk]:
        """Parse content outside think tags."""
        think_start = self._buffer.find(self.OPEN_TAG)
        orphan_close = self._buffer.find(self.CLOSE_TAG)

        # Handle orphan </think> - strip it (Step Fun AI sends reasoning via
        # reasoning_content but may leak closing tags in content)
        if orphan_close != -1 and (think_start == -1 or orphan_close < think_start):
            pre_orphan = self._buffer[:orphan_close]
            self._buffer = self._buffer[orphan_close + self.CLOSE_TAG_LEN :]
            if pre_orphan:
                return ContentChunk(ContentType.TEXT, pre_orphan)
            # Continue parsing after stripping orphan tag
            return self._parse_outside_think()

        if think_start == -1:
            # No tag found - check for partial tag at end
            # We buffer any trailing '<' and subsequent characters that could be part of <think> or </think>
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1:
                potential_tag = self._buffer[last_bracket:]
                tag_len = len(potential_tag)
                # Check if could be partial <think> or </think>
                if (
                    tag_len < self.OPEN_TAG_LEN
                    and self.OPEN_TAG.startswith(potential_tag)
                ) or (
                    tag_len < self.CLOSE_TAG_LEN
                    and self.CLOSE_TAG.startswith(potential_tag)
                ):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ContentChunk(ContentType.TEXT, emit)
                    return None

            # No partial tag found or it's irrelevant
            emit = self._buffer
            self._buffer = ""
            if emit:
                return ContentChunk(ContentType.TEXT, emit)
            return None
        else:
            # Found <think> tag
            pre_think = self._buffer[:think_start]
            self._buffer = self._buffer[think_start + self.OPEN_TAG_LEN :]
            self._in_think_tag = True
            if pre_think:
                return ContentChunk(ContentType.TEXT, pre_think)
            # Continue parsing inside think tag
            return self._parse_inside_think()

    def _parse_inside_think(self) -> Optional[ContentChunk]:
        """Parse content inside think tags."""
        think_end = self._buffer.find(self.CLOSE_TAG)

        if think_end == -1:
            # No closing tag - check for partial at end
            last_bracket = self._buffer.rfind("<")
            if (
                last_bracket != -1
                and len(self._buffer) - last_bracket < self.CLOSE_TAG_LEN
            ):
                # Check if the partial string could be the start of </think>
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
        else:
            # Found </think> tag
            thinking_content = self._buffer[:think_end]
            self._buffer = self._buffer[think_end + self.CLOSE_TAG_LEN :]
            self._in_think_tag = False
            if thinking_content:
                return ContentChunk(ContentType.THINKING, thinking_content)
            # Continue parsing outside think tag
            return self._parse_outside_think()

    def flush(self) -> Optional[ContentChunk]:
        """Flush any remaining buffered content."""
        if self._buffer:
            chunk_type = (
                ContentType.THINKING if self._in_think_tag else ContentType.TEXT
            )
            content = self._buffer
            self._buffer = ""
            return ContentChunk(chunk_type, content)
        return None

    def reset(self):
        """Reset parser state."""
        self._buffer = ""
        self._in_think_tag = False
