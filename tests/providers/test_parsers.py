import pytest

from free_claude_code.core.anthropic import (
    ContentType,
    ThinkTagParser,
)


def test_think_tag_parser_basic():
    parser = ThinkTagParser()
    chunks = list(parser.feed("Hello <think>reasoning</think> world"))

    assert len(chunks) == 3
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello "
    assert chunks[1].type == ContentType.THINKING
    assert chunks[1].content == "reasoning"
    assert chunks[2].type == ContentType.TEXT
    assert chunks[2].content == " world"


def test_think_tag_parser_streaming():
    parser = ThinkTagParser()

    # Partial tag
    chunks = list(parser.feed("Hello <thi"))
    assert len(chunks) == 1
    assert chunks[0].content == "Hello "

    # Complete tag
    chunks = list(parser.feed("nk>reasoning</think>"))
    assert len(chunks) == 1
    assert chunks[0].type == ContentType.THINKING
    assert chunks[0].content == "reasoning"


# --- New Robustness Tests ---


# --- Orphan </think> Tag Tests (Step Fun AI compatibility) ---


def test_orphan_close_tag_stripped():
    """Orphan </think> without opening tag should be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("Hello </think> world"))

    # Should get one text chunk with orphan tag stripped
    assert len(chunks) == 2
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello "
    assert chunks[1].type == ContentType.TEXT
    assert chunks[1].content == " world"


def test_orphan_close_tag_at_start():
    """Orphan </think> at start should be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("</think>Hello world"))

    assert len(chunks) == 1
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello world"


def test_orphan_close_tag_at_end():
    """Orphan </think> at end should be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("Hello world</think>"))

    assert len(chunks) == 1
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello world"


def test_multiple_orphan_close_tags():
    """Multiple orphan </think> tags should all be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("a</think>b</think>c"))

    text = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    assert text == "abc"
    assert "</think>" not in text


def test_orphan_close_tag_streaming():
    """Orphan </think> split across chunks should be stripped."""
    parser = ThinkTagParser()

    # Feed partial orphan tag
    chunks1 = list(parser.feed("Hello </thi"))
    assert len(chunks1) == 1
    assert chunks1[0].content == "Hello "

    # Complete the orphan tag
    chunks2 = list(parser.feed("nk> world"))
    assert len(chunks2) == 1
    assert chunks2[0].type == ContentType.TEXT
    assert chunks2[0].content == " world"


def test_orphan_close_with_valid_think_pair():
    """Orphan </think> followed by valid <think>...</think> pair."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("a</think>b<think>thinking</think>c"))

    types = [c.type for c in chunks]
    # contents = [c.content for c in chunks] # Unused

    assert ContentType.TEXT in types
    assert ContentType.THINKING in types
    # Text should be "ab" and "c", thinking should be "thinking"
    text_content = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    think_content = "".join(c.content for c in chunks if c.type == ContentType.THINKING)
    assert text_content == "abc"
    assert think_content == "thinking"


# --- Parametrized Edge Case Tests ---


@pytest.mark.parametrize(
    "input_text,expected_text",
    [
        ("Hello </think> world", "Hello  world"),
        ("</think>Hello world", "Hello world"),
        ("Hello world</think>", "Hello world"),
        ("a</think>b</think>c", "abc"),
        ("</think>", ""),
        ("</think></think>", ""),
    ],
    ids=[
        "middle",
        "start",
        "end",
        "multiple",
        "only_orphan",
        "consecutive_orphans",
    ],
)
def test_orphan_close_tag_parametrized(input_text, expected_text):
    """Parametrized: orphan </think> tags should be stripped from various positions."""
    parser = ThinkTagParser()
    chunks = list(parser.feed(input_text))
    text = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    assert text == expected_text
    assert "</think>" not in text


def test_think_tag_parser_empty_input():
    """Empty string input should yield no chunks."""
    parser = ThinkTagParser()
    chunks = list(parser.feed(""))
    assert chunks == []


def test_think_tag_parser_flush_no_content():
    """Flush with no buffered content should return None."""
    parser = ThinkTagParser()
    result = parser.flush()
    assert result is None


def test_think_tag_parser_flush_buffered_text():
    """Flush with buffered text returns TEXT chunk."""
    parser = ThinkTagParser()
    # Feed partial tag that stays buffered
    list(parser.feed("Hello <thi"))
    result = parser.flush()
    assert result is not None
    assert result.type == ContentType.TEXT
    assert "<thi" in result.content


def test_think_tag_parser_flush_inside_think():
    """Flush while inside <think> with buffered partial close tag returns THINKING chunk."""
    parser = ThinkTagParser()
    # Feed content that ends with a potential partial </think> tag, which stays buffered
    chunks = list(parser.feed("<think>partial reasoning</thi"))
    # "partial reasoning" is emitted, "</thi" stays buffered as potential close tag
    assert any(c.type == ContentType.THINKING for c in chunks)
    result = parser.flush()
    assert result is not None
    assert result.type == ContentType.THINKING
    assert "</thi" in result.content


def test_think_tag_parser_empty_think_tags():
    """Empty <think></think> pair should yield no thinking content."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("<think></think>remaining"))
    # Empty think yields nothing for thinking, just the remaining text
    # types = [c.type for c in chunks] # Unused
    text = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    assert text == "remaining"


def test_think_tag_parser_unicode():
    """Unicode content inside and outside think tags."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("日本語 <think>思考中 🤔</think> 結果"))
    thinking = "".join(c.content for c in chunks if c.type == ContentType.THINKING)
    text = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    assert thinking == "思考中 🤔"
    assert "日本語" in text
    assert "結果" in text

    # Should not crash; may or may not detect a tool depending on regex match
