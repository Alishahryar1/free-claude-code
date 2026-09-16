"""Tests for process-wide plain-text token estimation."""

from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.core import token_estimation


class _RecordingEncoder:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def encode(self, text: str, *, disallowed_special: tuple[str, ...]) -> list[int]:
        self.calls.append((text, disallowed_special))
        return self.tokens


def test_encoder_is_loaded_by_canonical_name() -> None:
    encoder = MagicMock()

    with patch.object(
        __import__("tiktoken"),
        "Encoding",
        return_value=encoder,
    ) as get_encoding:
        assert token_estimation._load_encoder() is encoder

    assert get_encoding.call_args.kwargs["name"] == "cl100k_base"


def test_encoder_acquisition_failure_uses_safe_fallback_warning() -> None:
    with (
        patch.object(
            __import__("tiktoken.load", fromlist=["load_tiktoken_bpe"]),
            "load_tiktoken_bpe",
            side_effect=RuntimeError("Bearer secret"),
        ),
        patch.object(token_estimation.logger, "warning") as warning,
    ):
        assert token_estimation._load_encoder() is None

    warning.assert_called_once_with(
        "cl100k_base token encoder unavailable ({}); using approximate token estimates",
        "RuntimeError",
    )
    assert "Bearer secret" not in str(warning.call_args)


def test_estimate_text_tokens_uses_cached_encoder() -> None:
    encoder = _RecordingEncoder([1, 2, 3])

    with patch.object(token_estimation, "_ENCODER", encoder):
        assert token_estimation.estimate_text_tokens("hello") == 3

    assert encoder.calls == [("hello", ())]


def test_empty_text_skips_encoder() -> None:
    encoder = _RecordingEncoder([1])

    with patch.object(token_estimation, "_ENCODER", encoder):
        assert token_estimation.estimate_text_tokens("") == 0

    assert encoder.calls == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a", 1),
        ("abcdefg", 1),
        ("abcdefgh", 2),
    ],
)
def test_estimate_text_tokens_falls_back_to_character_ratio(
    text: str,
    expected: int,
) -> None:
    with patch.object(token_estimation, "_ENCODER", None):
        assert token_estimation.estimate_text_tokens(text) == expected


def test_encoder_execution_errors_are_not_hidden() -> None:
    encoder = MagicMock()
    encoder.encode.side_effect = RuntimeError("encode failed")

    with (
        patch.object(token_estimation, "_ENCODER", encoder),
        pytest.raises(RuntimeError, match="encode failed"),
    ):
        token_estimation.estimate_text_tokens("hello")
