"""Process-wide best-effort plain-text token estimation."""

from importlib import resources
from threading import Lock
from typing import Protocol

from loguru import logger

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


class _TokenEncoder(Protocol):
    def encode(
        self, text: str, *, disallowed_special: tuple[str, ...]
    ) -> list[int]: ...


def _load_encoder() -> _TokenEncoder | None:
    try:
        from tiktoken import Encoding
        from tiktoken.load import load_tiktoken_bpe

        asset = resources.files("free_claude_code.core").joinpath(
            "data/cl100k_base.tiktoken"
        )
        with resources.as_file(asset) as path:
            ranks = load_tiktoken_bpe(
                str(path),
                expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
            )
        # Canonical cl100k_base definition; provenance and MIT notice in data/.
        return Encoding(
            name="cl100k_base",
            pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
            mergeable_ranks=ranks,
            special_tokens={
                "<|endoftext|>": 100257,
                "<|fim_prefix|>": 100258,
                "<|fim_middle|>": 100259,
                "<|fim_suffix|>": 100260,
                "<|endofprompt|>": 100276,
            },
        )
    except Exception as exc:
        logger.warning(
            "cl100k_base token encoder unavailable ({}); using approximate token estimates",
            type(exc).__name__,
        )
        return None


class _Uninitialized:
    pass


_ENCODER: _TokenEncoder | _Uninitialized | None = _Uninitialized()
_ENCODER_LOCK = Lock()


def initialize_token_estimation() -> _TokenEncoder | None:
    """Initialize once from local data; server startup runs this in a worker."""
    global _ENCODER
    with _ENCODER_LOCK:
        if isinstance(_ENCODER, _Uninitialized):
            _ENCODER = _load_encoder()
        return _ENCODER


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for plain text using the shared process-wide encoder."""
    if not text:
        return 0
    encoder = initialize_token_estimation()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return max(1, len(text) // 4)
