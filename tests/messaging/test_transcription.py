"""Tests for voice note transcription."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from messaging.transcription import (
    MAX_AUDIO_SIZE_BYTES,
    transcribe_audio,
)


def test_transcribe_file_not_found_raises():
    """Non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        transcribe_audio(Path("/nonexistent/file.ogg"), "audio/ogg")


def test_transcribe_file_too_large_raises():
    """File exceeding max size raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"x" * (MAX_AUDIO_SIZE_BYTES + 1))
        path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="too large"):
            transcribe_audio(path, "audio/ogg", whisper_device="auto")
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_local_success():
    """Local backend returns transcribed text."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg content")
        path = Path(f.name)
    try:
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": "Hello world"}
        fake_audio = {"array": [0.0], "sampling_rate": 16000}

        with (
            patch("messaging.transcription._load_audio", return_value=fake_audio),
            patch(
                "messaging.transcription._get_pipeline",
                return_value=mock_pipe,
            ),
        ):
            result = transcribe_audio(path, "audio/ogg", whisper_model="base")

        assert result == "Hello world"
        mock_pipe.assert_called_once_with(
            fake_audio, generate_kwargs={"language": "en", "task": "transcribe"}
        )
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_local_empty_segments_returns_no_speech():
    """Local backend with no speech returns placeholder."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg")
        path = Path(f.name)
    try:
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": ""}
        fake_audio = {"array": [0.0], "sampling_rate": 16000}

        with (
            patch("messaging.transcription._load_audio", return_value=fake_audio),
            patch(
                "messaging.transcription._get_pipeline",
                return_value=mock_pipe,
            ),
        ):
            result = transcribe_audio(path, "audio/ogg", whisper_model="base")

        assert result == "(no speech detected)"
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_invalid_device_raises():
    """Invalid whisper_device raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg")
        path = Path(f.name)
    try:
        # Mock settings to return invalid device "auto"
        mock_settings = MagicMock()
        mock_settings.whisper_device = "auto"
        mock_settings.whisper_model = "base"

        # Patch _load_audio to avoid ImportError from missing librosa
        # Device validation happens in _get_pipeline before torch import
        with (
            patch("messaging.transcription.get_settings", return_value=mock_settings),
            patch("messaging.transcription._load_audio"),
            pytest.raises(ValueError, match="whisper_device must be 'cpu' or 'cuda'"),
        ):
            transcribe_audio(path, "audio/ogg", whisper_device="auto")
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_local_import_error_raises():
    """Local backend when voice_local extra not installed raises ImportError."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg")
        path = Path(f.name)
    try:
        with (
            patch(
                "messaging.transcription._get_pipeline",
                side_effect=ImportError(
                    "Local Whisper requires the voice_local extra. "
                    "Install with: uv sync --extra voice_local"
                ),
            ),
            pytest.raises(ImportError, match="voice_local extra"),
        ):
            transcribe_audio(path, "audio/ogg", whisper_device="auto")
    finally:
        path.unlink(missing_ok=True)
