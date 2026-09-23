import math
import os
import wave
from pathlib import Path

import pytest

from free_claude_code.messaging.transcription import TranscriptionService
from free_claude_code.messaging.voice import Transcriber
from free_claude_code.providers.nvidia_nim.voice import NvidiaNimTranscriber
from smoke.lib.config import SmokeConfig

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("voice")]


@pytest.mark.asyncio
async def test_voice_transcription_backend_when_explicitly_enabled(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    if not smoke_config.settings.voice_note_enabled:
        pytest.skip("VOICE_NOTE_ENABLED is false")
    if os.getenv("FCC_SMOKE_RUN_VOICE") != "1":
        pytest.skip("set FCC_SMOKE_RUN_VOICE=1 to run transcription smoke")

    wav_path = tmp_path / "smoke-tone.wav"
    _write_tone_wav(wav_path)
    transcriber: Transcriber
    if smoke_config.settings.whisper_device == "nvidia_nim":
        api_key = smoke_config.settings.nvidia_nim_api_key
        if api_key is None:
            pytest.skip("NVIDIA_NIM_API_KEY is required")
        transcriber = NvidiaNimTranscriber(
            model=smoke_config.settings.whisper_model,
            api_key=api_key,
        )
    else:
        transcriber = TranscriptionService(
            model=smoke_config.settings.whisper_model,
            device=smoke_config.settings.whisper_device,
            huggingface_api_key=smoke_config.settings.huggingface_api_key,
        )
    try:
        text = await transcriber.transcribe(wav_path)
    except ImportError as exc:
        pytest.skip(str(exc))
    finally:
        await transcriber.close()
    assert isinstance(text, str)
    assert text.strip()


def _write_tone_wav(path: Path) -> None:
    sample_rate = 16000
    duration_s = 0.25
    amplitude = 8000
    frames = bytearray()
    for i in range(int(sample_rate * duration_s)):
        sample = int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate))
        frames.extend(sample.to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
