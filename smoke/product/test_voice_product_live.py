import os
from pathlib import Path

import pytest

from free_claude_code.messaging.transcription import TranscriptionService
from free_claude_code.providers.nvidia_nim.voice import NvidiaNimTranscriber
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import VoiceFixtureDriver

pytestmark = [pytest.mark.live]


@pytest.mark.smoke_target("voice")
@pytest.mark.asyncio
async def test_voice_local_backend_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    if not smoke_config.settings.voice_note_enabled:
        pytest.skip("missing_env: VOICE_NOTE_ENABLED is false")
    if os.getenv("FCC_SMOKE_RUN_VOICE") != "1":
        pytest.skip("missing_env: set FCC_SMOKE_RUN_VOICE=1 to run voice product smoke")
    if smoke_config.settings.whisper_device not in {"cpu", "cuda"}:
        pytest.skip("missing_env: WHISPER_DEVICE must be cpu or cuda")

    wav_path = tmp_path / "voice-local-product.wav"
    VoiceFixtureDriver.write_tone_wav(wav_path)
    transcriber = TranscriptionService(
        model=smoke_config.settings.whisper_model,
        device=smoke_config.settings.whisper_device,
        huggingface_api_key=smoke_config.settings.huggingface_api_key,
    )
    try:
        text = await transcriber.transcribe(wav_path)
    except ImportError as exc:
        pytest.skip(f"missing_env: {exc}")
    finally:
        await transcriber.close()

    assert isinstance(text, str)
    assert text.strip()


@pytest.mark.smoke_target("voice")
@pytest.mark.asyncio
async def test_voice_nim_backend_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not smoke_config.settings.voice_note_enabled:
        pytest.skip("missing_env: VOICE_NOTE_ENABLED is false")
    if os.getenv("FCC_SMOKE_RUN_VOICE") != "1":
        pytest.skip("missing_env: set FCC_SMOKE_RUN_VOICE=1 to run voice product smoke")
    if smoke_config.settings.whisper_device != "nvidia_nim":
        pytest.skip("missing_env: WHISPER_DEVICE must be nvidia_nim")
    api_key = smoke_config.settings.nvidia_nim_api_key
    if api_key is None:
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is required")

    wav_path = tmp_path / "voice-nim-product.wav"
    VoiceFixtureDriver.write_tone_wav(wav_path)
    transcriber = NvidiaNimTranscriber(
        model=smoke_config.settings.whisper_model,
        api_key=api_key,
    )
    try:
        text = await transcriber.transcribe(wav_path)
    finally:
        await transcriber.close()

    assert isinstance(text, str)
    assert text.strip()
