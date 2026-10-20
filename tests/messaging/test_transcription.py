"""Tests for the instance-owned local Whisper transcriber."""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.messaging.transcription import TranscriptionService


def _service(*, api_key: str = "") -> TranscriptionService:
    return TranscriptionService(
        model="base",
        device="cpu",
        huggingface_api_key=api_key,
    )


def _fake_optional_modules(
    pipeline: MagicMock,
) -> tuple[SimpleNamespace, SimpleNamespace, MagicMock, MagicMock]:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16=object(),
        float32=object(),
    )

    model = MagicMock()
    model.to.return_value = model
    model_loader = MagicMock()
    model_loader.from_pretrained.return_value = model
    processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())
    processor_loader = MagicMock()
    processor_loader.from_pretrained.return_value = processor

    transformers = SimpleNamespace(
        AutoModelForSpeechSeq2Seq=model_loader,
        AutoProcessor=processor_loader,
        pipeline=MagicMock(return_value=pipeline),
    )
    return torch, transformers, model_loader, processor_loader


@pytest.mark.asyncio
async def test_transcription_service_transcribes_and_reuses_its_pipeline(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"voice")
    pipeline = MagicMock(return_value={"text": " Hello world "})
    torch, transformers, model_loader, processor_loader = _fake_optional_modules(
        pipeline
    )
    fake_audio = {"array": [0.0], "sampling_rate": 16000}
    service = _service(api_key="hf-provider-key")

    with (
        patch.dict(
            "sys.modules",
            {"torch": torch, "transformers": transformers},
        ),
        patch(
            "free_claude_code.messaging.transcription._load_audio",
            return_value=fake_audio,
        ),
    ):
        first = await service.transcribe(audio_path)
        second = await service.transcribe(audio_path)

    assert first == "Hello world"
    assert second == "Hello world"
    assert transformers.pipeline.call_count == 1
    assert pipeline.call_count == 2
    model_loader.from_pretrained.assert_called_once_with(
        "openai/whisper-base",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        token="hf-provider-key",
    )
    processor_loader.from_pretrained.assert_called_once_with(
        "openai/whisper-base",
        token="hf-provider-key",
    )


@pytest.mark.asyncio
async def test_separate_services_do_not_share_pipeline_instances(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice")
    pipeline = MagicMock(return_value={"text": "ok"})
    torch, transformers, _model_loader, _processor_loader = _fake_optional_modules(
        pipeline
    )
    first = _service()
    second = _service()

    with (
        patch.dict(
            "sys.modules",
            {"torch": torch, "transformers": transformers},
        ),
        patch(
            "free_claude_code.messaging.transcription._load_audio",
            return_value={"array": [0.0], "sampling_rate": 16000},
        ),
    ):
        await first.transcribe(audio_path)
        await second.transcribe(audio_path)

    assert transformers.pipeline.call_count == 2


@pytest.mark.asyncio
async def test_transcription_service_serializes_concurrent_inference(
    tmp_path: Path,
) -> None:
    service = _service()
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice")
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def transcribe_sync(_path: Path) -> str:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1
        return "ok"

    with patch.object(service, "_transcribe_sync", side_effect=transcribe_sync):
        results = await asyncio.gather(
            service.transcribe(audio_path),
            service.transcribe(audio_path),
            service.transcribe(audio_path),
        )

    assert results == ["ok", "ok", "ok"]
    assert max_active == 1


@pytest.mark.asyncio
async def test_transcription_service_close_releases_pipeline_and_is_terminal(
    tmp_path: Path,
) -> None:
    service = _service()
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice")
    pipeline = MagicMock(return_value={"text": "ok"})

    with (
        patch.object(service, "_get_pipeline", return_value=pipeline),
        patch(
            "free_claude_code.messaging.transcription._load_audio",
            return_value={"array": [0.0], "sampling_rate": 16000},
        ),
    ):
        await service.transcribe(audio_path)

    service._pipeline = pipeline
    await service.close()
    await service.close()

    assert service._pipeline is None
    with pytest.raises(RuntimeError, match="closed"):
        await service.transcribe(audio_path)


@pytest.mark.asyncio
async def test_cancelled_transcription_keeps_ownership_until_thread_exits(
    tmp_path: Path,
) -> None:
    service = _service(api_key="hf-secret")
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice")
    started = threading.Event()
    release = threading.Event()
    pipeline = object()

    def blocking_transcribe(_path: Path) -> str:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release active transcription")
        service._pipeline = pipeline
        return "finished"

    close_task: asyncio.Task[None] | None = None
    with patch.object(service, "_transcribe_sync", side_effect=blocking_transcribe):
        transcribe_task = asyncio.create_task(service.transcribe(audio_path))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            transcribe_task.cancel()
            await asyncio.sleep(0)
            close_task = asyncio.create_task(service.close())
            await asyncio.sleep(0)

            assert not transcribe_task.done()
            assert not close_task.done()
            assert service._huggingface_api_key == "hf-secret"
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await transcribe_task
        assert close_task is not None
        await close_task

    assert service._pipeline is None
    assert service._huggingface_api_key is None
    with pytest.raises(RuntimeError, match="closed"):
        await service.transcribe(audio_path)


@pytest.mark.asyncio
async def test_transcription_service_returns_no_speech_placeholder(
    tmp_path: Path,
) -> None:
    service = _service()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"voice")
    pipeline = MagicMock(return_value={"text": []})

    with (
        patch.object(service, "_get_pipeline", return_value=pipeline),
        patch(
            "free_claude_code.messaging.transcription._load_audio",
            return_value={"array": [0.0], "sampling_rate": 16000},
        ),
    ):
        result = await service.transcribe(audio_path)

    assert result == "(no speech detected)"


def test_transcription_service_rejects_non_local_device() -> None:
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        TranscriptionService(model="base", device="nvidia_nim")


@pytest.mark.asyncio
async def test_transcription_service_reports_missing_local_extra(
    tmp_path: Path,
) -> None:
    service = _service()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"voice")

    with (
        patch.dict("sys.modules", {"torch": None}),
        pytest.raises(ImportError, match="voice_local extra"),
    ):
        await service.transcribe(audio_path)
