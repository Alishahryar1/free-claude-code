"""Tests for the NVIDIA NIM voice transcription adapter."""

import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.providers.nvidia_nim.voice import (
    _NIM_ASR_MODEL_MAP,
    NvidiaNimTranscriber,
)


def _fake_riva_client(
    transcript: str,
) -> tuple[SimpleNamespace, SimpleNamespace, MagicMock, MagicMock]:
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[SimpleNamespace(transcript=transcript)],
            )
        ]
    )
    asr_service = MagicMock()
    asr_service.offline_recognize.return_value = response
    auth = MagicMock()

    client = SimpleNamespace(
        Auth=MagicMock(return_value=auth),
        ASRService=MagicMock(return_value=asr_service),
        RecognitionConfig=MagicMock(return_value=object()),
    )
    riva = SimpleNamespace(__path__=[], client=client)
    return riva, client, asr_service, auth


@pytest.mark.asyncio
async def test_nvidia_nim_transcriber_calls_riva_with_owned_configuration(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"audio bytes")
    transcriber = NvidiaNimTranscriber(
        model="openai/whisper-large-v3",
        api_key=" test-nim-key ",
    )
    riva, client, asr_service, auth = _fake_riva_client(" hello from NIM ")

    with patch.dict(
        "sys.modules",
        {"riva": riva, "riva.client": client},
    ):
        result = await transcriber.transcribe(wav)

    assert result == " hello from NIM "
    client.Auth.assert_called_once_with(
        use_ssl=True,
        uri="grpc.nvcf.nvidia.com:443",
        metadata_args=[
            ["function-id", "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"],
            ["authorization", "Bearer test-nim-key"],
        ],
    )
    client.RecognitionConfig.assert_called_once_with(
        language_code="multi",
        max_alternatives=1,
        verbatim_transcripts=True,
    )
    asr_service.offline_recognize.assert_called_once_with(
        b"audio bytes",
        client.RecognitionConfig.return_value,
    )
    auth.channel.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_nvidia_nim_transcriber_closes_channel_when_recognition_fails(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"audio bytes")
    transcriber = NvidiaNimTranscriber(
        model="openai/whisper-large-v3",
        api_key="test-nim-key",
    )
    riva, client, asr_service, auth = _fake_riva_client("")
    asr_service.offline_recognize.side_effect = RuntimeError("recognition failed")

    with (
        patch.dict(
            "sys.modules",
            {"riva": riva, "riva.client": client},
        ),
        pytest.raises(RuntimeError, match="recognition failed"),
    ):
        await transcriber.transcribe(wav)

    auth.channel.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_nvidia_nim_transcriber_validates_key_and_model_before_import(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"audio")

    with pytest.raises(ValueError, match="non-empty"):
        await NvidiaNimTranscriber(
            model="openai/whisper-large-v3",
            api_key="",
        ).transcribe(wav)

    with pytest.raises(ValueError, match="No NVIDIA NIM config"):
        await NvidiaNimTranscriber(
            model="unknown/model",
            api_key="key",
        ).transcribe(wav)


@pytest.mark.asyncio
async def test_nvidia_nim_transcriber_close_is_terminal(tmp_path: Path) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"audio")
    transcriber = NvidiaNimTranscriber(
        model="openai/whisper-large-v3",
        api_key="key",
    )

    await transcriber.close()
    await transcriber.close()

    with pytest.raises(RuntimeError, match="closed"):
        await transcriber.transcribe(wav)


@pytest.mark.asyncio
async def test_nvidia_nim_transcriber_close_waits_for_active_work(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"audio")
    transcriber = NvidiaNimTranscriber(
        model="openai/whisper-large-v3",
        api_key="secret-key",
    )
    started = Event()
    release = Event()

    def blocking_transcribe(_file_path: Path) -> str:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release active transcription")
        return "finished"

    close_task: asyncio.Task[None] | None = None
    with patch.object(
        transcriber,
        "_transcribe_sync",
        side_effect=blocking_transcribe,
    ):
        transcribe_task = asyncio.create_task(transcriber.transcribe(wav))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            close_task = asyncio.create_task(transcriber.close())
            await asyncio.sleep(0)

            assert not close_task.done()
        finally:
            release.set()
            transcript = await transcribe_task
            if close_task is not None:
                await close_task

    assert transcript == "finished"
    assert transcriber._key == ""
    with pytest.raises(RuntimeError, match="closed"):
        await transcriber.transcribe(wav)


@pytest.mark.asyncio
async def test_cancelled_nim_transcription_keeps_key_until_thread_exits(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"audio")
    transcriber = NvidiaNimTranscriber(
        model="openai/whisper-large-v3",
        api_key="secret-key",
    )
    started = Event()
    release = Event()
    observed_keys: list[str] = []

    def blocking_transcribe(_file_path: Path) -> str:
        observed_keys.append(transcriber._key)
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release active transcription")
        observed_keys.append(transcriber._key)
        return "finished"

    close_task: asyncio.Task[None] | None = None
    with patch.object(
        transcriber,
        "_transcribe_sync",
        side_effect=blocking_transcribe,
    ):
        transcribe_task = asyncio.create_task(transcriber.transcribe(wav))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            transcribe_task.cancel()
            await asyncio.sleep(0)
            close_task = asyncio.create_task(transcriber.close())
            await asyncio.sleep(0)

            assert not transcribe_task.done()
            assert not close_task.done()
            assert transcriber._key == "secret-key"
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await transcribe_task
        assert close_task is not None
        await close_task

    assert observed_keys == ["secret-key", "secret-key"]
    assert transcriber._key == ""
    with pytest.raises(RuntimeError, match="closed"):
        await transcriber.transcribe(wav)


def test_nim_asr_model_map_entries_are_real_function_ids() -> None:
    for function_id, language_code in _NIM_ASR_MODEL_MAP.values():
        assert function_id
        assert function_id.strip().lower() != "none"
        parts = function_id.split("-")
        assert len(parts) == 5
        assert all(parts)
        assert language_code is not None
