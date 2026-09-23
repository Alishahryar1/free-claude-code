"""Instance-owned local Whisper transcription."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

_MODEL_MAP: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}
_WHISPER_SAMPLE_RATE = 16000


class TranscriptionService:
    """Own one lazily loaded local Whisper pipeline."""

    def __init__(
        self,
        *,
        model: str,
        device: str,
        huggingface_api_key: str | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(
                f"Local Whisper device must be 'cpu' or 'cuda', got {device!r}"
            )
        self._model_id = _MODEL_MAP.get(model, model)
        self._device = device
        self._huggingface_api_key = huggingface_api_key
        self._pipeline: Any | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def transcribe(self, file_path: Path) -> str:
        """Transcribe one audio file without blocking the event loop."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Transcription service is closed.")
            worker = asyncio.create_task(
                asyncio.to_thread(self._transcribe_sync, file_path)
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _wait_for_thread_exit(worker)
                raise

    async def close(self) -> None:
        """Prevent new work and release the owned model pipeline."""
        self._closed = True
        async with self._lock:
            self._pipeline = None
            self._huggingface_api_key = None

    def _transcribe_sync(self, file_path: Path) -> str:
        pipe = self._get_pipeline()
        audio = _load_audio(file_path)
        result = pipe(audio, generate_kwargs={"language": "en", "task": "transcribe"})
        text = result.get("text", "") or ""
        if isinstance(text, list):
            text = " ".join(text) if text else ""
        result_text = text.strip()
        logger.debug("Local transcription: {} chars", len(result_text))
        return result_text or "(no speech detected)"

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise ImportError(
                "Local Whisper requires the voice_local extra. "
                "Install with: uv sync --extra voice_local"
            ) from exc

        token = self._huggingface_api_key
        use_cuda = self._device == "cuda" and torch.cuda.is_available()
        pipeline_device = "cuda:0" if use_cuda else "cpu"
        model_dtype = torch.float16 if use_cuda else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            token=token,
        )
        model = model.to(pipeline_device)
        processor = AutoProcessor.from_pretrained(self._model_id, token=token)
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=pipeline_device,
        )
        logger.debug(
            "Loaded Whisper pipeline: model={} device={}",
            self._model_id,
            pipeline_device,
        )
        return self._pipeline


async def _wait_for_thread_exit(worker: asyncio.Task[str]) -> None:
    """Wait through repeated caller cancellation without cancelling thread work."""
    while not worker.done():
        try:
            await asyncio.shield(asyncio.wait((worker,)))
        except asyncio.CancelledError:
            continue
    if not worker.cancelled():
        worker.exception()


def _load_audio(file_path: Path) -> dict[str, Any]:
    """Load an audio file into the waveform shape expected by Whisper."""
    import librosa

    waveform, sample_rate = librosa.load(
        str(file_path), sr=_WHISPER_SAMPLE_RATE, mono=True
    )
    return {"array": waveform, "sampling_rate": sample_rate}
