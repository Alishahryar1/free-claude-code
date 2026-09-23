"""NVIDIA NIM / Riva offline ASR for voice notes (provider-owned transport)."""

import asyncio
from pathlib import Path

from loguru import logger

# NVIDIA NIM Whisper model mapping: (function_id, language_code)
_NIM_ASR_MODEL_MAP: dict[str, tuple[str, str]] = {
    "nvidia/parakeet-ctc-0.6b-zh-tw": ("8473f56d-51ef-473c-bb26-efd4f5def2bf", "zh-TW"),
    "nvidia/parakeet-ctc-0.6b-zh-cn": ("9add5ef7-322e-47e0-ad7a-5653fb8d259b", "zh-CN"),
    # function-id from NVIDIA NIM API docs (parakeet-ctc-0.6b-es).
    "nvidia/parakeet-ctc-0.6b-es": ("a9eeee8f-b509-4712-b19d-194361fa5f31", "es-US"),
    "nvidia/parakeet-ctc-0.6b-vi": ("f3dff2bb-99f9-403d-a5f1-f574a757deb0", "vi-VN"),
    "nvidia/parakeet-ctc-1.1b-asr": ("1598d209-5e27-4d3c-8079-4751568b1081", "en-US"),
    "nvidia/parakeet-ctc-0.6b-asr": ("d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965", "en-US"),
    "nvidia/parakeet-1.1b-rnnt-multilingual-asr": (
        "71203149-d3b7-4460-8231-1be2543a1fca",
        "",
    ),
    "openai/whisper-large-v3": ("b702f636-f60c-4a3d-a6f4-f3568c13bd7d", "multi"),
}

_RIVA_SERVER = "grpc.nvcf.nvidia.com:443"


class NvidiaNimTranscriber:
    """Own configured NVIDIA NIM / Riva transcription."""

    def __init__(self, *, model: str, api_key: str) -> None:
        self._model = model
        self._key = api_key.strip()
        self._lock = asyncio.Lock()
        self._closed = False

    async def transcribe(self, file_path: Path) -> str:
        """Transcribe one audio file without blocking the event loop."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("NVIDIA NIM transcriber is closed.")
            worker = asyncio.create_task(
                asyncio.to_thread(self._transcribe_sync, file_path)
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _wait_for_thread_exit(worker)
                raise

    async def close(self) -> None:
        """Close this stateless adapter to future work."""
        self._closed = True
        async with self._lock:
            self._key = ""

    def _transcribe_sync(self, file_path: Path) -> str:
        if not self._key:
            raise ValueError(
                "NVIDIA NIM transcription requires a non-empty "
                "nvidia_nim_api_key (configure NVIDIA_NIM_API_KEY)."
            )
        model_config = _NIM_ASR_MODEL_MAP.get(self._model)
        if model_config is None:
            raise ValueError(
                f"No NVIDIA NIM config found for model: {self._model}. "
                f"Supported models: {', '.join(_NIM_ASR_MODEL_MAP)}"
            )
        function_id, language_code = model_config
        try:
            import riva.client
        except ImportError as exc:
            raise ImportError(
                "NVIDIA NIM transcription requires the voice extra. "
                "Install with: uv sync --extra voice"
            ) from exc

        auth = riva.client.Auth(
            use_ssl=True,
            uri=_RIVA_SERVER,
            metadata_args=[
                ["function-id", function_id],
                ["authorization", f"Bearer {self._key}"],
            ],
        )
        try:
            asr_service = riva.client.ASRService(auth)
            config = riva.client.RecognitionConfig(
                language_code=language_code,
                max_alternatives=1,
                verbatim_transcripts=True,
            )
            data = file_path.read_bytes()
            response = asr_service.offline_recognize(data, config)

            transcript = ""
            results = getattr(response, "results", None)
            if results and results[0].alternatives:
                transcript = results[0].alternatives[0].transcript
            logger.debug("NIM transcription: {} chars", len(transcript))
            return transcript or "(no speech detected)"
        finally:
            auth.channel.close()


async def _wait_for_thread_exit(worker: asyncio.Task[str]) -> None:
    """Wait through repeated caller cancellation without cancelling thread work."""
    while not worker.done():
        try:
            await asyncio.shield(asyncio.wait((worker,)))
        except asyncio.CancelledError:
            continue
    if not worker.cancelled():
        worker.exception()
