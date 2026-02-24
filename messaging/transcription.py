"""Voice note transcription for messaging platforms.

Supports two providers:
- "whisper": Hugging Face transformers Whisper pipeline (offline, free)
- "nvidia_riva": NVIDIA RIVA ASR (requires RIVA server)
"""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings

# Max file size in bytes (25 MB)
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024

# Short model names -> full Hugging Face model IDs
_MODEL_MAP: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}

# Lazy-loaded pipelines: (model_id, device) -> pipeline
_pipeline_cache: dict[tuple[str, str], Any] = {}


def _resolve_model_id(whisper_model: str) -> str:
    """Resolve short name to full Hugging Face model ID."""
    return _MODEL_MAP.get(whisper_model, whisper_model)


def _get_pipeline(model_id: str, device: str) -> Any:
    """Lazy-load transformers Whisper pipeline. Raises ImportError if not installed."""
    global _pipeline_cache
    if device not in ("cpu", "cuda"):
        raise ValueError(f"whisper_device must be 'cpu' or 'cuda', got {device!r}")
    cache_key = (model_id, device)
    if cache_key not in _pipeline_cache:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            token = get_settings().hf_token
            if token:
                os.environ["HF_TOKEN"] = token

            use_cuda = device == "cuda" and torch.cuda.is_available()
            pipe_device = "cuda:0" if use_cuda else "cpu"
            model_dtype = torch.float16 if use_cuda else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=model_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            model = model.to(pipe_device)
            processor = AutoProcessor.from_pretrained(model_id)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=pipe_device,
            )
            _pipeline_cache[cache_key] = pipe
            logger.debug(
                f"Loaded Whisper pipeline: model={model_id} device={pipe_device}"
            )
        except ImportError as e:
            raise ImportError(
                "Voice notes require the voice extra. Install with: uv sync --extra voice"
            ) from e
    return _pipeline_cache[cache_key]


def transcribe_audio(
    file_path: Path,
    mime_type: str,
    *,
    whisper_model: str = "base",
    whisper_device: str = "cpu",
) -> str:
    """
    Transcribe audio file to text.

    Supports two providers (configured via TRANSCRIPTION_PROVIDER):
    - "whisper": Hugging Face transformers Whisper pipeline (offline, free)
    - "nvidia_riva": NVIDIA RIVA ASR (requires RIVA server)

    Args:
        file_path: Path to audio file (OGG, MP3, MP4, WAV, M4A supported)
        mime_type: MIME type of the audio (e.g. "audio/ogg")
        whisper_model: Model ID (e.g. "openai/whisper-base") or short name (only for whisper)
        whisper_device: "cpu" | "cuda" (only for whisper)

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file too large or invalid provider
        ImportError: If voice extra not installed (for whisper)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    size = file_path.stat().st_size
    if size > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Audio file too large ({size} bytes). Max {MAX_AUDIO_SIZE_BYTES} bytes."
        )

    # Get provider from settings (supports backward compatibility with direct args)
    settings = get_settings()
    provider = settings.transcription_provider

    if provider == "nvidia_riva":
        return _transcribe_riva(file_path)
    else:
        return _transcribe_local(file_path, whisper_model, whisper_device)


# Whisper expects 16 kHz sample rate
_WHISPER_SAMPLE_RATE = 16000


def _load_audio(file_path: Path) -> dict[str, Any]:
    """Load audio file to waveform dict. No ffmpeg required."""
    import librosa

    waveform, sr = librosa.load(str(file_path), sr=_WHISPER_SAMPLE_RATE, mono=True)
    return {"array": waveform, "sampling_rate": sr}


def _transcribe_local(file_path: Path, whisper_model: str, whisper_device: str) -> str:
    """Transcribe using transformers Whisper pipeline."""
    model_id = _resolve_model_id(whisper_model)
    pipe = _get_pipeline(model_id, whisper_device)
    audio = _load_audio(file_path)
    result = pipe(audio, generate_kwargs={"language": "en", "task": "transcribe"})
    text = result.get("text", "") or ""
    if isinstance(text, list):
        text = " ".join(text) if text else ""
    result_text = text.strip()
    logger.debug(f"Local transcription: {len(result_text)} chars")
    return result_text or "(no speech detected)"


def _transcribe_riva(file_path: Path) -> str:
    """Transcribe using NVIDIA RIVA ASR."""
    settings = get_settings()
    server = settings.nvidia_riva_server

    try:
        import riva.client
    except ImportError as e:
        raise ImportError(
            "NVIDIA RIVA transcription requires the riva client. "
            "Install with: pip install nvidia-riva-client"
        ) from e

    # Create auth with SSL for non-local servers
    use_ssl = not server.startswith("localhost:")
    auth = riva.client.Auth(uri=server, use_ssl=use_ssl)

    # Read audio file
    with open(file_path, "rb") as fh:
        audio_data = fh.read()

    # Create ASR service and configure recognition
    asr_service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    # Perform offline recognition
    response = asr_service.offline_recognize(audio_data, config)

    # Extract transcription
    results = response.results
    if not results:
        return "(no speech detected)"

    transcripts = [
        result.alternatives[0].transcript
        for result in results
        if result.alternatives
    ]

    result_text = " ".join(transcripts).strip()
    logger.debug(f"RIVA transcription: {len(result_text)} chars")
    return result_text or "(no speech detected)"
