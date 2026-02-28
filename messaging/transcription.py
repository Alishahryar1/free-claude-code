"""Voice note transcription for messaging platforms.

Supports:
- Local Whisper (cpu/cuda): Hugging Face transformers pipeline
- NVIDIA NIM: NVIDIA NIM Whisper/Parkeet
"""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings

# Max file size in bytes (25 MB)
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024

# NVIDIA NIM Whisper model mapping: (function_id, language_code)
_NIM_MODEL_MAP: dict[str, tuple[str, str]] = {
    "nvidia/parakeet-ctc-0.6b-zh-tw": ("8473f56d-51ef-473c-bb26-efd4f5def2bf", "zh-TW"),
    "nvidia/parakeet-ctc-0.6b-zh-cn": ("9add5ef7-322e-47e0-ad7a-5653fb8d259b", "zh-CN"),
    "nvidia/parakeet-ctc-0.6b-es": ("None", "es-US"),
    "nvidia/parakeet-ctc-0.6b-vi": ("f3dff2bb-99f9-403d-a5f1-f574a757deb0", "vi-VN"),
    "nvidia/parakeet-ctc-1.1b-asr": ("1598d209-5e27-4d3c-8079-4751568b1081", "en-US"),
    "nvidia/parakeet-ctc-0.6b-asr": ("d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965", "en-US"),
    "nvidia/parakeet-1.1b-rnnt-multilingual-asr": (
        "71203149-d3b7-4460-8231-1be2543a1fca",
        "",
    ),
    "openai/whisper-large-v3": ("b702f636-f60c-4a3d-a6f4-f3568c13bd7d", "multi"),
}

# Short model names -> full Hugging Face model IDs (for local Whisper)
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
                "Local Whisper requires the voice_local extra. Install with: uv sync --extra voice_local"
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

    Supports:
    - whisper_device="cpu"/"cuda": local Whisper (requires voice_local extra)
    - whisper_device="nvidia_nim": NVIDIA NIM Whisper API (requires voice extra)

    Args:
        file_path: Path to audio file (OGG, MP3, MP4, WAV, M4A supported)
        mime_type: MIME type of the audio (e.g. "audio/ogg")
        whisper_model: Model ID or short name (local) or NVIDIA NIM model
        whisper_device: "cpu" | "cuda" | "nvidia_nim" (defaults to WHISPER_DEVICE env var)

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file too large
        ImportError: If voice_local extra not installed (for local Whisper)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    size = file_path.stat().st_size
    if size > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Audio file too large ({size} bytes). Max {MAX_AUDIO_SIZE_BYTES} bytes."
        )

    # Use settings values (from WHISPER_DEVICE and WHISPER_MODEL env vars)
    # Function args are kept for backward compatibility but settings take precedence
    settings = get_settings()
    device = settings.whisper_device
    model = settings.whisper_model

    if device == "nvidia_nim":
        return _transcribe_nim(file_path, model)
    else:
        return _transcribe_local(file_path, model, device)


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


def _transcribe_nim(file_path: Path, model: str) -> str:
    """Transcribe using NVIDIA NIM Whisper API via Riva gRPC client."""
    try:
        import riva.client
    except ImportError as e:
        raise ImportError(
            "NVIDIA NIM transcription requires the voice extra. "
            "Install with: uv sync --extra voice"
        ) from e

    settings = get_settings()
    api_key = settings.nvidia_nim_api_key

    if not api_key:
        raise ValueError(
            "NVIDIA_NIM_API_KEY is required for nvidia_nim transcription. "
            "Set it in your .env file."
        )

    # Look up function ID and language code from model mapping
    model_config = _NIM_MODEL_MAP.get(model)
    if not model_config:
        raise ValueError(
            f"No NVIDIA NIM config found for model: {model}. "
            f"Supported models: {', '.join(_NIM_MODEL_MAP.keys())}"
        )
    function_id, language_code = model_config

    # Riva server configuration
    server = "grpc.nvcf.nvidia.com:443"

    # Auth with SSL and metadata
    auth = riva.client.Auth(
        use_ssl=True,
        uri=server,
        metadata_args=[
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ],
    )

    asr_service = riva.client.ASRService(auth)

    # Configure recognition - language_code from model config
    config = riva.client.RecognitionConfig(
        language_code=language_code,
        max_alternatives=1,
        verbatim_transcripts=True,
    )

    # Read audio file
    with open(file_path, "rb") as f:
        data = f.read()

    # Perform offline recognition
    response = asr_service.offline_recognize(data, config)

    # Extract text from response - use getattr for safe attribute access
    transcript = ""
    results = getattr(response, "results", None)
    if results and results[0].alternatives:
        transcript = results[0].alternatives[0].transcript

    logger.debug(f"NIM transcription: {len(transcript)} chars")
    return transcript or "(no speech detected)"
