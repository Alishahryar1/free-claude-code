"""OpenAI-compatible API base URL policy."""


def openai_v1_base_url(base_url: str) -> str:
    """Return the canonical ``/v1`` API base for a server root or API base."""
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"
