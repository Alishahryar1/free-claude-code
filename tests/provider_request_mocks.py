"""Shared MagicMock request objects for OpenAI-compatible provider tests."""

from unittest.mock import MagicMock


def make_openai_compat_stream_request(
    *, model: str = "test-model", stream: bool = True
) -> MagicMock:
    """Minimal request stub matching :meth:`OpenAIChatProvider._chat._build_request_body` needs."""
    req = MagicMock()
    req.model = model
    req.stream = stream
    req.messages = []
    req.system = None
    req.tools = None
    req.tool_choice = None
    req.metadata = None
    req.max_tokens = 4096
    req.temperature = None
    req.top_p = None
    req.top_k = None
    req.stop_sequences = None
    req.extra_body = None
    req.thinking = None
    return req
