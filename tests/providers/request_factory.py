"""Concrete Anthropic request factory for provider tests."""

from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest


def make_messages_request(
    model: str = "test-model", **overrides: Any
) -> MessagesRequest:
    """Build a real Messages request with provider-test defaults."""
    data: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "system": "System prompt",
        "stop_sequences": None,
        "tools": [],
        "extra_body": {},
        "thinking": {"enabled": True},
    }
    data.update(overrides)
    return MessagesRequest.model_validate(data)
