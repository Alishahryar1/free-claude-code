"""What a rejection teaches a provider's memories, even when no attempt remains.

A rejection is upstream evidence: the memory it updates outlives the request
whether or not admission allows the corrected retry. Groq's reasoning
vocabulary is pinned the same way in ``test_groq_reasoning``.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply
from tests.providers.test_openai_chat_output_cap import (
    _BadRequest,
    _ok,
    _send,
    _sent_caps,
)
from tests.providers.test_readable_history_negotiation import readable_history

REJECTED = (400, {"type": "invalid_request_error", "message": "Rejected"})


@pytest.mark.asyncio
async def test_output_cap_is_learned_even_when_no_attempt_remains():
    provider = GroqProvider(
        make_provider_config(api_key="key", base_url=GROQ_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=1),
    )
    error = _BadRequest("max_completion_tokens must be less than or equal to 40960")
    create = AsyncMock(side_effect=[error, _ok()])

    with patch.object(provider._client.chat.completions, "create", create):
        with pytest.raises(ExecutionFailure):
            await _send(provider)
        await _send(provider)

    assert _sent_caps(create) == [64000, 40960]


def _chat(cache: HistoryModeCache, max_attempts: int):
    return lambda: OpenRouterProvider(
        make_provider_config(api_key="a", base_url="https://provider.invalid/v1"),
        admission=immediate_admission(max_attempts=max_attempts),
        history_cache=cache,
    )


def _thoughts(body) -> tuple[bool, bool]:
    serialized = json.dumps(body)
    return "Old" in serialized, "New" in serialized


@pytest.mark.asyncio
async def test_rejected_cached_history_mode_is_forgotten_even_when_no_attempt_remains():
    cache = HistoryModeCache()

    # Two rejections teach the cache that this upstream wants no thoughts.
    async with _harness(
        "chat",
        lambda bodies: REJECTED if len(bodies) < 3 else (200, _events_for("chat")),
        chat_provider_factory=_chat(cache, 5),
    ) as (send, bodies, _):
        await _saved_reply(send("messages", readable_history()), "messages")
    assert _thoughts(bodies[-1]) == (False, False)

    # The cached mode is rejected on the only attempt this request has.
    async with _harness(
        "chat", lambda _bodies: REJECTED, chat_provider_factory=_chat(cache, 1)
    ) as (send, bodies, _):
        with pytest.raises(ExecutionFailure):
            await _saved_reply(send("messages", readable_history()), "messages")
    assert [_thoughts(body) for body in bodies] == [(False, False)]

    # The next request starts from every thought again.
    async with _harness("chat", chat_provider_factory=_chat(cache, 1)) as (
        send,
        bodies,
        _,
    ):
        await _saved_reply(send("messages", readable_history()), "messages")
    assert [_thoughts(body) for body in bodies] == [(True, True)]
