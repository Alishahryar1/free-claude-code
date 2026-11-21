"""A provider with a fixed API key never resends a request it rejected for auth.

Refreshing a fixed snapshot returns the same credentials, so a retry after a
401 or 403 would only repeat the rejected request.
"""

from collections.abc import Callable

import httpx
import httpx2
import pytest

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.providers.anthropic.provider import AnthropicProvider
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.gemini import GeminiProvider
from tests.providers import test_anthropic_provider, test_custom_provider
from tests.providers.support import (
    MockResponse,
    immediate_admission,
    make_provider_config,
    mock_response,
)

REQUEST = MessagesRequest(
    model="m", max_tokens=64, messages=[{"role": "user", "content": "Hello"}]
)
ERROR = {"type": "error", "error": {"type": "authentication_error", "message": "no"}}


def _anthropic(reply: Callable[[httpx.Request | httpx2.Request], MockResponse]):
    async def native_reply(request: httpx.Request) -> httpx.Response:
        return await mock_response(reply(request), httpx.Response)

    return test_anthropic_provider.provider(native_reply, max_attempts=3)


def _gemini(reply: Callable[[httpx.Request | httpx2.Request], MockResponse]):
    async def native_reply(request: httpx.Request) -> httpx.Response:
        return await mock_response(reply(request), httpx.Response)

    return GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=3),
        transport=httpx.MockTransport(native_reply),
    )


def _custom(api_format: str):
    def build(
        reply: Callable[
            [httpx.Request | httpx2.Request], httpx.Response | httpx2.Response
        ],
    ):
        sdk = api_format != "anthropic_messages"
        return test_custom_provider.provider(
            test_custom_provider.definition(api_format=api_format, api_key="key"),
            openai_handler=reply if sdk else None,
            messages_handler=None if sdk else reply,
        )

    return build


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    ("build", "response"),
    [
        pytest.param(_anthropic, httpx.Response, id="anthropic"),
        pytest.param(_gemini, httpx.Response, id="gemini"),
        pytest.param(_custom("openai_chat"), httpx2.Response, id="custom-chat"),
        pytest.param(
            _custom("openai_responses"), httpx2.Response, id="custom-responses"
        ),
        pytest.param(
            _custom("anthropic_messages"), httpx.Response, id="custom-messages"
        ),
    ],
)
async def test_authentication_rejection_is_sent_once(
    build: Callable[
        [Callable[[httpx.Request | httpx2.Request], MockResponse]],
        AnthropicProvider | GeminiProvider | CustomProvider,
    ],
    response: type[httpx.Response] | type[httpx2.Response],
    status: int,
):
    generations = 0

    def reply(request: httpx.Request | httpx2.Request):
        nonlocal generations
        if request.method == "POST":
            generations += 1
        return response(status, json=ERROR)

    provider = build(reply)
    try:
        with (
            bind_stream_delivery(StreamDeliveryState()),
            pytest.raises(ExecutionFailure) as caught,
        ):
            _ = [
                frame
                async for chunk in provider.stream_messages(REQUEST)
                for frame in chunk.frames
            ]
    finally:
        await provider.cleanup()

    assert caught.value.status_code == status
    assert generations == 1
