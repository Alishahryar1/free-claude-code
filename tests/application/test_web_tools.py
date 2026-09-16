"""Application web-tool behavior through its explicit outbound capability."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import ModelRouter
from free_claude_code.application.web_tools.ports import WebFetchEgressPolicy
from free_claude_code.application.web_tools.service import WebToolService
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import Message, MessagesRequest, Tool
from free_claude_code.core.web_tools import WebFetchResult
from tests.web_tools_support import StubWebToolsClient


async def _no_provider(_provider_id):
    pytest.fail("This request must not resolve a provider")


def _service(settings, client, *, token_counter=lambda *_: 5):
    return WebToolService(
        settings=settings,
        client=client,
        executor=ProviderExecutor(_no_provider, progress_timeout_seconds=30),
        token_counter=token_counter,
    )


def _request(tool, *, typed=True):
    return MessagesRequest(
        model="claude-sonnet",
        messages=[Message(role="user", content="https://example.com/")],
        tools=[
            Tool(
                name=tool,
                type=(
                    "web_fetch_20250910"
                    if tool == "web_fetch"
                    else "web_search_20250305"
                )
                if typed
                else None,
            )
        ],
        tool_choice={"type": "tool", "name": tool},
    )


async def _drain(body: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in body])


def test_ordinary_client_tool_needs_no_web_or_token_work():
    settings = Settings()
    service = _service(
        settings,
        StubWebToolsClient(),
        token_counter=lambda *_: pytest.fail("Unhandled request counted tokens"),
    )
    routed = ModelRouter(settings).resolve_messages_request(
        _request("web_search", typed=False)
    )
    assert service.try_stream_messages(routed, request_id="req_unhandled") is None


def test_disabled_tool_fails_during_preparation():
    settings = Settings().model_copy(update={"enable_web_server_tools": False})
    service = _service(settings, StubWebToolsClient())
    routed = ModelRouter(settings).resolve_messages_request(_request("web_fetch"))
    with pytest.raises(InvalidRequestError, match="disabled"):
        service.try_stream_messages(routed, request_id="req_disabled")


@pytest.mark.asyncio
async def test_concurrent_requests_keep_their_own_egress_policy():
    entered, release = asyncio.Event(), asyncio.Event()

    class Client(StubWebToolsClient):
        def __init__(self):
            self.policies = []

        async def fetch(
            self, url: str, *, egress: WebFetchEgressPolicy
        ) -> WebFetchResult:
            self.policies.append(egress)
            if len(self.policies) == 2:
                entered.set()
            await release.wait()
            return WebFetchResult(
                url=url, title="page", media_type="text/plain", data="text"
            )

    client = Client()
    settings = [
        Settings().model_copy(
            update={
                "web_fetch_allow_private_networks": False,
                "web_fetch_allowed_schemes": "https",
            }
        ),
        Settings().model_copy(
            update={
                "web_fetch_allow_private_networks": True,
                "web_fetch_allowed_schemes": "http, https",
            }
        ),
    ]
    tasks = []
    try:
        for index, snapshot in enumerate(settings):
            service = _service(snapshot, client)
            body = service.try_stream_messages(
                ModelRouter(snapshot).resolve_messages_request(_request("web_fetch")),
                request_id=f"req_{index}",
            )
            assert body is not None
            tasks.append(asyncio.create_task(_drain(body)))
        await asyncio.wait_for(entered.wait(), 1)
        assert client.policies == [
            WebFetchEgressPolicy(False, frozenset({"https"})),
            WebFetchEgressPolicy(True, frozenset({"http", "https"})),
        ]
        release.set()
        outputs = await asyncio.gather(*tasks)
        assert all("web_fetch_tool_result" in output for output in outputs)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
