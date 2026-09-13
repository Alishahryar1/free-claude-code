"""Connected-account dispatch preserves both ingresses across all three egresses."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import httpx2
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.endpoint import HttpEndpoint
from free_claude_code.providers.github_copilot.auth import CopilotAuthManager
from free_claude_code.providers.github_copilot.provider import GitHubCopilotProvider
from free_claude_code.providers.github_copilot.types import (
    CopilotEgress,
    CopilotEndpoint,
    CopilotModel,
)
from free_claude_code.providers.openai_chat import OpenAIChatTransport
from free_claude_code.providers.openai_responses.transport import (
    OpenAIResponsesTransport,
)
from tests.providers.copilot_support import FakeRuntime, FakeSession
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_anthropic_messages_transport import _events as messages_events
from tests.providers.test_anthropic_messages_transport import _sse as messages_sse
from tests.providers.test_openai_responses_transport import (
    _completed_event,
    _text_delta,
)
from tests.providers.test_openai_responses_transport import (
    _sse as responses_sse,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
@pytest.mark.parametrize("responses", [False, True])
async def test_conversion_runs_once_under_current_account_lease(
    tmp_path, egress, responses
):
    harness = Harness(tmp_path, egress)
    cls, name = {
        (CopilotEgress.CHAT, False): (OpenAIChatTransport, "_build_request_body"),
        (CopilotEgress.CHAT, True): (
            OpenAIChatTransport,
            "_build_responses_request_body",
        ),
        (CopilotEgress.RESPONSES, False): (
            OpenAIResponsesTransport,
            "_build_messages_body",
        ),
        (CopilotEgress.RESPONSES, True): (
            OpenAIResponsesTransport,
            "_build_native_body",
        ),
        (CopilotEgress.MESSAGES, False): (AnthropicMessagesTransport, "_messages_body"),
        (CopilotEgress.MESSAGES, True): (AnthropicMessagesTransport, "_responses_body"),
    }[egress, responses]
    original = getattr(cls, name)

    def build(transport, *args, **kwargs):
        assert harness.runtime.session_calls == 1
        assert not harness.runtime.sessions[0].closed
        assert harness.provider._active == 1
        assert harness.seen == []
        return original(transport, *args, **kwargs)

    try:
        with patch.object(cls, name, autospec=True, side_effect=build) as conversion:
            assert "ok" in await collect(harness.stream(responses))
        assert conversion.call_count == len(harness.seen) == 1
        assert harness.provider._active == 0
    finally:
        await harness.close()


class Wire(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False
        self.close_entered = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.close_gate.set()
        self.read_gate = asyncio.Event()
        self.read_gate.set()
        self.read_entered = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.read_entered.set()
        await self.read_gate.wait()
        yield self.content

    async def aclose(self) -> None:
        self.close_entered.set()
        await self.close_gate.wait()
        self.closed = True


class Runtime(FakeRuntime):
    def __init__(self, egress: CopilotEgress) -> None:
        super().__init__()
        self.egress = egress
        # Names deliberately do not identify the transport family.
        self.name = (
            "gpt-looking" if egress is CopilotEgress.MESSAGES else "claude-looking"
        )
        self.available = (
            CopilotModel(
                ProviderModelInfo(model_id=self.name, max_output_tokens=4096),
                MessagesModelCapabilities(
                    max_output_tokens=4096,
                    adaptive_thinking="optional",
                    supports_output_effort=True,
                    supported_efforts=("low", "high"),
                ),
                ("none", "low", "high"),
                "low",
            ),
        )

    async def session(self, model_id: str) -> FakeSession:
        session = await super().session(model_id)
        session.value = CopilotEndpoint(
            self.egress,
            HttpEndpoint(
                "https://copilot.invalid",
                {"X-Session": f"session-{model_id}"},
                "account-secret",
            ),
        )
        return session


class Harness:
    def __init__(self, tmp_path: Path, egress: CopilotEgress) -> None:
        self.runtime = Runtime(egress)
        state = tmp_path / "copilot.json"
        state.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "revision": 3,
                    "enabled": True,
                    "identity": {"host": "github.com", "login": "octocat"},
                }
            ),
            encoding="utf-8",
        )
        self.auth = CopilotAuthManager(
            state_path=state, runtime_factory=lambda: self.runtime
        )
        self.seen: list[httpx.Request | httpx2.Request] = []
        self.wires: list[Wire] = []
        self.statuses: list[int] = []
        self.block_close = False
        self.block_read = False
        self.messages_content = messages_sse(*messages_events("ok"))
        self.responses_content = responses_sse(
            _text_delta("ok"), _completed_event()
        ).encode()
        self.chat_content = (
            "data: "
            + json.dumps(
                {
                    "id": "chat-id",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "upstream",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()
        self.provider = GitHubCopilotProvider(
            make_provider_config(
                None, "https://configuration.invalid", http_read_timeout=3
            ),
            auth=self.auth,
            admission=immediate_admission(max_attempts=2),
            messages_transport=httpx.MockTransport(self.messages),
            openai_transport=httpx2.MockTransport(self.openai),
        )

    def wire(self, content: bytes) -> Wire:
        wire = Wire(content)
        if self.block_close:
            wire.close_gate.clear()
        if self.block_read:
            wire.read_gate.clear()
        self.wires.append(wire)
        return wire

    def messages(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return httpx.Response(
                status,
                headers={"set-cookie": "inherited=secret; Path=/"},
                json={"error": {"message": "expired"}},
            )
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "set-cookie": "inherited=secret; Path=/",
            },
            stream=self.wire(self.messages_content),
        )

    def openai(self, request: httpx2.Request) -> httpx2.Response:
        self.seen.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return httpx2.Response(
                status,
                headers={"set-cookie": "inherited=secret; Path=/"},
                json={"error": {"message": "expired"}},
            )
        content = (
            self.responses_content
            if request.url.path.endswith("/responses")
            else self.chat_content
        )
        return httpx2.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "set-cookie": "inherited=secret; Path=/",
            },
            stream=self.wire(content),
        )

    def stream(
        self,
        responses: bool,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        history: bool = False,
    ) -> AsyncIterator[str]:
        request = _request(responses, self.runtime.name, history=history)
        if isinstance(request, MessagesRequest):
            return self.provider.stream_messages(
                request, response_model="public-alias", reasoning=reasoning
            )
        return self.provider.stream_responses(
            request, response_model="public-alias", reasoning=reasoning
        )

    async def close(self) -> None:
        await self.provider.cleanup()
        assert self.runtime.close_calls == 0
        await self.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
@pytest.mark.parametrize("supports_off", [False, True])
async def test_classifier_uses_current_lease_controls_and_clears_raw_hints(
    tmp_path, egress, supports_off
):
    harness = Harness(tmp_path, egress)
    current = harness.runtime.available[0]
    efforts = ("none", "high") if supports_off else ("high",)
    harness.runtime.available = (
        replace(
            current,
            supported_efforts=efforts,
            info=replace(
                current.info,
                reasoning_capability=ReasoningCapability.OPTIONAL
                if supports_off
                else ReasoningCapability.UNKNOWN,
            ),
        ),
    )
    request = MessagesRequest.model_validate(
        {
            "model": harness.runtime.name,
            "messages": [{"role": "user", "content": "classify"}],
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "xhigh"},
        }
    )
    try:
        output = [
            event
            async for event in harness.provider.stream_messages(
                request,
                reasoning=ReasoningPolicy.prefer_off(),
                model_info=ProviderModelInfo(
                    harness.runtime.name, reasoning_capability=ReasoningCapability.NONE
                ),
            )
        ]
        assert text_content(parse_sse_text("".join(output))) == "ok"
        body = json.loads(harness.seen[0].content)
        if egress is CopilotEgress.MESSAGES:
            assert body["thinking"] == {"type": "disabled"}
            assert body["max_tokens"] == (64 if supports_off else 4096)
        elif egress is CopilotEgress.RESPONSES:
            assert body.get("reasoning") == (
                {"effort": "none"} if supports_off else None
            )
            assert body.get("max_output_tokens") == (64 if supports_off else None)
        else:
            assert body.get("reasoning_effort") == ("none" if supports_off else None)
            assert body.get("max_tokens") == (64 if supports_off else None)
        assert request.thinking is not None and request.thinking.budget_tokens == 1024
        assert all(wire.closed for wire in harness.wires)
    finally:
        await harness.close()


def _request(
    responses: bool, model: str, *, history: bool = False
) -> MessagesRequest | OpenAIResponsesRequest:
    if responses:
        return OpenAIResponsesRequest.model_validate(
            {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "hello"},
                            {
                                "type": "input_image",
                                "image_url": "https://images.invalid/picture.png",
                            },
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-owned",
                        "name": "read_file",
                        "arguments": '{"path":"file.txt"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call-owned",
                        "output": "the file",
                    },
                ]
                if history
                else "hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    }
                ]
                if history
                else None,
            }
        )
    return MessagesRequest.model_validate(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://images.invalid/picture.png",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-owned",
                            "name": "read_file",
                            "input": {"path": "file.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-owned",
                            "content": "the file",
                        }
                    ],
                },
            ]
            if history
            else [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "read_file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ]
            if history
            else None,
        }
    )


async def collect(stream: AsyncIterator[str]) -> str:
    return "".join([event async for event in stream])


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
@pytest.mark.parametrize("responses", [False, True])
async def test_six_cells_use_metadata_and_preserve_text_usage_public_model_and_history(
    tmp_path: Path, egress: CopilotEgress, responses: bool
) -> None:
    harness = Harness(tmp_path, egress)
    try:
        events = parse_sse_text(await collect(harness.stream(responses, history=True)))
        assert len(harness.seen) == 1
        request = harness.seen[0]
        assert (
            request.url.path
            == {
                CopilotEgress.MESSAGES: "/v1/messages",
                CopilotEgress.RESPONSES: "/responses",
                CopilotEgress.CHAT: "/chat/completions",
            }[egress]
        )
        assert request.headers["X-Session"] == f"session-{harness.runtime.name}"
        credential = (
            "x-api-key" if egress is CopilotEgress.MESSAGES else "authorization"
        )
        assert "account-secret" in request.headers[credential]
        body = json.loads(request.content)
        assert body["model"] == harness.runtime.name
        assert body["stream"] is True
        raw = request.content.decode()
        assert raw.count("call-owned") == 2
        assert "https://images.invalid/picture.png" in raw
        assert "file.txt" in raw and "the file" in raw
        assert harness.wires[0].closed
        assert not harness.runtime.sessions[0].closed
        if responses:
            assert events[-1].event == "response.completed"
            assert events[-1].data["response"]["model"] == "public-alias"
            assert events[-1].data["response"]["usage"]["output_tokens"] == 2
            assert "ok" in "".join(event.raw for event in events)
        else:
            assert_anthropic_stream_contract(events)
            assert text_content(events) == "ok"
            assert events[0].data["message"]["model"] == "public-alias"
            assert events[-2].data["usage"]["output_tokens"] == 2
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_auth_refresh_and_later_requests_never_inherit_response_cookies(
    tmp_path: Path, egress: CopilotEgress
) -> None:
    harness = Harness(tmp_path, egress)
    harness.statuses = [401]
    try:
        await collect(harness.stream(True))
        await collect(harness.stream(True))
        assert len(harness.seen) == 3
        assert all("cookie" not in request.headers for request in harness.seen)
        assert harness.runtime.sessions[0].endpoint_calls == 2
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", [CopilotEgress.CHAT, CopilotEgress.RESPONSES])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize(
    "policy,wire_effort",
    [
        (DEFAULT_REASONING_POLICY, None),
        (ReasoningPolicy.on(), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.off(), "none"),
    ],
)
async def test_non_messages_controls_use_advertised_efforts(
    tmp_path: Path,
    egress: CopilotEgress,
    responses: bool,
    policy: ReasoningPolicy,
    wire_effort: str | None,
) -> None:
    harness = Harness(tmp_path, egress)
    try:
        await collect(harness.stream(responses, reasoning=policy))
        body = json.loads(harness.seen[0].content)
        if egress is CopilotEgress.CHAT:
            assert body.get("reasoning_effort") == wire_effort
        else:
            assert body.get("reasoning", {}).get("effort") == wire_effort
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", [CopilotEgress.CHAT, CopilotEgress.RESPONSES])
@pytest.mark.parametrize(
    "policy",
    [
        ReasoningPolicy.on(budget_tokens=1024),
        ReasoningPolicy.on(effort=ReasoningEffort.MAX),
        ReasoningPolicy.on(),
        ReasoningPolicy.off(),
    ],
)
async def test_unrepresentable_controls_reject_before_inference(
    tmp_path: Path, egress: CopilotEgress, policy: ReasoningPolicy
) -> None:
    harness = Harness(tmp_path, egress)
    harness.runtime.available = (
        replace(
            harness.runtime.available[0],
            supported_efforts=("low",),
            default_effort=None,
        ),
    )
    try:
        with pytest.raises(InvalidRequestError):
            await collect(harness.stream(True, reasoning=policy))
        assert not harness.seen
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_native_messages_uses_capabilities_and_exact_budget(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, CopilotEgress.MESSAGES)
    try:
        request = MessagesRequest(
            model=harness.runtime.name,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=99999,
        )
        await collect(
            harness.provider.stream_messages(
                request, reasoning=ReasoningPolicy.on(budget_tokens=1024)
            )
        )
        body = json.loads(harness.seen[0].content)
        assert body["max_tokens"] == 4096
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_native_reasoning_carrier_replays_only_to_messages_egress(
    tmp_path: Path, egress: CopilotEgress
) -> None:
    harness = Harness(tmp_path, egress)
    block: JsonObject = {
        "type": "thinking",
        "thinking": "private reasoning",
        "signature": "exact-signature",
    }
    carrier = encode_replay(
        ReplayRecord(
            ReplayOrigin(
                "github_copilot/anthropic_messages", "messages", "", "", "prior-model"
            ),
            block,
        )
    )
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": harness.runtime.name,
            "input": [
                {"type": "reasoning", "encrypted_content": carrier},
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )
    try:
        if egress is CopilotEgress.MESSAGES:
            await collect(harness.provider.stream_responses(request))
            body = json.loads(harness.seen[0].content)
            assert body["messages"][0]["content"][0] == block
        else:
            await collect(harness.provider.stream_responses(request))
            body = json.loads(harness.seen[0].content)
            wire = json.dumps(body)
            assert "[Earlier reasoning]" in wire
            assert "private reasoning" in wire
            assert "exact-signature" not in wire
            assert carrier not in wire
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_cancelled_inference_keeps_sdk_lease_until_http_response_closes(
    tmp_path: Path, egress: CopilotEgress
) -> None:
    harness = Harness(tmp_path, egress)
    harness.block_read = harness.block_close = True
    operation = asyncio.create_task(collect(harness.stream(True)))
    try:
        while not harness.wires:
            await asyncio.sleep(0)
        wire = harness.wires[0]
        await wire.read_entered.wait()
        operation.cancel()
        await wire.close_entered.wait()
        disconnect = asyncio.create_task(harness.auth.disconnect())
        cleanup = asyncio.create_task(harness.provider.cleanup())
        await asyncio.sleep(0)
        assert not disconnect.done() and not cleanup.done()
        assert not harness.runtime.sessions[0].closed
        wire.close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await disconnect
        await cleanup
        assert wire.closed and harness.runtime.sessions[0].closed
    finally:
        for wire in harness.wires:
            wire.read_gate.set()
            wire.close_gate.set()
        await asyncio.gather(operation, return_exceptions=True)
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_http_failure_leaves_shared_account_available_and_releases_stream(
    tmp_path: Path, egress: CopilotEgress
) -> None:
    harness = Harness(tmp_path, egress)
    harness.statuses = [400]
    try:
        with pytest.raises(ExecutionFailure):
            await collect(harness.stream(False))
        assert harness.auth.is_connected()
        await collect(harness.stream(False))
        assert len(harness.seen) == 2
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_discovery_is_live_and_generation_cleanup_does_not_disconnect_account(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, CopilotEgress.CHAT)
    first = await harness.provider.list_model_infos()
    harness.runtime.available = (
        replace(
            harness.runtime.available[0],
            info=ProviderModelInfo(model_id="another-model"),
        ),
    )
    second = await harness.provider.list_model_infos()
    assert {model.model_id for model in first} == {harness.runtime.name}
    assert {model.model_id for model in second} == {"another-model"}
    await harness.provider.cleanup()
    assert harness.auth.is_connected()
    assert harness.runtime.close_calls == 0
    await harness.auth.close()


@pytest.mark.asyncio
async def test_auto_is_never_resolved_by_an_inference_turn(tmp_path: Path) -> None:
    harness = Harness(tmp_path, CopilotEgress.CHAT)
    try:
        with pytest.raises(InvalidRequestError, match="concrete"):
            harness.provider.stream_responses(
                OpenAIResponsesRequest(model="auto", input="hi")
            )
        assert not harness.seen and harness.runtime.session_calls == 0
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_repeated_cancellation_drains_physical_response_before_account_release(
    tmp_path: Path, egress: CopilotEgress
) -> None:
    harness = Harness(tmp_path, egress)
    harness.block_read = harness.block_close = True
    operation = asyncio.create_task(collect(harness.stream(True)))
    disconnect: asyncio.Task[object] | None = None
    try:
        while not harness.wires:
            await asyncio.sleep(0)
        wire = harness.wires[0]
        await wire.read_entered.wait()
        operation.cancel()
        await wire.close_entered.wait()
        operation.cancel()
        disconnect = asyncio.create_task(harness.auth.disconnect())
        for _ in range(8):
            await asyncio.sleep(0)
        assert not operation.done(), (
            "cancellation returned before HTTP response cleanup"
        )
        assert not disconnect.done(), "account released while HTTP cleanup was pending"
        wire.close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await disconnect
        assert wire.closed
    finally:
        for wire in harness.wires:
            wire.read_gate.set()
            wire.close_gate.set()
        await asyncio.gather(operation, return_exceptions=True)
        if disconnect is not None:
            await disconnect
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_rejected_credentials_are_replaced_by_the_refreshed_sdk_snapshot(
    tmp_path: Path, egress: CopilotEgress
) -> None:
    harness = Harness(tmp_path, egress)
    try:
        # Prime the endpoint through the public account lease, without inference.
        async with harness.auth.lease(harness.runtime.name):
            pass
        session = harness.runtime.sessions[0]
        session.value = CopilotEndpoint(
            egress,
            HttpEndpoint(
                "https://refreshed.invalid",
                {"X-Session": "fresh-session"},
                "fresh-secret",
            ),
        )
        harness.statuses = [401]
        await collect(harness.stream(True))
        credential = (
            "x-api-key" if egress is CopilotEgress.MESSAGES else "authorization"
        )
        first, second = harness.seen
        assert first.url.host == "copilot.invalid"
        assert "account-secret" in first.headers[credential]
        assert second.url.host == "refreshed.invalid"
        assert "fresh-secret" in second.headers[credential]
        assert second.headers["x-session"] == "fresh-session"
        assert "account-secret" not in str(second.headers)
    finally:
        await harness.close()
