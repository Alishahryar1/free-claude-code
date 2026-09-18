"""A web response survives the executor's timeout and resource-release boundaries."""

import asyncio
from contextlib import aclosing, asynccontextmanager

import pytest
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from free_claude_code.api.response_streams import (
    bind_response_lifetime,
    openai_responses_sse_streaming_response,
)
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.responses_execution import ResponsesBinding
from free_claude_code.application.web_tools.responses import ResponsesWebTools
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses.items import message_item
from free_claude_code.core.openai_responses.web_history import prepare_web_history
from free_claude_code.core.sse import parse_sse_text
from tests.api.test_response_streams import _serve
from tests.application.test_execution import _routed_responses_request, _target
from tests.application.test_responses_web_workflow import (
    URL,
    WebClient,
    request,
    web_call,
    wire,
)


class SearchThenFinish:
    def __init__(self, outcome, *, cleanup_error=False):
        self.outcome = outcome
        self.cleanup_error = cleanup_error
        self.client = WebClient()
        self.entered = asyncio.Event()
        self.closed = []
        self.selected = []
        self.turns = 0

    @asynccontextmanager
    async def bind_responses(self, original, **kwargs):
        try:
            yield ResponsesBinding("chat", self.stream)
        finally:
            self.closed.append("binding")
            if self.cleanup_error:
                raise RuntimeError("binding release failed")

    async def stream(self, turn, *, input_tokens):
        self.turns += 1
        number = self.turns
        try:
            if number == 1:
                output = [
                    web_call(
                        turn.tools[0]["name"], {"action": "search", "query": "docs"}
                    )
                ]
            else:
                self.entered.set()
                if isinstance(self.outcome, BaseException):
                    raise self.outcome
                if self.outcome == "wait":
                    await asyncio.Event().wait()
                output = [message_item("msg_end", "Done.", "completed")]
            for frame in wire(output, number=number):
                yield frame
        finally:
            self.closed.append(number)

    async def response(self):
        async def resolve(provider_id):
            self.selected.append(provider_id)
            return self

        routed = _routed_responses_request(_target("fallback", "fallback-model"))
        routed.request.tools = [{"type": "web_search"}]
        executor = ProviderExecutor(
            resolve,
            progress_timeout_seconds=60,
            responses_token_counter=lambda _: 1,
            responses_web_tools=ResponsesWebTools(self.client),
        )
        return await openai_responses_sse_streaming_response(
            executor.stream_responses(
                routed, raw_log_payload={}, request_id="web-failure"
            ),
            headers={},
            pre_start_error_response=lambda exc: JSONResponse({"error": str(exc)}, 500),
            request_id="web-failure",
        )


async def _consume(response):
    try:
        return parse_sse_text(
            "".join([frame async for frame in response.body_iterator])
        )
    finally:
        await response.body_iterator.aclose()


def _assert_retained_failure(events, scenario, error_type="api_error"):
    terminals = [
        event
        for event in events
        if event.event
        in ("response.completed", "response.failed", "response.incomplete")
    ]
    assert len(terminals) == 1
    final = terminals[0].data["response"]
    assert final["status"] == "failed"
    assert final["error"]["type"] == error_type
    assert final["output"][0]["type"] == "web_search_call"
    assert final["output"][0]["status"] == "completed"
    retained = prepare_web_history(request(input=final["output"]))
    assert len(retained.records) == 1
    result = retained.records[0]["result"]
    assert isinstance(result, dict)
    matches = result["results"]
    assert isinstance(matches, list) and isinstance(matches[0], dict)
    assert matches[0]["url"] == URL
    done = [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ]
    assert done == final["output"]
    assert scenario.client.searches == ["docs"]
    assert scenario.selected == ["provider"]
    assert scenario.closed == [1, 2, "binding"]


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True])
async def test_noncanonical_failure_after_search_retains_completed_work(grouped):
    error = RuntimeError("second turn failed")
    if grouped:
        error = ExceptionGroup(
            "turn failed",
            [
                error,
                ExecutionFailure(FailureKind.RATE_LIMIT, 429, "rate limited", False),
            ],
        )
    scenario = SearchThenFinish(error)
    events = await _consume(await scenario.response())
    _assert_retained_failure(
        events, scenario, "rate_limit_error" if grouped else "api_error"
    )


@pytest.mark.asyncio
async def test_executor_timeout_after_search_retains_completed_work(monkeypatch):
    timeouts = []

    def controlled_timeout(deadline):
        timeout = asyncio.timeout(None)
        timeouts.append(timeout)
        return timeout

    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )
    scenario = SearchThenFinish("wait")
    response = await scenario.response()
    consuming = asyncio.create_task(_consume(response))
    try:
        await asyncio.wait_for(scenario.entered.wait(), 5)
        timeouts[-1].reschedule(asyncio.get_running_loop().time())
        events = await asyncio.wait_for(consuming, 5)
        _assert_retained_failure(events, scenario, "timeout_error")
    finally:
        consuming.cancel()
        await asyncio.gather(consuming, return_exceptions=True)


@pytest.mark.asyncio
async def test_binding_cleanup_cannot_append_a_second_terminal():
    scenario = SearchThenFinish("complete", cleanup_error=True)
    events = await _consume(await scenario.response())
    assert [
        event.event
        for event in events
        if event.event
        in ("response.completed", "response.failed", "response.incomplete")
    ] == ["response.completed"]
    assert scenario.closed == [1, 2, "binding"]


@pytest.mark.asyncio
async def test_provider_cleanup_failure_preserves_final_answer_and_web_replay():
    class FailedRelease(SearchThenFinish):
        async def stream(self, turn, *, input_tokens):
            async for frame in super().stream(turn, input_tokens=input_tokens):
                yield frame
            if self.turns == 2:
                raise RuntimeError("provider release failed")

    scenario = FailedRelease("complete")
    events = await _consume(await scenario.response())
    assert [
        event.event
        for event in events
        if event.event
        in ("response.completed", "response.failed", "response.incomplete")
    ] == ["response.completed"]
    final = events[-1].data["response"]
    assert final["output"][0]["status"] == "completed"
    assert final["output"][1]["content"][0]["text"] == "Done."
    assert len(prepare_web_history(request(input=final["output"])).records) == 1
    assert [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ] == final["output"]
    assert scenario.client.searches == ["docs"]
    assert scenario.selected == ["provider"]
    assert scenario.closed == [1, 2, "binding"]


@pytest.mark.asyncio
async def test_invalid_terminal_cannot_publish_a_successful_response():
    class InvalidTerminal(SearchThenFinish):
        async def stream(self, turn, *, input_tokens):
            async with aclosing(
                super().stream(turn, input_tokens=input_tokens)
            ) as source:
                async for frame in source:
                    if self.turns == 2 and "event: response.completed\n" in frame:
                        # The provider's final snapshot omits its completed answer.
                        frame = list(wire([], number=2))[-1]
                    yield frame

    scenario = InvalidTerminal("complete")
    events = await _consume(await scenario.response())
    _assert_retained_failure(events, scenario)
    assert "disagrees" in events[-1].data["response"]["error"]["message"]


@pytest.mark.asyncio
async def test_provider_cleanup_timeout_does_not_complete_pending_web_work(monkeypatch):
    timeouts = []
    releasing = asyncio.Event()

    def controlled_timeout(deadline):
        timeout = asyncio.timeout(None)
        timeouts.append(timeout)
        return timeout

    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )

    class StalledRelease(SearchThenFinish):
        async def stream(self, turn, *, input_tokens):
            async for frame in super().stream(turn, input_tokens=input_tokens):
                yield frame
            releasing.set()
            await asyncio.Event().wait()

    scenario = StalledRelease("complete")
    consuming = asyncio.create_task(_consume(await scenario.response()))
    try:
        await asyncio.wait_for(releasing.wait(), 5)
        timeouts[-1].reschedule(asyncio.get_running_loop().time())
        events = await asyncio.wait_for(consuming, 5)
        assert [
            event.event
            for event in events
            if event.event
            in ("response.completed", "response.failed", "response.incomplete")
        ] == ["response.failed"]
        final = events[-1].data["response"]
        assert final["error"]["type"] == "timeout_error"
        assert final["output"][0]["type"] == "web_search_call"
        assert final["output"][0]["status"] == "incomplete"
        assert not scenario.client.searches
        assert scenario.turns == 1
        assert scenario.closed == [1, "binding"]
        assert scenario.selected == ["provider"]
    finally:
        consuming.cancel()
        await asyncio.gather(consuming, return_exceptions=True)


@pytest.mark.asyncio
async def test_timeout_releasing_a_completed_response_cannot_change_its_outcome(
    monkeypatch,
):
    timeouts = []
    releasing = asyncio.Event()

    def controlled_timeout(deadline):
        timeout = asyncio.timeout(None)
        timeouts.append(timeout)
        return timeout

    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )

    class StalledRelease(SearchThenFinish):
        @asynccontextmanager
        async def bind_responses(self, original, **kwargs):
            try:
                yield ResponsesBinding("chat", self.stream)
            finally:
                releasing.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.closed.append("binding")

    scenario = StalledRelease("complete")
    consuming = asyncio.create_task(_consume(await scenario.response()))
    try:
        await asyncio.wait_for(releasing.wait(), 5)
        timeouts[-1].reschedule(asyncio.get_running_loop().time())
        events = await asyncio.wait_for(consuming, 5)
        assert [
            event.event
            for event in events
            if event.event
            in ("response.completed", "response.failed", "response.incomplete")
        ] == ["response.completed"]
        assert scenario.closed == [1, 2, "binding"]
        assert scenario.selected == ["provider"]
    finally:
        consuming.cancel()
        await asyncio.gather(consuming, return_exceptions=True)


@pytest.mark.asyncio
async def test_external_cancellation_releases_web_binding():
    scenario = SearchThenFinish("wait")
    response = await scenario.response()
    consuming = asyncio.create_task(_consume(response))
    try:
        await asyncio.wait_for(scenario.entered.wait(), 5)
        consuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consuming
        assert scenario.closed == [1, 2, "binding"]
        assert scenario.selected == ["provider"]
    finally:
        consuming.cancel()
        await asyncio.gather(consuming, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_in_exception_group_is_not_a_failed_response():
    scenario = SearchThenFinish(
        BaseExceptionGroup(
            "cancelled turn",
            [
                asyncio.CancelledError(),
                ExecutionFailure(FailureKind.UPSTREAM, 502, "provider failed", False),
            ],
        )
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        await _consume(await scenario.response())
    assert raised.value is scenario.outcome
    assert scenario.closed == [1, 2, "binding"]


@pytest.mark.asyncio
async def test_disconnect_while_emitting_failure_replay_releases_response_once():
    scenario = SearchThenFinish(RuntimeError("second turn failed"))
    response = await scenario.response()
    released = []

    async def release():
        released.append(list(scenario.closed))

    async def send(message):
        if b"fcc:web:" in message.get("body", b""):
            raise OSError("client disconnected")

    await bind_response_lifetime(response, release)
    with pytest.raises(ClientDisconnect):
        await _serve(response, send=send)
    assert released == [[1, 2, "binding"]]
