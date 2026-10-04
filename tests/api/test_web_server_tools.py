import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.responses import JSONResponse, StreamingResponse

import free_claude_code.api.web_tools.constants as web_tool_constants
from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.api.web_tools import egress as web_egress
from free_claude_code.api.web_tools.egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
)
from free_claude_code.api.web_tools.outbound import (
    _drain_response_body_capped,
    _read_response_body_capped,
    _run_web_fetch,
)
from free_claude_code.api.web_tools.request import is_web_server_tool_request
from free_claude_code.api.web_tools.streaming import stream_web_server_tool_response
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.routing import (
    ModelRouter,
    ResolvedModel,
    RoutedMessagesRequest,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import Message, MessagesRequest, Tool
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.version import package_version
from free_claude_code.messaging.event_parser import parse_cli_event

_STRICT_EGRESS = WebFetchEgressPolicy(
    allow_private_network_targets=False,
    allowed_schemes=frozenset({"http", "https"}),
)
_PROVIDER_IDS = tuple(PROVIDER_CATALOG)


def test_web_tool_user_agent_reports_installed_package_version() -> None:
    assert {
        "User-Agent": (f"Mozilla/5.0 compatible; free-claude-code/{package_version()}")
    } == web_tool_constants._WEB_TOOL_HTTP_HEADERS


class FixedProviderModelRouter(ModelRouter):
    """Test double that pins provider identity."""

    def __init__(
        self,
        settings: Settings,
        provider_id: str,
        *,
        provider_model: str | None = None,
    ) -> None:
        super().__init__(settings)
        self._fixed_provider_id = provider_id
        self._fixed_provider_model = provider_model

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        provider_model = self._fixed_provider_model or request.model
        resolved = ResolvedModel(
            original_model=request.model,
            provider_id=self._fixed_provider_id,
            provider_model=provider_model,
            provider_model_ref=f"{self._fixed_provider_id}/{provider_model}",
            reasoning_preference=ReasoningPreference.OFF,
        )
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model
        return RoutedMessagesRequest(
            request=routed,
            resolved=resolved,
            reasoning=ReasoningPolicy.off(),
        )


def test_web_server_tool_not_detected_when_tool_only_listed():
    """Listing web_search without forcing it must not skip the upstream provider."""
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )

    assert not is_web_server_tool_request(request)


def test_web_server_tool_detected_when_tool_choice_forces_it():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert is_web_server_tool_request(request)


def test_web_server_tool_not_detected_when_forced_name_missing_from_tools():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
        tools=[Tool(name="other", type="function")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert not is_web_server_tool_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
async def test_service_rejects_forced_server_tool_when_local_handler_is_disabled(
    provider_id: str,
):
    """Every provider needs FCC's local handler for forced server tools."""
    settings = Settings()
    assert settings.enable_web_server_tools is False
    service = MessagesHandler(
        settings,
        provider_resolver=lambda _: MagicMock(),
        model_router=FixedProviderModelRouter(settings, provider_id),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Perform a web search for the query: DeepSeek V4 model release 2026",
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )
    with pytest.raises(InvalidRequestError, match="ENABLE_WEB_SERVER_TOOLS"):
        await service.create(request)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://192.168.1.1/",
        "http://10.0.0.1/",
        "http://[::1]/",
        "http://localhost/foo",
        "http://mybox.local/",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_enforce_web_fetch_egress_blocks_internal_or_disallowed(url: str):
    with pytest.raises(WebFetchEgressViolation):
        enforce_web_fetch_egress(url, _STRICT_EGRESS)


def test_enforce_web_fetch_egress_allows_global_literal_ip():
    enforce_web_fetch_egress("http://8.8.8.8/", _STRICT_EGRESS)


def test_enforce_web_fetch_egress_skips_private_checks_when_opted_in():
    enforce_web_fetch_egress(
        "http://127.0.0.1/",
        WebFetchEgressPolicy(
            allow_private_network_targets=True,
            allowed_schemes=frozenset({"http", "https"}),
        ),
    )


def _cm(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _stream_cm(response: httpx.Response) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _json_body(response: JSONResponse) -> dict[str, Any]:
    payload = json.loads(bytes(response.body).decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


async def _streaming_body_text(response: StreamingResponse) -> str:
    parts = [
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ]
    return "".join(parts)


def _aiohttp_response(
    status: int,
    *,
    url: str = "http://8.8.8.8/",
    location: str | None = None,
    body: bytes = b"hello world",
) -> MagicMock:
    r = MagicMock()
    r.status = status
    r.url = url
    hdrs: dict[str, str] = {}
    if location is not None:
        hdrs["location"] = location
    r.headers = hdrs
    r.charset = "utf-8"
    r.raise_for_status = MagicMock()
    r.request_info = MagicMock()
    r.history = ()

    async def iter_chunked(_n: int) -> Any:
        yield body

    r.content.iter_chunked = MagicMock(side_effect=iter_chunked)
    return r


def _aiohttp_client_session_patch(
    *responses: MagicMock,
) -> tuple[MagicMock, MagicMock]:
    """Build ``ClientSession`` mock that serves ``responses`` to successive ``get`` calls."""
    queue = list(responses)
    n = 0

    def get_side(*_a: Any, **_k: Any) -> Any:
        nonlocal n
        resp = queue[n] if n < len(queue) else queue[-1]
        n += 1
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    session = MagicMock()
    session.get = MagicMock(side_effect=get_side)

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=session)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    return client_cm, session


def test_enforce_web_fetch_egress_documents_connect_time_pinning():
    assert enforce_web_fetch_egress.__doc__ and "resolved addresses" in (
        enforce_web_fetch_egress.__doc__ or ""
    )
    assert (
        web_egress.get_validated_stream_addrinfos_for_egress.__doc__
        and "pinning"
        in (web_egress.get_validated_stream_addrinfos_for_egress.__doc__ or "")
    )
    assert "DNS-pinned" in (_run_web_fetch.__doc__ or "")


@pytest.mark.asyncio
async def test_run_web_fetch_follows_redirect_when_each_hop_is_allowed():
    res_redirect = _aiohttp_response(
        302, url="http://8.8.8.8/start", location="/final", body=b""
    )
    res_ok = _aiohttp_response(200, url="http://8.8.8.8/final", body=b"hello world")
    client_cm, session = _aiohttp_client_session_patch(res_redirect, res_ok)
    with patch(
        "free_claude_code.api.web_tools.outbound.ClientSession", return_value=client_cm
    ):
        out = await _run_web_fetch("http://8.8.8.8/start", _STRICT_EGRESS)

    assert out["data"] == "hello world"
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_run_web_fetch_truncates_large_body_to_byte_cap(monkeypatch):
    huge = b"x" * 5000
    res_ok = _aiohttp_response(200, url="http://8.8.8.8/big", body=huge)
    client_cm, _ = _aiohttp_client_session_patch(res_ok)
    monkeypatch.setattr(web_tool_constants, "_MAX_WEB_FETCH_RESPONSE_BYTES", 100)
    with patch(
        "free_claude_code.api.web_tools.outbound.ClientSession", return_value=client_cm
    ):
        out = await _run_web_fetch("http://8.8.8.8/big", _STRICT_EGRESS)

    assert len(out["data"]) <= 100
    assert out["data"] == "x" * 100


@pytest.mark.asyncio
async def test_run_web_fetch_redirect_to_blocked_host_raises():
    res_redirect = _aiohttp_response(
        302,
        url="http://8.8.8.8/start",
        location="http://127.0.0.1/secret",
        body=b"",
    )
    client_cm, session = _aiohttp_client_session_patch(res_redirect)
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation),
    ):
        await _run_web_fetch("http://8.8.8.8/start", _STRICT_EGRESS)

    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_run_web_fetch_redirect_without_location_raises():
    res_bad = _aiohttp_response(302, url="http://8.8.8.8/here", body=b"")
    client_cm, _ = _aiohttp_client_session_patch(res_bad)
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation, match="missing Location"),
    ):
        await _run_web_fetch("http://8.8.8.8/here", _STRICT_EGRESS)


@pytest.mark.asyncio
async def test_run_web_fetch_excess_redirects_raises():
    res1 = _aiohttp_response(302, url="http://8.8.8.8/a", location="/b", body=b"")
    res2 = _aiohttp_response(302, url="http://8.8.8.8/b", location="/c", body=b"")
    client_cm, _ = _aiohttp_client_session_patch(res1, res2)
    with (
        patch("free_claude_code.api.web_tools.constants._MAX_WEB_FETCH_REDIRECTS", 1),
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation, match="exceeded maximum redirects"),
    ):
        await _run_web_fetch("http://8.8.8.8/a", _STRICT_EGRESS)


@pytest.mark.asyncio
async def test_streams_web_search_server_tool_result(monkeypatch):
    async def fake_search(query: str) -> list[dict[str, str]]:
        assert query == "DeepSeek V4 model release 2026"
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content=(
                    "Perform a web search for the query: DeepSeek V4 model release 2026"
                ),
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=42, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    starts = [e for e in events if e.event == "content_block_start"]
    assert starts[0].data["content_block"]["type"] == "server_tool_use"
    assert starts[0].data["content_block"]["name"] == "web_search"
    tool_use_id = starts[0].data["content_block"]["id"]
    assert starts[1].data["content_block"]["type"] == "web_search_tool_result"
    assert starts[1].data["content_block"]["tool_use_id"] == tool_use_id
    assert starts[1].data["content_block"]["content"][0]["url"] == (
        "https://example.com/v4"
    )
    text_deltas = [
        e
        for e in events
        if e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
    ]
    assert text_deltas, "summary must be streamed as text_delta"
    assert "example.com" in text_content(events)
    cli_text: list[str] = []
    for ev in events:
        cli_text.extend(
            str(p.get("text", ""))
            for p in parse_cli_event(ev.data)
            if p.get("type") == "text_delta"
        )
    assert "example.com" in "".join(cli_text)
    deltas = [e for e in events if e.event == "message_delta"]
    assert deltas[-1].data["usage"]["server_tool_use"] == {"web_search_requests": 1}


@pytest.mark.asyncio
async def test_service_streams_forced_web_search_by_default(monkeypatch):
    async def fake_search(_query: str) -> list[dict[str, str]]:
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = MagicMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(
            settings,
            _PROVIDER_IDS[0],
            provider_model="upstream-model",
        ),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        stream=True,
        messages=[Message(role="user", content="Search for DeepSeek V4")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    response = await service.create(request)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    raw = await _streaming_body_text(response)
    assert "event: message_start" in raw
    assert "DeepSeek V4 Released" in raw
    message_start = next(
        event.data["message"]
        for event in parse_sse_text(raw)
        if event.event == "message_start"
    )
    assert message_start["model"] == request.model
    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_service_aggregates_forced_web_search_when_stream_false(monkeypatch):
    async def fake_search(_query: str) -> list[dict[str, str]]:
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = MagicMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(settings, _PROVIDER_IDS[0]),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="Search for DeepSeek V4")],
        stream=False,
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    response = await service.create(request)

    assert isinstance(response, JSONResponse)
    assert response.headers["content-type"].startswith("application/json")
    body = _json_body(response)
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert body["content"][1]["content"][0]["url"] == "https://example.com/v4"
    assert "DeepSeek V4 Released" in body["content"][2]["text"]
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}
    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_forced_web_fetch_ignores_stale_url_from_prior_user_turns(monkeypatch):
    """Only the latest user message supplies the URL (not earlier transcript text)."""
    target = "https://new-only.example.com/page"

    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        assert url == target
        return {
            "url": url,
            "title": "T",
            "media_type": "text/plain",
            "data": "x",
        }

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_fetch", fake_fetch
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Earlier turn https://stale.com/old-article ignore this",
            ),
            Message(role="assistant", content="ok"),
            Message(
                role="user",
                content=f"Please fetch {target} for the summary",
            ),
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=1, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    assert target in raw


@pytest.mark.asyncio
async def test_service_aggregates_forced_web_fetch_when_stream_false(monkeypatch):
    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        return {
            "url": url,
            "title": "Example Article",
            "media_type": "text/plain",
            "data": "Article body",
        }

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_fetch", fake_fetch
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = MagicMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(settings, _PROVIDER_IDS[0]),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="Fetch https://example.com/article")],
        stream=False,
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    response = await service.create(request)

    assert isinstance(response, JSONResponse)
    assert response.headers["content-type"].startswith("application/json")
    body = _json_body(response)
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_fetch_tool_result",
        "text",
    ]
    assert body["content"][1]["content"]["content"]["title"] == "Example Article"
    assert body["content"][2]["text"] == "Article body"
    assert body["usage"]["server_tool_use"] == {"web_fetch_requests": 1}
    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_streams_web_fetch_server_tool_result(monkeypatch):
    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        assert url == "https://example.com/article"
        return {
            "url": url,
            "title": "Example Article",
            "media_type": "text/plain",
            "data": "Article body",
        }

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_fetch", fake_fetch
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(role="user", content="Fetch https://example.com/article please")
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=42, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    starts = [e for e in events if e.event == "content_block_start"]
    assert starts[0].data["content_block"]["type"] == "server_tool_use"
    tool_use_id = starts[0].data["content_block"]["id"]
    assert starts[1].data["content_block"]["type"] == "web_fetch_tool_result"
    assert starts[1].data["content_block"]["tool_use_id"] == tool_use_id
    assert starts[1].data["content_block"]["content"]["content"]["title"] == (
        "Example Article"
    )
    assert any(
        e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
        for e in events
    )
    assert "Article body" in text_content(events)
    cli_text: list[str] = []
    for ev in events:
        cli_text.extend(
            str(p.get("text", ""))
            for p in parse_cli_event(ev.data)
            if p.get("type") == "text_delta"
        )
    assert "Article body" in "".join(cli_text)
    deltas = [e for e in events if e.event == "message_delta"]
    assert deltas[-1].data["usage"]["server_tool_use"] == {"web_fetch_requests": 1}


@pytest.mark.asyncio
async def test_streams_web_fetch_error_summary_generic_by_default(monkeypatch):
    secret = "sensitive-upstream-token"

    async def boom(_url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        raise ValueError(secret)

    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", boom)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Fetch https://example.com/sensitive-path?x=1 please",
            )
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    with patch("free_claude_code.api.web_tools.outbound.logger.warning") as log_warn:
        raw = "".join(
            [
                event
                async for event in stream_web_server_tool_response(
                    request,
                    input_tokens=1,
                    web_fetch_egress=_STRICT_EGRESS,
                    verbose_client_errors=False,
                )
            ]
        )

    assert secret not in raw
    assert "ValueError" not in raw
    assert "Web tool request failed." in raw
    err_events = parse_sse_text(raw)
    assert_anthropic_stream_contract(err_events)
    assert any(
        e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
        for e in err_events
    )
    cli_err_text: list[str] = []
    for ev in err_events:
        cli_err_text.extend(
            str(p.get("text", ""))
            for p in parse_cli_event(ev.data)
            if p.get("type") == "text_delta"
        )
    assert "Web tool request failed." in "".join(cli_err_text)
    log_blob = " ".join(str(a) for c in log_warn.call_args_list for a in c.args)
    assert secret not in log_blob
    assert "example.com" in log_blob
    assert "/sensitive-path" not in log_blob


@pytest.mark.asyncio
async def test_streams_web_fetch_error_summary_verbose_includes_exception_class(
    monkeypatch,
):
    async def boom(_url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        raise OSError(5, "oops")

    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", boom)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="Fetch https://example.com/x")],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request,
                input_tokens=1,
                web_fetch_egress=_STRICT_EGRESS,
                verbose_client_errors=True,
            )
        ]
    )
    assert "OSError" in raw


@pytest.mark.asyncio
async def test_read_response_body_capped_truncates_single_oversized_chunk():
    cap = 500

    async def aiter_bytes(chunk_size=None):
        yield b"z" * (cap * 20)

    response = MagicMock()
    response.aiter_bytes = aiter_bytes

    out = await _read_response_body_capped(response, cap)
    assert len(out) == cap
    assert out == b"z" * cap


@pytest.mark.asyncio
async def test_drain_response_body_capped_stops_after_first_chunk_when_oversized():
    cap = 300
    chunk_calls = {"n": 0}

    async def aiter_bytes(chunk_size=None):
        chunk_calls["n"] += 1
        yield b"y" * (cap * 10)

    response = MagicMock()
    response.aiter_bytes = aiter_bytes

    await _drain_response_body_capped(response, cap)
    assert chunk_calls["n"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
async def test_service_rejects_listed_server_tools_for_every_provider(
    provider_id: str,
) -> None:
    settings = Settings()
    service = MessagesHandler(
        settings,
        provider_resolver=lambda _: MagicMock(),
        model_router=FixedProviderModelRouter(settings, provider_id),
    )
    request = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[Message(role="user", content="q")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )
    with pytest.raises(InvalidRequestError, match="cannot pass listed Anthropic"):
        await service.create(request)
