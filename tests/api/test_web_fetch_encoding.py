"""Bounded fetch decoding must not ask aiohttp to read its buffered body."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.api.web_tools.egress import WebFetchEgressPolicy
from free_claude_code.api.web_tools.outbound import _run_web_fetch


@pytest.mark.asyncio
@pytest.mark.parametrize("charset", [None, "utf-8"])
async def test_fetch_decodes_without_get_encoding_before_body(charset):
    response = MagicMock()
    response.status = 200
    response.url = "https://example.com/docs"
    response.headers = {"content-type": "text/plain"}
    response.charset = charset
    response.get_encoding.side_effect = RuntimeError("Body has not been read")

    async def body(_chunk_size: int):
        yield "Public café documentation".encode()

    response.content.iter_chunked.side_effect = body
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get.return_value = response_context
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    policy = WebFetchEgressPolicy(False, frozenset({"https"}))
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.get_validated_stream_addrinfos_for_egress",
            return_value=[],
        ),
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=session_context,
        ),
    ):
        result = await _run_web_fetch("https://example.com/docs", policy)
    assert result["data"] == "Public café documentation"
    response.get_encoding.assert_not_called()
    session.get.assert_called_once_with(
        "https://example.com/docs", allow_redirects=False
    )
