from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.catalog_http import fetch_proxy_models_response


class _ModelsResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _ModelsResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info

    def read(self) -> bytes:
        return self._body


def test_fetch_proxy_models_uses_canonical_bearer_request() -> None:
    with patch(
        "free_claude_code.cli.launchers.catalog_http.open_local_request",
        return_value=_ModelsResponse(b'{"data": []}'),
    ) as open_local_request:
        response = fetch_proxy_models_response("http://127.0.0.1:9191/", "proxy-token")

    assert response == {"data": []}
    request = open_local_request.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:9191/v1/models?view=responses"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer proxy-token"


def test_fetch_proxy_models_can_request_messages_view() -> None:
    with patch(
        "free_claude_code.cli.launchers.catalog_http.open_local_request",
        return_value=_ModelsResponse(b'{"data": []}'),
    ) as open_local_request:
        response = fetch_proxy_models_response(
            "http://127.0.0.1:9191/",
            "proxy-token",
            view="messages",
        )

    assert response == {"data": []}
    request = open_local_request.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:9191/v1/models?view=messages"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer proxy-token"


def test_fetch_proxy_models_rejects_non_object_json() -> None:
    with (
        patch(
            "free_claude_code.cli.launchers.catalog_http.open_local_request",
            return_value=_ModelsResponse(b"[]"),
        ),
        pytest.raises(ValueError, match="JSON object"),
    ):
        fetch_proxy_models_response("http://127.0.0.1:9191", "proxy-token")
