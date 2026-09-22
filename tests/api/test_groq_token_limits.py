import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.api.support import create_test_app
from tests.providers.test_groq_tpm import (
    _ITPM_MESSAGE,
    _MODEL,
    _ORIGINAL_MAX,
    _detail,
    _provider,
    _status_error,
)


@pytest.mark.parametrize(
    ("wire", "stream"), [("messages", False), ("messages", True), ("responses", True)]
)
@pytest.mark.parametrize("quota", ["itpm", "tpm"])
def test_groq_quota_failure_reaches_clients_after_correction(wire, stream, quota):
    provider = _provider()
    error = (
        _status_error(detail=_detail(_ITPM_MESSAGE))
        if quota == "itpm"
        else _status_error()
    )
    create = AsyncMock(side_effect=error)
    payload = {"model": f"groq/{_MODEL}", "stream": stream}
    if wire == "messages":
        payload.update(
            messages=[{"role": "user", "content": "Hello"}], max_tokens=_ORIGINAL_MAX
        )
    else:
        payload.update(input="Hello", max_output_tokens=_ORIGINAL_MAX)
    try:
        with (
            patch(
                "free_claude_code.api.routes.resolve_provider", return_value=provider
            ),
            patch.object(provider._client.chat.completions, "create", create),
            TestClient(create_test_app()) as client,
        ):
            response = client.post(f"/v1/{wire}", json=payload)
        assert response.status_code == 400
        assert response.headers["x-should-retry"] == "false"
        body = response.json()["error"]
        assert body["type"] == "invalid_request_error"
        assert "token-rate allowance" in body["message"]
        assert (
            "Limit 7000, Requested 23253" in body["message"]
            if quota == "itpm"
            else "Limit 8000" in body["message"]
        )
        assert response.headers["request-id"] in body["message"]
        assert create.await_count == (1 if quota == "itpm" else 2)
        assert (
            create.await_args_list[0].kwargs["max_completion_tokens"] == _ORIGINAL_MAX
        )
        if quota == "tpm":
            assert (
                create.await_args_list[1].kwargs["max_completion_tokens"]
                < _ORIGINAL_MAX
            )
    finally:
        asyncio.run(provider.cleanup())
