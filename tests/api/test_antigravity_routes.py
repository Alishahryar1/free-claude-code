"""Antigravity session setup has no persistent Admin API."""

import pytest
from fastapi.testclient import TestClient

from tests.api.support import create_test_app


@pytest.mark.parametrize(
    "method,path", [("GET", ""), ("POST", "/connect"), ("POST", "/disconnect")]
)
def test_antigravity_connection_routes_are_removed(method: str, path: str) -> None:
    with TestClient(create_test_app(), client=("127.0.0.1", 50000)) as client:
        response = client.request(method, "/admin/api/integrations/antigravity" + path)
    assert response.status_code == 404
