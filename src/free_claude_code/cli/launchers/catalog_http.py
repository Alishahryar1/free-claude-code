"""Authenticated local catalog acquisition for terminal launchers."""

import json
from typing import Literal
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.core.json_types import JsonObject, JsonValue

CATALOG_TIMEOUT_SECONDS = 35.0


def fetch_proxy_models_response(
    proxy_root_url: str,
    auth_token: str,
    view: Literal["messages", "responses"] = "responses",
) -> JsonObject:
    """Fetch the authenticated FCC-local `/v1/models` response directly."""

    url = f"{proxy_root_url.rstrip('/')}/v1/models?view={view}"
    request = Request(
        url,
        headers={"Authorization": f"Bearer {auth_token}"},
        method="GET",
    )
    with open_local_request(request, timeout=CATALOG_TIMEOUT_SECONDS) as response:
        payload: JsonValue = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("model list response was not a JSON object")
    return payload
