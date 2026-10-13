"""Anthropic API credentials and request-owned protocol headers."""

import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import date

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.native import RESERVED_MESSAGES_EXTRA_FIELDS
from free_claude_code.core.json_types import JsonObject

API_VERSION = "2023-06-01"
_BETA = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def api_headers(api_key: str, workspace_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "anthropic-version": API_VERSION}
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id
    return headers


def native_request(
    body: JsonObject,
    headers: Mapping[str, str] | None,
) -> tuple[JsonObject, dict[str, str]]:
    result = deepcopy(body)
    result.pop("original_model", None)
    result.pop("resolved_provider_model", None)
    extra = result.pop("extra_body", None)
    if extra is not None:
        if not isinstance(extra, dict):
            raise InvalidRequestError("Messages extra_body must be an object.")
        for key, value in extra.items():
            if key.lower() in RESERVED_MESSAGES_EXTRA_FIELDS or key in result:
                raise InvalidRequestError(
                    f"Messages extra_body cannot override {key!r}."
                )
            result[key] = value
    incoming = {key.lower(): value for key, value in (headers or {}).items()}
    protocol: dict[str, str] = {}
    version = incoming.get("anthropic-version")
    if version is not None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", version) is None:
            raise InvalidRequestError("Invalid anthropic-version header.")
        try:
            date.fromisoformat(version)
        except ValueError as error:
            raise InvalidRequestError("Invalid anthropic-version header.") from error
        protocol["anthropic-version"] = version
    betas = result.pop("betas", [])
    if not isinstance(betas, list) or any(not isinstance(beta, str) for beta in betas):
        raise InvalidRequestError("Messages betas must be a list of header tokens.")
    values = [
        value.strip()
        for value in incoming.get("anthropic-beta", "").split(",")
        if value.strip()
    ]
    for beta in [*values, *betas]:
        if not isinstance(beta, str) or _BETA.fullmatch(beta) is None:
            raise InvalidRequestError(
                "Messages beta names must be valid header tokens."
            )
    merged = tuple(dict.fromkeys([*values, *betas]))
    if merged:
        protocol["anthropic-beta"] = ",".join(merged)
    return result, protocol
