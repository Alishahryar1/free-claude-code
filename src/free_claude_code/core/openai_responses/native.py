"""Native OpenAI Responses request and event handling."""

from typing import cast

from free_claude_code.core.failures import UnsupportedRequestFeature
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningPolicy

from .ids import tool_item_id_for_kind
from .models import OpenAIResponsesRequest
from .reasoning import responses_reasoning_config, responses_reasoning_policy


def build_native_responses_request(
    request: OpenAIResponsesRequest,
    *,
    model: str,
    reasoning: ReasoningPolicy,
    preserve_features: bool = False,
) -> JsonObject:
    """Build the stateless upstream body without translating Responses input."""

    body = cast(
        JsonObject,
        request.model_dump(mode="json", exclude_none=True),
    )
    if preserve_features:
        body.update(request.model_extra or {})
        if request.previous_response_id:
            raise UnsupportedRequestFeature(
                "Recovery requires materialized response history."
            )
    if isinstance(items := body.get("input"), list):
        for item in items:
            if not isinstance(item, dict) or item.get("type") not in (
                "function_call",
                "custom_tool_call",
            ):
                continue
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id != tool_item_id_for_kind(
                item_id,
                kind="custom" if item["type"] == "custom_tool_call" else "function",
            ):
                # Full calls replay by call_id; incompatible item IDs are optional.
                del item["id"]
    body["model"] = model
    body["stream"] = True
    body["store"] = False
    body.pop("previous_response_id", None)
    if not (
        preserve_features and request.reasoning is not None
    ) and reasoning != responses_reasoning_policy(request.reasoning):
        if reasoning_config := responses_reasoning_config(reasoning):
            body["reasoning"] = reasoning_config
        else:
            body.pop("reasoning", None)
    return body
