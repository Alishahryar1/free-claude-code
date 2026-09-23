"""Native OpenAI Responses request and event handling."""

import uuid
from collections.abc import Mapping
from copy import deepcopy
from typing import cast

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy

from .errors import openai_error_from_failure
from .events import format_response_sse_event
from .ids import tool_item_id_for_kind
from .models import OpenAIResponsesRequest
from .reasoning import responses_reasoning_config, responses_reasoning_policy

_TERMINAL_EVENT_TYPES = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)


def build_native_responses_request(
    request: OpenAIResponsesRequest,
    *,
    model: str,
    reasoning: ReasoningPolicy,
) -> JsonObject:
    """Build the stateless upstream body without translating Responses input."""

    body = cast(
        JsonObject,
        request.model_dump(mode="json", exclude_none=True),
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
    if reasoning != responses_reasoning_policy(request.reasoning):
        if reasoning_config := responses_reasoning_config(reasoning):
            body["reasoning"] = reasoning_config
        else:
            body.pop("reasoning", None)
    return body


class NativeResponsesRelay:
    """Relay one native Responses lifecycle while retaining upstream identity."""

    def __init__(self, *, public_model: str) -> None:
        self._public_model = public_model
        self._response: JsonObject | None = None
        self._response_id: str | None = None
        self._terminal_type: str | None = None
        self._next_sequence_number = 0

    @property
    def response_id(self) -> str | None:
        return self._response_id

    @property
    def terminal_type(self) -> str | None:
        return self._terminal_type

    @property
    def completed(self) -> bool:
        return self._terminal_type is not None

    def feed(self, event_type: str, payload: Mapping[str, object]) -> str:
        """Format public model metadata and canonical whole-second timestamps."""

        if self._terminal_type is not None:
            raise ValueError(
                f"Responses event {event_type!r} arrived after terminal "
                f"event {self._terminal_type!r}."
            )
        data = cast(JsonObject, deepcopy(dict(payload)))
        response = data.get("response")
        if isinstance(response, dict):
            response["model"] = self._public_model
            # SDK models coerce integer timestamps to floats; strict clients
            # require JSON integers. Preserve any actual fractional precision.
            for field in ("created_at", "completed_at"):
                timestamp = response.get(field)
                if isinstance(timestamp, float) and timestamp.is_integer():
                    response[field] = int(timestamp)
            self._response = cast(JsonObject, deepcopy(response))
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                self._response_id = response_id
        sequence_number = data.get("sequence_number")
        if isinstance(sequence_number, int) and not isinstance(sequence_number, bool):
            self._next_sequence_number = max(
                self._next_sequence_number,
                sequence_number + 1,
            )
        if event_type in _TERMINAL_EVENT_TYPES:
            self._terminal_type = event_type
        return format_response_sse_event(event_type, data)

    def synthesize_failure(self, failure: ExecutionFailure) -> str:
        """Terminate one already-public truncated stream with a safe failure."""

        if self._terminal_type is not None:
            raise ValueError(
                f"Cannot synthesize failure after {self._terminal_type!r}."
            )
        response = (
            deepcopy(self._response)
            if self._response is not None
            else cast(
                JsonObject,
                {
                    "id": self._response_id or f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "output": [],
                },
            )
        )
        response["id"] = (
            self._response_id or response.get("id") or (f"resp_{uuid.uuid4().hex}")
        )
        response["model"] = self._public_model
        response["status"] = "failed"
        response["error"] = cast(JsonValue, openai_error_from_failure(failure))
        return self.feed(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": self._next_sequence_number,
                "response": response,
            },
        )
