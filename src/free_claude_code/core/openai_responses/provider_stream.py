"""Native Responses stream failures retained for provider recovery."""

from typing import Any


class ResponsesStreamFailure(RuntimeError):
    """An upstream Responses stream reported a terminal failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        body: dict[str, Any] | None = None,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.body = body
        self.event_type = event_type
        self.payload = payload


def responses_stream_failure_from_event(
    event_type: str,
    data: dict[str, Any],
) -> ResponsesStreamFailure:
    """Retain one native failure event for provider-owned retry decisions."""

    response = data.get("response")
    response = response if isinstance(response, dict) else {}
    error = response.get("error", data.get("error"))
    if not isinstance(error, dict):
        error = data if event_type == "error" else {}
    message = error.get("message")
    code = error.get("code") or error.get("type")
    return ResponsesStreamFailure(
        message if isinstance(message, str) and message else "OpenAI response failed.",
        code=code if isinstance(code, str) else None,
        body=dict(error),
        event_type=event_type,
        payload=data,
    )
