"""Gemini compatibility errors and Google-shaped public failures."""

from ..json_types import JsonObject


class GeminiConversionError(ValueError):
    """A Gemini request cannot be faithfully represented at this boundary."""


def gemini_error_payload(status_code: int, message: str) -> JsonObject:
    status = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        405: "UNIMPLEMENTED",
        413: "RESOURCE_EXHAUSTED",
        422: "INVALID_ARGUMENT",
        429: "RESOURCE_EXHAUSTED",
        500: "INTERNAL",
        501: "UNIMPLEMENTED",
        502: "UNAVAILABLE",
        503: "UNAVAILABLE",
        504: "DEADLINE_EXCEEDED",
        529: "UNAVAILABLE",
    }.get(status_code, "UNKNOWN")
    return {"error": {"code": status_code, "status": status, "message": message}}
