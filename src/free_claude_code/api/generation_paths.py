"""Public generation route classification shared by HTTP boundaries."""

from typing import Literal

type WireApi = Literal["messages", "responses", "gemini"]


def generation_api(path: str) -> WireApi | None:
    if path == "/v1/messages":
        return "messages"
    if path == "/v1/responses":
        return "responses"
    if path.startswith("/v1beta/models/"):
        return "gemini"
    return None
