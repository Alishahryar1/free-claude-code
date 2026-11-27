"""Public generation route classification shared by HTTP boundaries."""

from typing import Literal

type WireApi = Literal["messages", "responses", "gemini", "chat"]

_GEMINI_ACTIONS = (":generateContent", ":streamGenerateContent")


def generation_api(path: str) -> WireApi | None:
    if path == "/v1/messages":
        return "messages"
    if path == "/v1/responses":
        return "responses"
    if path == "/v1/chat/completions":
        return "chat"
    if path.startswith("/v1beta/models/"):
        return "gemini"
    # Gemini SDKs with a custom base URL call /v1/models/... too. OpenAI clients
    # share that prefix, so only Gemini generation methods are Gemini there.
    if path.startswith("/v1/models/") and path.endswith(_GEMINI_ACTIONS):
        return "gemini"
    return None
