"""Construct a continuation without changing the original protocol request."""

from copy import deepcopy
from dataclasses import replace

from .anthropic.models import Message, MessagesRequest
from .anthropic.passthrough import NativeMessagesRequest
from .json_types import JsonValue
from .openai_responses import OpenAIResponsesRequest
from .recovery import CandidateIncompatible, RecoveryCheckpoint

type RecoveryRequest = MessagesRequest | NativeMessagesRequest | OpenAIResponsesRequest

_CONTINUE = (
    "The previous response was interrupted by a connection failure. Continue from "
    "the exact assistant output above without repeating it. The original request "
    "and tools still apply."
)


def continue_request[T: RecoveryRequest](
    request: T, checkpoint: RecoveryCheckpoint
) -> T:
    """Append committed history to a fresh copy of the original request."""
    if checkpoint.published_tools:
        raise CandidateIncompatible("Published client tools require their results.")
    if not checkpoint.content:
        return deepcopy(request)
    content = deepcopy(list(checkpoint.content))
    if isinstance(request, OpenAIResponsesRequest):
        original = request.input
        items: list[JsonValue] = (
            list(deepcopy(original))
            if isinstance(original, list)
            else [{"role": "user", "content": original or ""}]
        )
        return request.model_copy(
            update={
                "input": [*items, *content, {"role": "user", "content": _CONTINUE}]
            },
            deep=True,
        )
    messages: list[JsonValue] = [
        {"role": "assistant", "content": content},
        {"role": "user", "content": _CONTINUE},
    ]
    if isinstance(request, NativeMessagesRequest):
        body = deepcopy(request.body)
        original_messages = body["messages"]
        assert isinstance(original_messages, list)
        body["messages"] = [*original_messages, *messages]
        return replace(request, body=body)
    return request.model_copy(
        update={
            "messages": [
                *deepcopy(request.messages),
                *(Message.model_validate(message) for message in messages),
            ]
        },
        deep=True,
    )
