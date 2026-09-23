"""Responses object and output item builders."""

from typing import Any


def message_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def reasoning_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def encrypted_reasoning_item(
    item_id: str, encrypted_content: str, status: str
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "encrypted_content": encrypted_content,
    }
