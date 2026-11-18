"""Carry completed Messages WebSearch turns into Gemini history."""

from collections.abc import Iterable, Iterator
from typing import Any, cast

from ..anthropic.content import get_block_attr, get_block_type
from ..anthropic.server_tool_types import HIDDEN_WEB_SEARCH_NAME
from ..history_replay import (
    ReplayRecord,
    decode_replay,
    is_replay,
    tool_history_context,
)
from .errors import GeminiConversionError


def signed_gemini_turn(blocks: Iterable[Any]) -> ReplayRecord | None:
    for block in blocks:
        if get_block_type(block) == "redacted_thinking" and is_replay(
            carrier := get_block_attr(block, "data")
        ):
            record = decode_replay(carrier)
            if record.origin.protocol == "gemini":
                return record
    return None


def restore_web_search_history(
    messages: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for message in messages:
        blocks = message.get("content")
        if message["role"] != "assistant" or not isinstance(blocks, list):
            yield message
            continue
        search_ids = [block["id"] for block in blocks if _is_search(block)]
        if not search_ids:
            yield message
            continue
        result_blocks = [block for block in blocks if _is_search_result(block)]
        by_id = {block["tool_use_id"]: block for block in result_blocks}
        if (
            len(by_id) != len(result_blocks)
            or len(by_id) != len(search_ids)
            or by_id.keys() != set(search_ids)
        ):
            raise GeminiConversionError("Incomplete WebSearch history.")
        record = signed_gemini_turn(blocks)
        if record is None:
            yield {
                **message,
                "content": [
                    {"type": "text", "text": tool_history_context(block)}
                    if _is_search(block) or _is_search_result(block)
                    else block
                    for block in blocks
                ],
            }
            continue
        assistant, search_results = _restore_turn(blocks, record, by_id)
        yield {**message, "content": assistant}
        # The request converter joins these and the client's results into the
        # same Gemini functionResponse turn, in original function-call order.
        yield {"role": "user", "content": search_results}


def _is_search(block: dict[str, Any]) -> bool:
    return block.get("type") == "server_tool_use" and block.get("name") == "web_search"


def _is_search_result(block: dict[str, Any]) -> bool:
    return block.get("type") == "web_search_tool_result"


def _restore_turn(
    blocks: list[dict[str, Any]],
    record: ReplayRecord,
    by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    error = "Signed Gemini WebSearch history does not match its original calls."
    native_calls: list[dict[str, Any]] = [
        part["functionCall"]
        for part in cast(list[dict[str, Any]], record.native["parts"])
        if isinstance(part.get("functionCall"), dict)
    ]
    visible_calls = [
        block
        for block in blocks
        if block.get("type") == "tool_use" or _is_search(block)
    ]
    if len(native_calls) != len(visible_calls):
        raise GeminiConversionError(error)
    lifted: dict[str, dict[str, Any]] = {}
    results = []
    for visible, native in zip(visible_calls, native_calls, strict=True):
        if not _is_search(visible):
            # The existing history replay validator checks ordinary calls and
            # text against the signed native content after reassembly.
            continue
        name = native.get("name")
        args = native.get("args")
        use_id = visible["id"]
        if (
            not isinstance(name, str)
            or not (
                name == HIDDEN_WEB_SEARCH_NAME
                or (
                    name.startswith(HIDDEN_WEB_SEARCH_NAME + "_")
                    and name.removeprefix(HIDDEN_WEB_SEARCH_NAME + "_").isdigit()
                )
            )
            or not isinstance(args, dict)
            or not isinstance(query := args.get("query"), str)
            or query.strip() != visible["input"].get("query")
        ):
            raise GeminiConversionError(error)
        # Gemini may omit a call ID. The replay validator removes the portable
        # alias from both the call and result when restoring the native turn.
        call_id = native.get("id") or use_id
        lifted[use_id] = {
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": args,
        }
        payload = by_id[use_id]["content"]
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": payload
                if isinstance(payload, dict)
                else {"results": payload},
                "is_error": isinstance(payload, dict),
            }
        )
    return (
        [
            lifted[block["id"]] if _is_search(block) else block
            for block in blocks
            if not _is_search_result(block)
        ],
        results,
    )
