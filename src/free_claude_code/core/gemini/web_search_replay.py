"""Carry completed Messages WebSearch turns into Gemini history."""

from collections.abc import Iterable, Iterator

from free_claude_code.core.json_types import (
    JsonObject,
    json_objects,
    json_text,
    optional_string,
)

from ..anthropic.content import get_block_attr, get_block_type
from ..anthropic.server_tool_types import HIDDEN_WEB_SEARCH_NAME
from ..anthropic.web_history import public_search_result, web_history_text
from ..history_replay import (
    ReplayRecord,
    decode_replay,
    is_replay,
)
from .errors import GeminiConversionError


def signed_gemini_turn(blocks: Iterable[object]) -> ReplayRecord | None:
    for block in blocks:
        if get_block_type(block) == "redacted_thinking" and is_replay(
            carrier := get_block_attr(block, "data")
        ):
            record = decode_replay(carrier)
            if record.origin.protocol == "gemini":
                return record
    return None


def restore_web_search_history(
    messages: Iterable[JsonObject],
) -> Iterator[JsonObject]:
    for message in messages:
        blocks = message.get("content")
        if message["role"] != "assistant" or not isinstance(blocks, list):
            yield message
            continue
        blocks = list(json_objects(blocks))
        search_ids = [json_text(block["id"]) for block in blocks if _is_search(block)]
        if not search_ids:
            yield message
            continue
        result_blocks = [block for block in blocks if _is_search_result(block)]
        by_id = {json_text(block["tool_use_id"]): block for block in result_blocks}
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
                    {"type": "text", "text": web_history_text(block)}
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


def _is_search(block: JsonObject) -> bool:
    return block.get("type") == "server_tool_use" and block.get("name") == "web_search"


def _is_search_result(block: JsonObject) -> bool:
    return block.get("type") == "web_search_tool_result"


def _restore_turn(
    blocks: list[JsonObject],
    record: ReplayRecord,
    by_id: dict[str, JsonObject],
) -> tuple[list[JsonObject], list[JsonObject]]:
    error = "Signed Gemini WebSearch history does not match its original calls."
    native_calls = [
        call
        for part in json_objects(record.native["parts"])
        if isinstance(call := part.get("functionCall"), dict)
    ]
    visible_calls = [
        block
        for block in blocks
        if block.get("type") == "tool_use" or _is_search(block)
    ]
    if len(native_calls) != len(visible_calls):
        raise GeminiConversionError(error)
    lifted: dict[str, JsonObject] = {}
    results: list[JsonObject] = []
    for visible, native in zip(visible_calls, native_calls, strict=True):
        if not _is_search(visible):
            # The existing history replay validator checks ordinary calls and
            # text against the signed native content after reassembly.
            continue
        name = native.get("name")
        args = native.get("args")
        use_id = json_text(visible["id"])
        visible_input = visible.get("input")
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
            or not isinstance(visible_input, dict)
            or query.strip() != visible_input.get("query")
        ):
            raise GeminiConversionError(error)
        # Gemini may omit a call ID. The replay validator removes the portable
        # alias from both the call and result when restoring the native turn.
        call_id = optional_string(native.get("id")) or use_id
        lifted[use_id] = {
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": args,
        }
        payload = public_search_result(by_id[use_id]["content"])
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
            lifted[json_text(block["id"])] if _is_search(block) else block
            for block in blocks
            if not _is_search_result(block)
        ],
        results,
    )
