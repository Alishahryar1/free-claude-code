"""Direct GenerateContent translations into the three supported upstream wires."""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

from ..anthropic.native import NativeMessagesOptions
from ..history_replay import ReplayRecord, encode_replay
from ..json_types import JsonObject, JsonValue
from ..openai_chat import (
    ChatToolResultImages,
    close_chat_tool_result_turns,
    image_tool_result_label,
)
from ..openai_responses import responses_reasoning_config
from ..openai_tool_names import OpenAIToolNameCodec
from ..reasoning import ReasoningPolicy
from .errors import GeminiConversionError
from .history import normalized_history
from .models import GeminiRequest, InlineData
from .replay import CallReplay
from .schema import json_schema

type Destination = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class GeminiPreparedRequest:
    body: JsonObject
    tool_names: OpenAIToolNameCodec
    tool_schemas: dict[str, JsonObject]
    reserved_tool_ids: frozenset[str]
    call_replay: tuple[CallReplay, ...]


def _dump(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _append(items: list[JsonObject], role: str, part: JsonObject) -> None:
    if (
        items
        and items[-1].get("role") == role
        and isinstance(items[-1].get("content"), list)
    ):
        cast(list[JsonValue], items[-1]["content"]).append(part)
    else:
        items.append({"role": role, "content": [part]})


def _image(image: InlineData, destination: Destination) -> JsonObject:
    if destination == "messages":
        return {"type": "image", "source": image.source()}
    if destination == "responses":
        return {"type": "input_image", "image_url": image.url()}
    return {"type": "image_url", "image_url": {"url": image.url()}}


def _reasoning(
    items: list[JsonObject], records: list[ReplayRecord], destination: Destination
) -> None:
    for record in records:
        carrier = encode_replay(record)
        if destination == "messages":
            _append(items, "assistant", {"type": "redacted_thinking", "data": carrier})
        elif destination == "responses":
            items.append({"type": "reasoning", "encrypted_content": carrier})
        else:
            if not items or items[-1].get("role") != "assistant":
                items.append({"role": "assistant", "content": []})
            details = items[-1].setdefault("reasoning_details", [])
            assert isinstance(details, list)
            details.append({"type": "reasoning.encrypted", "data": carrier})


def _build(request: GeminiRequest, destination: Destination) -> GeminiPreparedRequest:
    history = normalized_history(request)
    declarations = [
        item for tool in request.tools for item in tool.function_declarations
    ]
    names = [item.name for item in declarations]
    for turn in history:
        names.extend(
            part.function_call.name
            for part in turn.content.parts
            if part.function_call is not None
        )
    codec = OpenAIToolNameCodec.from_names(names)
    schemas = {
        item.name: deepcopy(item.parameters_json_schema)
        if item.parameters_json_schema is not None
        else json_schema(item.parameters)
        if item.parameters is not None
        else {"type": "object", "properties": {}}
        for item in declarations
    }
    items: list[JsonObject] = []
    pending_images: list[JsonObject] = []
    reserved: set[str] = set()
    call_replay: list[CallReplay] = []

    def flush_images() -> None:
        items.extend(pending_images)
        pending_images.clear()

    for turn in history:
        content = turn.content
        role = (
            "assistant"
            if content.role == "model"
            or (
                content.role is None
                and any(part.function_call is not None for part in content.parts)
            )
            else "user"
        )
        records = [record for bundle in turn.replay for record in bundle.records]
        call_replay.extend(record for bundle in turn.replay for record in bundle.calls)
        if records:
            if role != "assistant":
                raise GeminiConversionError("Reasoning replay requires model content")
            flush_images()
            _reasoning(items, records, destination)
        for part in content.parts:
            call, result = part.function_call, part.function_response
            if result is None and any(
                value is not None for value in (part.text, part.inline_data, call)
            ):
                flush_images()
            if call is not None:
                if role != "assistant":
                    raise GeminiConversionError("functionCall requires model content")
                assert call.id is not None
                reserved.add(call.id)
                name = codec.encode(call.name)
                if destination == "messages":
                    _append(
                        items,
                        "assistant",
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": name,
                            "input": deepcopy(call.args),
                        },
                    )
                elif destination == "responses":
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": name,
                            "arguments": _dump(call.args),
                        }
                    )
                else:
                    if not items or items[-1].get("role") != "assistant":
                        items.append({"role": "assistant", "content": []})
                    calls = items[-1].setdefault("tool_calls", [])
                    assert isinstance(calls, list)
                    calls.append(
                        {
                            "type": "function",
                            "id": call.id,
                            "function": {"name": name, "arguments": _dump(call.args)},
                        }
                    )
            elif result is not None:
                assert result.id is not None
                text = _dump(result.response)
                if destination == "messages":
                    output: list[JsonValue] = [{"type": "text", "text": text}]
                    output.extend(
                        _image(part.inline_data, destination) for part in result.parts
                    )
                    _append(
                        items,
                        "user",
                        {
                            "type": "tool_result",
                            "tool_use_id": result.id,
                            "content": output,
                            "is_error": "error" in result.response,
                        },
                    )
                elif destination == "responses":
                    output = [{"type": "input_text", "text": text}]
                    output.extend(
                        _image(part.inline_data, destination) for part in result.parts
                    )
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": result.id,
                            "output": output if result.parts else text,
                        }
                    )
                else:
                    items.append(
                        {"role": "tool", "tool_call_id": result.id, "content": text}
                    )
                    if result.parts:
                        images: list[JsonValue] = [
                            {"type": "text", "text": image_tool_result_label(result.id)}
                        ]
                        images.extend(
                            _image(part.inline_data, destination)
                            for part in result.parts
                        )
                        pending_images.append(
                            ChatToolResultImages(role="user", content=images)
                        )
            elif part.inline_data is not None:
                if role != "user":
                    raise GeminiConversionError(
                        "Inline images require user content or a function result"
                    )
                _append(items, role, _image(part.inline_data, destination))
            elif part.text is not None:
                if part.thought and records:
                    continue  # The record contains the signed/native readable copy.
                kind = "output_text" if role == "assistant" else "input_text"
                _append(
                    items,
                    role,
                    {
                        "type": kind if destination == "responses" else "text",
                        "text": part.text,
                    },
                )
    flush_images()
    if not items:
        raise GeminiConversionError("contents has no representable conversation input")
    body: JsonObject = {"model": request.model, "stream": True}
    if destination == "chat":
        body["messages"] = cast(list[JsonValue], close_chat_tool_result_turns(items))
    else:
        body["messages" if destination == "messages" else "input"] = cast(
            list[JsonValue], items
        )
    if request.system_instruction is not None:
        system = "\n\n".join(
            part.text or "" for part in request.system_instruction.parts
        )
        if destination == "chat":
            messages = cast(list[JsonValue], body["messages"])
            messages.insert(0, {"role": "system", "content": system})
        else:
            body["system" if destination == "messages" else "instructions"] = system
    choice = request.tool_config.function_calling_config
    allowed = set(choice.allowed_function_names)
    tools: list[JsonValue] = []
    for declaration in declarations:
        if allowed and declaration.name not in allowed:
            continue
        name = codec.encode(declaration.name)
        tool: JsonObject = {"name": name}
        if declaration.description is not None:
            tool["description"] = declaration.description
        tool["input_schema" if destination == "messages" else "parameters"] = schemas[
            declaration.name
        ]
        tools.append(
            {"type": "function", "function": tool}
            if destination == "chat"
            else tool
            if destination == "messages"
            else {"type": "function", **tool}
        )
    if tools:
        body["tools"] = tools
        mode = {
            "MODE_UNSPECIFIED": "auto",
            "AUTO": "auto",
            "NONE": "none",
            "ANY": "any" if destination == "messages" else "required",
        }[choice.mode]
        body["tool_choice"] = {"type": mode} if destination == "messages" else mode
    _controls(request, body, destination)
    return GeminiPreparedRequest(
        body, codec, schemas, frozenset(reserved), tuple(call_replay)
    )


def _controls(
    request: GeminiRequest, body: JsonObject, destination: Destination
) -> None:
    config = request.generation_config
    mapping = {
        "max_output_tokens": "max_output_tokens"
        if destination == "responses"
        else "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
    }
    if destination == "chat":
        mapping.update(
            seed="seed",
            presence_penalty="presence_penalty",
            frequency_penalty="frequency_penalty",
        )
    if destination == "messages":
        mapping["top_k"] = "top_k"
    for name in (
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = getattr(config, name)
        if value is not None:
            if name not in mapping:
                raise GeminiConversionError(
                    f"{destination} cannot represent generationConfig.{name}"
                )
            body[mapping[name]] = value
    if config.stop_sequences:
        if destination == "responses":
            raise GeminiConversionError("Responses cannot represent stopSequences")
        body["stop_sequences" if destination == "messages" else "stop"] = list(
            config.stop_sequences
        )
    if not config.structured:
        return
    schema = config.response_json_schema
    if config.response_schema is not None:
        schema = json_schema(config.response_schema)
    if config.response_format is not None:
        schema = config.response_format.text.schema_
    if schema is None and destination == "messages":
        raise GeminiConversionError("Messages JSON output requires a schema")
    format_: JsonObject = (
        {"type": "json_object"}
        if schema is None
        else {
            "type": "json_schema",
            "name": "gemini_output",
            "schema": deepcopy(schema),
        }
    )
    if destination == "responses":
        body["text"] = {"format": format_}
    elif destination == "messages":
        body["output_config"] = {
            "format": {"type": "json_schema", "schema": deepcopy(schema)}
        }
    else:
        body["response_format"] = (
            {
                "type": "json_schema",
                "json_schema": {"name": "gemini_output", "schema": deepcopy(schema)},
            }
            if schema is not None
            else format_
        )


def build_gemini_chat_request(request: GeminiRequest) -> GeminiPreparedRequest:
    return _build(request, "chat")


def build_gemini_messages_request(
    request: GeminiRequest, options: NativeMessagesOptions
) -> GeminiPreparedRequest:
    prepared = _build(request, "messages")
    prepared.body["max_tokens"] = options.max_tokens
    if options.thinking is not None:
        prepared.body["thinking"] = deepcopy(options.thinking)
    if options.output_effort is not None:
        config = prepared.body.setdefault("output_config", {})
        assert isinstance(config, dict)
        config["effort"] = options.output_effort
    return prepared


def build_gemini_responses_request(
    request: GeminiRequest, reasoning: ReasoningPolicy
) -> GeminiPreparedRequest:
    prepared = _build(request, "responses")
    if config := responses_reasoning_config(reasoning):
        prepared.body["reasoning"] = dict(config)
    prepared.body["store"] = False
    prepared.body["include"] = ["reasoning.encrypted_content"]
    return prepared
