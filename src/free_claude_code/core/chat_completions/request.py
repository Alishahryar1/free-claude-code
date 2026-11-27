"""Preserve native Chat requests and translate only representable fields."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from free_claude_code.core.json_types import load_json_value

from ..history_replay import (
    decode_replay,
    google_thought_signature,
    is_replay,
    readable_reasoning,
)
from ..json_types import (
    JsonObject,
    JsonValue,
    json_array,
    json_object,
    json_objects,
    json_text,
)


class ChatConversionError(ValueError):
    """The destination cannot represent a Chat request without data loss."""


def _validate_tool_name(tool: JsonObject) -> None:
    kind = tool.get("type")
    if kind in ("function", "custom"):
        declaration = tool.get(str(kind))
        if not isinstance(declaration, dict) or not (
            isinstance(name := declaration.get("name"), str) and name.strip()
        ):
            raise ValueError("Function and custom tools require a name")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)

    model: str = Field(min_length=1)
    messages: list[JsonObject] = Field(min_length=1)
    stream: bool = False
    reasoning_effort: str | None = None
    n: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stream_options: JsonObject | None = None
    tools: list[JsonObject] | None = None
    tool_choice: str | JsonObject | None = None
    response_format: JsonObject | None = None

    @model_validator(mode="after")
    def validate_messages(self) -> Self:
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.stream_options is not None and (
            "include_usage" in self.stream_options
            and not isinstance(self.stream_options["include_usage"], (bool, type(None)))
        ):
            raise ValueError("stream_options.include_usage must be a boolean")
        for tool in self.tools or []:
            _validate_tool_name(tool)
        choice = self.tool_choice
        if isinstance(choice, str) and choice not in {"auto", "none", "required"}:
            raise ValueError("Unsupported Chat tool_choice")
        if isinstance(choice, dict):
            chosen = choice.get(str(choice.get("type")))
            if choice.get("type") == "allowed_tools":
                if (
                    not isinstance(chosen, dict)
                    or chosen.get("mode") not in ("auto", "required")
                    or not isinstance(allowed := chosen.get("tools"), list)
                ):
                    raise ValueError("allowed_tools requires a mode and tool list")
                for entry in allowed:
                    if not isinstance(entry, dict):
                        raise ValueError("allowed_tools entries must be objects")
                    _validate_tool_name(entry)
            else:
                _validate_tool_name(choice)
        if (
            self.response_format is not None
            and self.response_format.get("type") == "json_schema"
        ):
            specification = self.response_format.get("json_schema")
            if not isinstance(specification, dict) or not isinstance(
                specification.get("schema"), dict
            ):
                raise ValueError("json_schema response_format requires a schema object")
        for message in self.messages:
            if message.get("role") not in {
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
                "function",
            }:
                raise ValueError("Unsupported Chat message role")
            refusal = message.get("refusal")
            if refusal is not None and (
                message.get("role") != "assistant" or not isinstance(refusal, str)
            ):
                raise ValueError("Refusal history requires an assistant string")
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)):
                raise ValueError("Chat content must be a string, array or null")
            if message.get("role") == "tool" and not (
                isinstance(result_id := message.get("tool_call_id"), str) and result_id
            ):
                raise ValueError("Tool messages require a nonempty string tool_call_id")
            for part in content_parts(content):
                if part.get("type") == "text" and not isinstance(part.get("text"), str):
                    raise ValueError("Text content requires text")
                if part.get("type") == "image_url":
                    image = part.get("image_url")
                    if not isinstance(image, dict) or not isinstance(
                        image.get("url"), str
                    ):
                        raise ValueError("image_url requires a URL")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be an array")
            for call in calls:
                if not isinstance(call, dict) or not isinstance(
                    kind := call.get("type"), str
                ):
                    raise ValueError("Tool calls require a type")
                if kind in {"function", "custom"} and not (
                    isinstance(call.get("id"), str) and call["id"]
                ):
                    raise ValueError("Tool calls require an ID")
                function = call.get(kind)
                if kind in {"function", "custom"} and (
                    not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function["name"]
                    or not isinstance(
                        function.get("arguments" if kind == "function" else "input"),
                        str,
                    )
                ):
                    raise ValueError("Tool calls require a name and string arguments")
            details = message.get("reasoning_details") or []
            if not isinstance(details, list) or any(
                not isinstance(detail, dict) for detail in details
            ):
                raise ValueError("reasoning_details must be an array of objects")
            for detail in details:
                assert isinstance(detail, dict)
                if detail.get("type") == "reasoning.encrypted" and not isinstance(
                    detail.get("data"), str
                ):
                    raise ValueError("Encrypted reasoning requires data")
        return self

    def body(self) -> JsonObject:
        return self.model_dump(mode="json", exclude_unset=True)

    @property
    def include_usage(self) -> bool:
        options = self.body().get("stream_options")
        return not self.stream or (
            isinstance(options, dict) and options.get("include_usage") is True
        )

    def reserved_tool_ids(self) -> frozenset[str]:
        return frozenset(
            call_id
            for message in self.messages
            for call in json_objects(message.get("tool_calls") or [])
            if isinstance(call_id := call.get("id"), str) and call_id
        )


@dataclass(frozen=True, slots=True)
class ChatProjection:
    body: JsonObject
    call_kinds: Mapping[str, str]


def project_chat_request(request: ChatRequest) -> ChatProjection:
    """Project the shared Chat input before building a foreign request."""
    body = request.body()
    if request.n not in (None, 1):
        raise ChatConversionError("This destination supports only n=1")
    modalities = body.get("modalities")
    if isinstance(modalities, list) and "audio" in modalities:
        raise ChatConversionError("Audio output requires native Chat egress")
    tools = [
        tool
        for tool in json_objects(body.get("tools") or [])
        if tool.get("type") == "function"
    ]
    choice = request.tool_choice
    if isinstance(choice, dict) and choice.get("type") == "allowed_tools":
        allowed = json_object(choice["allowed_tools"])
        names = {
            json_object(tool["function"]).get("name")
            for tool in json_objects(allowed["tools"])
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict)
        }
        tools = [
            tool for tool in tools if json_object(tool["function"])["name"] in names
        ]
        choice = json_text(allowed["mode"])
    legacy_choice = body.get("function_call")
    if legacy_choice == "none":
        if choice not in (None, "auto", "none"):
            raise ChatConversionError("function_call=none conflicts with tool_choice")
        choice = "none"
    elif legacy_choice not in (None, "auto"):
        raise ChatConversionError(
            "A forced legacy function_call requires native Chat egress"
        )
    if isinstance(choice, dict):
        chosen = choice.get("function")
        if (
            choice.get("type") != "function"
            or not isinstance(chosen, dict)
            or not any(
                json_object(tool["function"])["name"] == chosen.get("name")
                for tool in tools
            )
        ):
            raise ChatConversionError(
                "Named tool_choice requires an available function"
            )
    if choice == "required" and not tools:
        raise ChatConversionError("Required tool_choice has no available functions")
    body["tools"] = tools
    body["tool_choice"] = choice if tools else None
    if not tools or choice == "none":
        body.pop("parallel_tool_calls", None)

    messages = list(json_objects(body["messages"]))
    latest_user = next(
        (msg for msg in reversed(messages) if msg["role"] == "user"), None
    )
    pending_custom: set[str] = set()
    call_kinds: dict[str, str] = {}
    projected: list[JsonObject] = []
    for message in messages:
        if message["role"] == "assistant" and message.get("audio") is not None:
            raise ChatConversionError(
                "Assistant audio history requires native Chat egress"
            )
        if message["role"] == "assistant" and google_thought_signature(
            message.get("extra_content")
        ):
            raise ChatConversionError(
                "Signed Google assistant messages require native Chat egress"
            )
        if message["role"] == "function" or (
            message["role"] == "assistant" and message.get("function_call") is not None
        ):
            raise ChatConversionError(
                "Legacy function call history requires native Chat egress"
            )
        calls = list(json_objects(message.get("tool_calls") or []))
        if calls:
            for call in calls:
                if google_thought_signature(call.get("extra_content")):
                    raise ChatConversionError(
                        "Signed Google tool calls require native Chat egress"
                    )
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    if call_id in call_kinds:
                        raise ChatConversionError(
                            "Duplicate tool call ID in Chat history"
                        )
                    kind = json_text(call["type"])
                    call_kinds[call_id] = kind
                    if kind == "custom":
                        pending_custom.add(call_id)
            message["tool_calls"] = [
                call for call in calls if call.get("type") in ("function", "custom")
            ]
        if message["role"] == "tool":
            result_id = json_text(message["tool_call_id"])
            if call_kinds.get(result_id) not in (None, "function", "custom"):
                continue
            if (
                call_kinds.get(result_id) == "custom"
                and result_id not in pending_custom
            ):
                raise ChatConversionError(
                    "Duplicate custom tool result in Chat history"
                )
            pending_custom.discard(result_id)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in json_objects(content):
                if part.get("type") in ("text", "image_url"):
                    parts.append(part)
                elif (
                    part.get("type") == "file"
                    and isinstance(file := part.get("file"), dict)
                    and file.get("file_id") is not None
                ):
                    raise ChatConversionError(
                        "A stored file requires native Chat egress"
                    )
                elif (
                    message["role"] == "assistant"
                    and part.get("type") == "refusal"
                    and isinstance(refusal := part.get("refusal"), str)
                ):
                    parts.append({"type": "text", "text": refusal})
            message["content"] = parts
            if message["role"] == "tool" and content and not parts:
                raise ChatConversionError("The tool result has no translatable content")
            if message is latest_user and not any(
                part.get("type") == "image_url" or part.get("text") for part in parts
            ):
                raise ChatConversionError(
                    "The latest user message has no translatable content"
                )
        elif message is latest_user and not content:
            raise ChatConversionError(
                "The latest user message has no translatable content"
            )
        if (
            not isinstance(content, str)
            and not message.get("content")
            and message["role"] != "tool"
            and not (
                message["role"] == "assistant"
                and any(
                    message.get(field)
                    for field in (
                        "tool_calls",
                        "refusal",
                        "reasoning",
                        "reasoning_content",
                        "reasoning_details",
                    )
                )
            )
        ):
            continue
        projected.append(message)
    if pending_custom:
        raise ChatConversionError("Unresolved custom tool call in Chat history")
    if not any(message["role"] not in ("system", "developer") for message in projected):
        raise ChatConversionError("Chat request has no translatable conversation")
    body["messages"] = projected
    return ChatProjection(body, call_kinds)


def content_parts(value: JsonValue) -> list[JsonObject]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise ChatConversionError("Chat content must contain objects")
    parts: list[JsonObject] = []
    for part in value:
        if not isinstance(part, dict):
            raise ChatConversionError("Chat content must contain objects")
        parts.append(deepcopy(part))
    return parts


def function_history(message: JsonObject) -> Iterator[tuple[str, str, JsonObject]]:
    """Represent completed free-form calls on protocols with object arguments."""
    for call in json_objects(message.get("tool_calls") or []):
        kind = json_text(call["type"])
        definition = json_object(call[kind])
        args: JsonObject = (
            {"input": definition["input"]} if kind == "custom" else tool_arguments(call)
        )
        yield json_text(call["id"]), json_text(definition["name"]), args


def tool_arguments(call: JsonObject) -> JsonObject:
    try:
        args = load_json_value(json_text(json_object(call["function"])["arguments"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ChatConversionError("Function arguments must be JSON objects") from exc
    if not isinstance(args, dict):
        raise ChatConversionError("Function arguments must be JSON objects")
    return args


def _image_source(part: JsonObject) -> JsonObject:
    image = part.get("image_url")
    if not isinstance(image, dict) or not isinstance(url := image.get("url"), str):
        raise ChatConversionError("image_url requires a URL")
    if image.get("detail") not in (None, "auto"):
        raise ChatConversionError("Image detail requires native Chat egress")
    if url.startswith("data:") and ";base64," in url:
        media, data = url[5:].split(";base64,", 1)
        return {"type": "base64", "media_type": media, "data": data}
    return {"type": "url", "url": url}


def _messages_content(value: JsonValue) -> list[JsonValue]:
    result: list[JsonValue] = []
    for part in content_parts(value):
        if part.get("type") == "text":
            result.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            result.append({"type": "image", "source": _image_source(part)})
        else:
            raise ChatConversionError(f"Unsupported Chat content: {part.get('type')}")
    return result


def _reasoning_items(message: JsonObject) -> list[JsonObject]:
    details = json_objects(message.get("reasoning_details") or [])
    readable: list[tuple[JsonValue, str, bool]] = [
        (None, text, False)
        for key in ("reasoning_content", "reasoning")
        if isinstance(text := message.get(key), str) and text
    ]
    carriers: list[JsonObject] = []
    for detail in details:
        kind = detail.get("type")
        if kind == "reasoning.encrypted":
            carriers.append(detail)
        elif kind in ("reasoning.text", "reasoning.summary"):
            if detail.get("signature"):
                raise ChatConversionError(
                    "Signed reasoning details require native Chat egress"
                )
            readable.extend(
                (detail.get("id"), text, summary)
                for text, summary in readable_reasoning({"reasoning_details": [detail]})
            )
        elif not isinstance(kind, str):
            raise ChatConversionError("Reasoning detail requires a type")
    for detail in carriers:
        data = detail.get("data")
        if not is_replay(data):
            continue
        native = readable_reasoning(decode_replay(data).native)
        source = detail.get("id")
        if source is not None:
            # Source IDs associate streamed readable deltas with their final native
            # record. Edited echoes must not silently replay the original text.
            for summary in (False, True):
                echoed = "".join(
                    text
                    for owner, text, kind in readable
                    if owner == source and kind == summary
                )
                original = "".join(text for text, kind in native if kind == summary)
                if echoed and echoed != original:
                    raise ChatConversionError(
                        "Readable reasoning changed its protected replay record"
                    )
            readable = [part for part in readable if part[0] != source]
        else:
            # Older Chat writers emit one scalar echo with an unlabelled carrier.
            represented = {
                ("".join(text for text, kind in native if kind == summary), summary)
                for summary in (False, True)
            }
            readable = [
                (owner, text, summary)
                for owner, text, summary in readable
                if owner is not None or (text, summary) not in represented
            ]
    items: list[JsonObject] = []
    for _, text, summary in readable:
        items.append(
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text}] if summary else [],
                **(
                    {}
                    if summary
                    else {"content": [{"type": "reasoning_text", "text": text}]}
                ),
            }
        )
    items.extend(
        {"type": "reasoning", "encrypted_content": detail["data"]}
        for detail in carriers
    )
    return items


def _tools(body: JsonObject) -> list[JsonObject]:
    result = []
    for tool in json_objects(body.get("tools") or []):
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise ChatConversionError("Only function tools can be translated")
        result.append(
            {
                key: function[key]
                for key in ("name", "description", "parameters", "strict")
                if key in function
            }
        )
    return result


def chat_messages_body(request: ChatRequest) -> JsonObject:
    body = project_chat_request(request).body
    messages: list[JsonObject] = []
    system: list[JsonValue] = []
    for message in json_objects(body["messages"]):
        role = json_text(message["role"])
        if role in {"system", "developer"}:
            parts = _messages_content(message.get("content"))
            if any(json_object(part)["type"] != "text" for part in parts):
                raise ChatConversionError("System instructions must be text")
            system.extend(parts)
            continue
        parts = _messages_content(message.get("content"))
        if message.get("refusal"):
            parts.append({"type": "text", "text": message["refusal"]})
        if role == "tool":
            parts = [
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": parts,
                }
            ]
            role = "user"
        elif role == "assistant":
            from ..history_replay import responses_reasoning_blocks

            parts = [
                block
                for item in _reasoning_items(message)
                for block in responses_reasoning_blocks(item)
            ] + parts
            parts.extend(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": args,
                }
                for call_id, name, args in function_history(message)
            )
        if messages and messages[-1]["role"] == role:
            json_array(messages[-1]["content"]).extend(parts)
        else:
            messages.append({"role": role, "content": parts})
    result: JsonObject = {
        "model": request.model,
        "messages": messages,
        "stream": True,
    }
    if system:
        result["system"] = system
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            result[key] = body[key]
    result["max_tokens"] = (
        body.get("max_completion_tokens") or body.get("max_tokens") or 4096
    )
    if body.get("stop") is not None:
        result["stop_sequences"] = (
            [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        )
    if tools := _tools(body):
        result["tools"] = [
            {
                "name": tool["name"],
                "input_schema": tool.get("parameters", {"type": "object"}),
                **(
                    {"description": tool["description"]}
                    if "description" in tool
                    else {}
                ),
                **({"strict": tool["strict"]} if "strict" in tool else {}),
            }
            for tool in tools
        ]
    choice = body.get("tool_choice")
    if choice is not None:
        result["tool_choice"] = (
            {"type": {"auto": "auto", "none": "none", "required": "any"}[choice]}
            if isinstance(choice, str)
            else {
                "type": "tool",
                "name": json_object(json_object(choice)["function"])["name"],
            }
        )
    if body.get("parallel_tool_calls") is not None:
        choice_payload = json_object(result.setdefault("tool_choice", {"type": "auto"}))
        choice_payload["disable_parallel_tool_use"] = not body["parallel_tool_calls"]
    if fmt := request.response_format:
        if fmt.get("type") == "json_schema":
            result["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": json_object(fmt["json_schema"])["schema"],
                }
            }
        elif fmt.get("type") != "text":
            raise ChatConversionError(
                "Messages requires json_schema for structured output"
            )
    return result


def _responses_content(value: JsonValue, *, role: str) -> list[JsonObject]:
    parts = []
    for part in content_parts(value):
        if part.get("type") == "text":
            parts.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": part["text"],
                }
            )
        elif part.get("type") == "image_url" and role != "assistant":
            image = json_object(part["image_url"])
            parts.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    **({"detail": image["detail"]} if "detail" in image else {}),
                }
            )
        else:
            raise ChatConversionError(f"Unsupported Chat content: {part.get('type')}")
    return parts


def chat_responses_body(request: ChatRequest) -> JsonObject:
    projected = project_chat_request(request)
    body = projected.body
    if body.get("stop"):
        raise ChatConversionError("Responses cannot represent stop sequences")
    items: list[JsonObject] = []
    for message in json_objects(body["messages"]):
        role = json_text(message["role"])
        if role == "tool":
            call_id = json_text(message["tool_call_id"])
            output = message.get("content") or ""
            if isinstance(output, list):
                output = _responses_content(output, role="tool")
            items.append(
                {
                    "type": "custom_tool_call_output"
                    if projected.call_kinds.get(call_id) == "custom"
                    else "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
            continue
        if role == "assistant":
            items.extend(_reasoning_items(message))
        parts = _responses_content(message.get("content"), role=role)
        if message.get("refusal"):
            parts.append({"type": "refusal", "refusal": message["refusal"]})
        if parts:
            items.append({"role": role, "content": parts})
        for call in json_objects(message.get("tool_calls") or []):
            kind = json_text(call["type"])
            detail = json_object(call[kind])
            call_id = json_text(call["id"])
            field = "input" if kind == "custom" else "arguments"
            items.append(
                {
                    "type": "custom_tool_call" if kind == "custom" else "function_call",
                    "call_id": call_id,
                    "name": detail["name"],
                    field: detail[field],
                }
            )
    result: JsonObject = {
        "model": request.model,
        "input": items,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    for key in (
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "metadata",
        "store",
        "service_tier",
        "prompt_cache_key",
        "user",
    ):
        if body.get(key) is not None:
            result[key] = body[key]
    if limit := body.get("max_completion_tokens") or body.get("max_tokens"):
        result["max_output_tokens"] = limit
    if tools := _tools(body):
        # Responses defaults to strict normalization; Chat defaults to non-strict.
        result["tools"] = [
            {
                "type": "function",
                **tool,
                "strict": tool["strict"] if tool.get("strict") is not None else False,
            }
            for tool in tools
        ]
    if (choice := body.get("tool_choice")) is not None:
        result["tool_choice"] = (
            choice
            if isinstance(choice, str)
            else {
                "type": "function",
                "name": json_object(json_object(choice)["function"])["name"],
            }
        )
    if body.get("reasoning_effort") is not None:
        result["reasoning"] = {"effort": body["reasoning_effort"]}
    if fmt := request.response_format:
        kind = fmt.get("type")
        if kind == "json_schema":
            specification = json_object(fmt["json_schema"])
            result["text"] = {
                "format": {
                    "type": kind,
                    **{
                        key: specification[key]
                        for key in ("name", "description", "schema", "strict")
                        if key in specification
                    },
                }
            }
        elif kind in ("text", "json_object"):
            result["text"] = {"format": {"type": kind}}
        else:
            raise ChatConversionError("Unsupported response_format")
    return result
