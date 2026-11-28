"""Preserve native Chat requests and translate only representable fields."""

from copy import deepcopy
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from free_claude_code.core.json_types import load_json_value

from ..history_replay import decode_replay, is_replay, readable_reasoning
from ..json_types import (
    JsonObject,
    JsonValue,
    json_array,
    json_object,
    json_objects,
    json_text,
)
from ..openai_request_hints import OPENAI_REQUEST_HINTS


class ChatConversionError(ValueError):
    """The destination cannot represent a Chat request without data loss."""


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
            kind = tool.get("type")
            declaration = tool.get(str(kind))
            if kind in {"function", "custom"} and (
                not isinstance(declaration, dict)
                or not isinstance(name := declaration.get("name"), str)
                or not name.strip()
            ):
                raise ValueError("Tool declarations require a name")
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
                    or any(not isinstance(tool, dict) for tool in allowed)
                ):
                    raise ValueError("allowed_tools requires a mode and tool list")
            elif (
                not isinstance(chosen, dict)
                or not isinstance(chosen.get("name"), str)
                or not chosen["name"]
            ):
                raise ValueError("Named tool_choice requires a tool name")
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
            if message.get("role") == "tool" and not message.get("tool_call_id"):
                raise ValueError("Tool messages require tool_call_id")
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
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                ):
                    raise ValueError("Tool calls require an ID")
                kind = call.get("type")
                function = call.get(str(kind))
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
            json_text(call["id"])
            for message in self.messages
            for call in function_calls(message)
        )


def translated_body(request: ChatRequest, *, supported: set[str]) -> JsonObject:
    """Reject unsupported controls before any upstream attempt."""
    body = request.body()
    common = {"model", "messages", "stream", "stream_options", "n"}
    unsupported = [
        key
        for key, value in body.items()
        if key not in common | supported | OPENAI_REQUEST_HINTS and value is not None
    ]
    if set(request.stream_options or {}) - {"include_usage"}:
        unsupported.append("stream_options")
    if (
        isinstance(request.tool_choice, dict)
        and request.tool_choice.get("type") != "function"
    ):
        unsupported.append("tool_choice")
    if body.get("n") not in (None, 1):
        unsupported.append("n")
    if unsupported:
        raise ChatConversionError(
            f"This destination cannot represent Chat fields: {', '.join(sorted(unsupported))}"
        )
    return deepcopy(body)


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


def function_calls(message: JsonObject) -> list[JsonObject]:
    calls = message.get("tool_calls", [])
    if calls is None:
        return []
    if not isinstance(calls, list):
        raise ChatConversionError("tool_calls must be an array")
    result: list[JsonObject] = []
    for call in calls:
        if (
            not isinstance(call, dict)
            or call.get("type") != "function"
            or not isinstance(call.get("function"), dict)
            or not isinstance(call.get("id"), str)
        ):
            raise ChatConversionError("Only named function tool calls are supported")
        if set(call) - {"id", "type", "function"}:
            raise ChatConversionError("Unsupported tool call metadata")
        result.append(call)
    return result


def tool_arguments(call: JsonObject) -> JsonObject:
    try:
        args = load_json_value(json_text(json_object(call["function"])["arguments"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ChatConversionError("Function arguments must be JSON objects") from exc
    if not isinstance(args, dict):
        raise ChatConversionError("Function arguments must be JSON objects")
    return args


def validate_message_fields(message: JsonObject) -> None:
    unsupported = {key for key, value in message.items() if value is not None} - {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "refusal",
    }
    if unsupported:
        raise ChatConversionError(
            f"Unsupported Chat message fields: {sorted(unsupported)}"
        )
    if message["role"] == "function":
        raise ChatConversionError("Legacy function messages require native Chat egress")


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
    # The detail array is in stream order. Keep each item's position so an
    # echo restoring signed and unsigned blocks does not reverse their order.
    readable: list[tuple[int, JsonValue | None, str, bool]] = [
        (-1, None, text, False)
        for key in ("reasoning_content", "reasoning")
        if isinstance(text := message.get(key), str) and text
    ]
    carriers: list[tuple[int, JsonObject]] = []
    for position, detail in enumerate(details):
        kind = detail.get("type")
        if kind == "reasoning.encrypted":
            carriers.append((position, detail))
        elif kind in ("reasoning.text", "reasoning.summary"):
            if detail.get("signature"):
                raise ChatConversionError(
                    "Signed reasoning details require native Chat egress"
                )
            readable.extend(
                (position, detail.get("id"), text, summary)
                for text, summary in readable_reasoning({"reasoning_details": [detail]})
            )
        else:
            raise ChatConversionError("Unsupported reasoning detail")
    for _, detail in carriers:
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
                    for _, owner, text, kind in readable
                    if owner == source and kind == summary
                )
                original = "".join(text for text, kind in native if kind == summary)
                if echoed and echoed != original:
                    raise ChatConversionError(
                        "Readable reasoning changed its protected replay record"
                    )
            readable = [part for part in readable if part[1] != source]
        else:
            # Older Chat writers emit one scalar echo with an unlabelled carrier.
            represented = {
                ("".join(text for text, kind in native if kind == summary), summary)
                for summary in (False, True)
            }
            readable = [
                part
                for part in readable
                if part[1] is not None or (part[2], part[3]) not in represented
            ]
    rendered: list[tuple[int, int, JsonObject]] = []
    for position, _, text, summary in readable:
        rendered.append(
            (
                position,
                0,
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": text}]
                    if summary
                    else [],
                    **(
                        {}
                        if summary
                        else {"content": [{"type": "reasoning_text", "text": text}]}
                    ),
                },
            )
        )
    for position, detail in carriers:
        rendered.append(
            (
                position,
                1,
                {"type": "reasoning", "encrypted_content": detail["data"]},
            )
        )
    return [
        item for _, _, item in sorted(rendered, key=lambda entry: (entry[0], entry[1]))
    ]


def _tools(body: JsonObject) -> list[JsonObject]:
    result = []
    for tool in json_objects(body.get("tools") or []):
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise ChatConversionError("Only function tools can be translated")
        if set(function) - {"name", "description", "parameters", "strict"}:
            raise ChatConversionError("Unsupported function declaration fields")
        result.append(function)
    return result


def chat_messages_body(request: ChatRequest) -> JsonObject:
    body = translated_body(
        request,
        supported={
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "stop",
            "reasoning_effort",
            "response_format",
        },
    )
    messages: list[JsonObject] = []
    system: list[JsonValue] = []
    for message in json_objects(body["messages"]):
        validate_message_fields(message)
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
                    "id": call["id"],
                    "name": json_object(call["function"])["name"],
                    "input": tool_arguments(call),
                }
                for call in function_calls(message)
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
    body = translated_body(
        request,
        supported={
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "reasoning_effort",
            "response_format",
            "metadata",
            "store",
            "service_tier",
        },
    )
    items: list[JsonObject] = []
    for message in json_objects(body["messages"]):
        validate_message_fields(message)
        role = json_text(message["role"])
        if role == "tool":
            output = message.get("content") or ""
            if isinstance(output, list):
                output = _responses_content(output, role="tool")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
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
        items.extend(
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": json_object(call["function"])["name"],
                "arguments": json_object(call["function"])["arguments"],
            }
            for call in function_calls(message)
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
        *sorted(OPENAI_REQUEST_HINTS),
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
        result["text"] = {
            "format": {"type": "json_schema", **json_object(fmt["json_schema"])}
            if fmt.get("type") == "json_schema"
            else fmt
        }
    return result
