"""Request construction for native GenerateContent, without a Chat intermediate."""

from collections.abc import Mapping
from copy import deepcopy

from ..anthropic.conversion import (
    resolve_anthropic_tool_choice,
    unrepresentable_client_tool_constraints,
)
from ..anthropic.image_sources import portable_anthropic_image_url
from ..anthropic.models import MessagesRequest
from ..anthropic.request_serialization import (
    dump_messages_request,
    is_noop_context_management,
    translated_messages,
)
from ..chat_completions.request import (
    ChatRequest,
    content_parts,
    function_calls,
    tool_arguments,
    translated_body,
)
from ..history_replay import (
    decode_replay,
    is_replay,
    messages_replay_key,
    protected_reasoning,
    readable_reasoning,
    resolve_messages_replay,
)
from ..json_types import (
    JsonObject,
    JsonValue,
    json_array,
    json_object,
    json_objects,
    json_text,
)
from ..openai_responses import (
    OpenAIResponsesRequest,
    is_hosted_history,
    latest_user_input,
)
from ..reasoning import ReasoningControl, ReasoningPolicy
from .errors import GeminiConversionError
from .models import GeminiRequest, InlineData, ThinkingConfig
from .web_search_replay import restore_web_search_history


def _image(url: str) -> JsonObject:
    if not url.startswith("data:") or ";base64," not in url:
        raise GeminiConversionError("Native Gemini requires inline base64 images")
    media, data = url[5:].split(";base64,", 1)
    image = InlineData.model_validate({"mimeType": media, "data": data})
    return {"inlineData": image.model_dump(by_alias=True)}


def _text_parts(value: JsonValue, *, responses: bool = False) -> list[JsonObject]:
    parts: list[JsonObject] = []
    for part in content_parts(value):
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            if not isinstance(part.get("text"), str):
                raise GeminiConversionError("Text content requires a string")
            parts.append({"text": part["text"]})
        elif kind in ("image_url", "input_image"):
            image = part.get("image_url")
            url = (
                image
                if responses
                else image.get("url")
                if isinstance(image, dict)
                else None
            )
            detail = (
                part.get("detail")
                if responses
                else image.get("detail")
                if isinstance(image, dict)
                else None
            )
            if detail not in (None, "auto"):
                raise GeminiConversionError(
                    "Native Gemini cannot represent image detail"
                )
            if not isinstance(url, str):
                raise GeminiConversionError("Image requires a data URL")
            parts.append(_image(url))
        elif not isinstance(kind, str):
            raise GeminiConversionError("Content block requires a type")
    return parts


def _append(contents: list[JsonObject], role: str, parts: list[JsonObject]) -> None:
    if not parts:
        return
    if contents and contents[-1]["role"] == role:
        json_array(contents[-1]["parts"]).extend(parts)
    else:
        contents.append({"role": role, "parts": parts})


def _reasoning(
    item: Mapping[str, JsonValue], carrier: JsonValue = None
) -> list[JsonObject]:
    if carrier:
        if not is_replay(carrier):
            # Foreign provider native state is never a Google thought signature.
            raise GeminiConversionError(
                "Native Gemini cannot replay another provider's protected reasoning"
            )
        record = decode_replay(carrier)
        if not protected_reasoning(record.native):
            return [
                {"text": text, "thought": True}
                for text, _ in readable_reasoning(record.native)
            ]
        return [{"thoughtSignature": carrier}]
    return [{"text": text, "thought": True} for text, _ in readable_reasoning(item)]


def apply_reasoning(
    body: JsonObject, policy: ReasoningPolicy, *, native: bool = False
) -> None:
    config = json_object(body.setdefault("generationConfig", {}))
    if native and policy == ReasoningPolicy.provider_default():
        return
    thinking = json_object(config.setdefault("thinkingConfig", {}))
    if (
        native
        and policy.control is not ReasoningControl.OFF
        and policy == ThinkingConfig.model_validate(thinking).policy()
    ):
        return
    # Visibility is a separate native choice. Cross-protocol callers request
    # summaries whenever reasoning output is allowed.
    if not native:
        thinking["includeThoughts"] = policy.output_enabled
    if policy.control is ReasoningControl.OFF:
        thinking.pop("thinkingLevel", None)
        thinking.update(thinkingBudget=0, includeThoughts=False)
    elif (budget := policy.numeric_budget_tokens) is not None:
        thinking.pop("thinkingLevel", None)
        thinking["thinkingBudget"] = budget


def _tool_config(
    choice: JsonValue, declarations: list[JsonObject], *, responses: bool = False
) -> JsonObject | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        mode = {"auto": "AUTO", "none": "NONE", "required": "ANY", "any": "ANY"}.get(
            choice
        )
        if mode is None:
            raise GeminiConversionError("Unsupported tool_choice")
        if not declarations:
            if mode == "ANY":
                raise GeminiConversionError(
                    "Required tool choice has no available functions"
                )
            return None
        return {"functionCallingConfig": {"mode": mode}}
    if isinstance(choice, dict):
        name = (
            choice.get("name")
            if responses
            else json_object(choice.get("function", {})).get("name")
        )
        if (
            choice.get("type") in ("function", "tool")
            and isinstance(name, str)
            and any(tool["name"] == name for tool in declarations)
        ):
            return {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}
            }
    raise GeminiConversionError("Unsupported tool_choice")


def _format(config: JsonObject, format_: JsonValue) -> None:
    if not isinstance(format_, dict):
        raise GeminiConversionError("Structured output format must be an object")
    kind = format_.get("type")
    if kind == "text":
        return
    if kind == "json_object":
        config["responseMimeType"] = "application/json"
    elif kind == "json_schema":
        schema_config = format_.get("json_schema", format_)
        if not isinstance(schema_config, dict) or not isinstance(
            schema := schema_config.get("schema"), dict
        ):
            raise GeminiConversionError(
                "Structured output requires a JSON schema object"
            )
        config.update(
            responseMimeType="application/json",
            responseJsonSchema=deepcopy(schema),
        )
    else:
        raise GeminiConversionError("Unsupported structured output format")


def _function_result(
    call_id: str, name: str, value: JsonValue, *, responses: bool = False
) -> JsonObject:
    parts = _text_parts(value, responses=responses)
    text = "\n".join(json_text(part["text"]) for part in parts if "text" in part)
    result: JsonObject = {"id": call_id, "name": name, "response": {"output": text}}
    images = [part for part in parts if "inlineData" in part]
    if images:
        result["parts"] = images
    return {"functionResponse": result}


def chat_to_gemini(
    request: ChatRequest, *, allow_pending_calls: bool = False
) -> JsonObject:
    body = translated_body(request)
    contents: list[JsonObject] = []
    system: list[JsonObject] = []
    calls: dict[str, str] = {}
    for message in json_objects(body["messages"]):
        role = json_text(message["role"])
        if role in {"system", "developer"}:
            parts = _text_parts(message.get("content"))
            if any("text" not in part for part in parts):
                raise GeminiConversionError("System instructions must contain text")
            system.extend(parts)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in calls:
                raise GeminiConversionError("Orphaned tool response")
            _append(
                contents,
                "user",
                [_function_result(call_id, calls.pop(call_id), message.get("content"))],
            )
            continue
        parts = []
        if role == "assistant":
            reasoning = {
                key: message[key]
                for key in ("reasoning", "reasoning_content", "reasoning_details")
                if key in message
            }
            parts.extend(_reasoning(reasoning))
            for detail in json_objects(message.get("reasoning_details") or []):
                if detail.get("type") == "reasoning.encrypted":
                    parts.extend(_reasoning(detail, detail.get("data")))
                elif detail.get("type") in (
                    "reasoning.text",
                    "reasoning.summary",
                ) and detail.get("signature"):
                    raise GeminiConversionError(
                        "Unsupported protected reasoning detail"
                    )
        parts.extend(_text_parts(message.get("content")))
        if message.get("refusal"):
            parts.append({"text": message["refusal"]})
        for call in function_calls(message):
            name = json_text(json_object(call["function"])["name"])
            call_id = json_text(call["id"])
            if call_id in calls:
                raise GeminiConversionError("Duplicate function call ID")
            calls[call_id] = name
            parts.append(
                {
                    "functionCall": {
                        "id": call_id,
                        "name": name,
                        "args": tool_arguments(call),
                    }
                }
            )
        _append(contents, "model" if role == "assistant" else "user", parts)
    if calls and not allow_pending_calls:
        raise GeminiConversionError("Unresolved tool calls in history")
    config: JsonObject = {}
    for source, target in {
        "temperature": "temperature",
        "top_p": "topP",
        "seed": "seed",
        "presence_penalty": "presencePenalty",
        "frequency_penalty": "frequencyPenalty",
    }.items():
        if body.get(source) is not None:
            config[target] = body[source]
    if limit := body.get("max_completion_tokens") or body.get("max_tokens"):
        config["maxOutputTokens"] = limit
    if body.get("stop") is not None:
        config["stopSequences"] = (
            [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        )
    if fmt := body.get("response_format"):
        _format(config, fmt)
    result: JsonObject = {
        "model": request.model,
        "contents": contents,
        "generationConfig": config,
    }
    if system:
        result["systemInstruction"] = {"parts": system}
    declarations = []
    for tool in json_objects(body.get("tools") or []):
        if tool.get("type") != "function":
            continue
        function = json_object(tool["function"])
        if function.get("strict"):
            raise GeminiConversionError(
                "Native Gemini cannot enforce strict function tools"
            )
        declarations.append(
            {
                "name": function["name"],
                "parametersJsonSchema": function.get("parameters", {"type": "object"}),
                **(
                    {"description": function["description"]}
                    if "description" in function
                    else {}
                ),
            }
        )
    if declarations:
        result["tools"] = [{"functionDeclarations": declarations}]
    if tool_config := _tool_config(body.get("tool_choice"), declarations):
        result["toolConfig"] = tool_config
    if body.get("parallel_tool_calls") is False:
        raise GeminiConversionError(
            "Native Gemini cannot enforce parallel_tool_calls=false"
        )
    return result


def messages_to_gemini(
    request: MessagesRequest, *, allow_pending_calls: bool = False
) -> JsonObject:
    body = dump_messages_request(request)
    if not is_noop_context_management(request.context_management):
        raise GeminiConversionError(
            "Native Gemini cannot apply context_management edits"
        )
    contents: list[JsonObject] = []
    system: list[JsonObject] = []
    if body.get("system"):
        system = [
            {"text": part["text"]}
            for part in content_parts(body["system"])
            if part.get("type") == "text"
        ]
    calls: dict[str, str] = {}
    result_parts: list[JsonObject] | None = None
    result_order: dict[str, int] = {}
    turns = [
        message.model_dump(mode="json", exclude_unset=True)
        for message in translated_messages(request)
    ]
    for message in restore_web_search_history(turns):
        role = json_text(message["role"])
        parts: list[JsonObject] = []
        if message.get("reasoning_content"):
            parts.append({"text": message["reasoning_content"], "thought": True})
        for block in resolve_messages_replay(content_parts(message.get("content"))):
            if not isinstance(block, dict):
                raise GeminiConversionError("Invalid Messages content")
            kind = block.get("type")
            if kind == "text":
                parts.append({"text": block["text"]})
            elif kind == "image":
                parts.append(_image(portable_anthropic_image_url(block["source"])))
            elif kind in ("thinking", "redacted_thinking"):
                key = messages_replay_key(block)
                parts.extend(_reasoning(block, block.get(key) if key else None))
            elif kind == "tool_use":
                if block.get("extra_content"):
                    raise GeminiConversionError(
                        "Saved compatibility tool metadata cannot be replayed on native Gemini; start a fresh conversation"
                    )
                call_id, name = str(block["id"]), str(block["name"])
                if call_id in calls:
                    raise GeminiConversionError("Duplicate pending tool call ID")
                calls[call_id] = name
                parts.append(
                    {
                        "functionCall": {
                            "id": call_id,
                            "name": name,
                            "args": block["input"],
                        }
                    }
                )
            elif kind == "tool_result":
                call_id = str(block["tool_use_id"])
                if call_id not in calls:
                    raise GeminiConversionError("Orphaned tool result")
                if parts:
                    raise GeminiConversionError(
                        "Tool results must precede other user content"
                    )
                if result_parts is None:
                    result_order = {
                        pending: index for index, pending in enumerate(calls)
                    }
                    result_parts = []
                    contents.append({"role": "user", "parts": result_parts})
                elif contents[-1]["parts"] is not result_parts:
                    raise GeminiConversionError(
                        "Parallel tool results must be consecutive"
                    )
                value = block.get("content")
                output_parts: list[JsonObject] = (
                    [] if isinstance(value, dict) else content_parts(value)
                )
                text = (
                    deepcopy(value)
                    if isinstance(value, dict)
                    else "\n".join(
                        json_text(part["text"])
                        for part in output_parts
                        if part.get("type") == "text"
                    )
                )
                result: JsonObject = {
                    "id": call_id,
                    "name": calls.pop(call_id),
                    "response": {"error" if block.get("is_error") else "output": text},
                }
                images = [
                    _image(portable_anthropic_image_url(part["source"]))
                    for part in output_parts
                    if part.get("type") == "image"
                ]
                if images:
                    result["parts"] = images
                result_parts.append({"functionResponse": result})
            else:
                raise GeminiConversionError(f"Unsupported Messages content {kind!r}")
        if result_parts is not None:
            result_parts.sort(
                key=lambda part: result_order[
                    json_text(json_object(part["functionResponse"])["id"])
                ]
            )
        if parts:
            if role == "user" and calls:
                raise GeminiConversionError(
                    "Unresolved tool calls before user follow-up"
                )
            # A tool result and a later user prompt are distinct Gemini turns.
            contents.append(
                {"role": "model" if role == "assistant" else "user", "parts": parts}
            )
        if not calls:
            result_parts = None
    if calls and not allow_pending_calls:
        raise GeminiConversionError("Unresolved tool calls in history")
    config: JsonObject = {}
    for source, target in {
        "max_tokens": "maxOutputTokens",
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "stop_sequences": "stopSequences",
    }.items():
        if body.get(source) is not None:
            config[target] = body[source]
    output = json_object(body.get("output_config") or {})
    if fmt := output.get("format"):
        _format(config, fmt)
    result: JsonObject = {
        "model": request.model,
        "contents": contents,
        "generationConfig": config,
    }
    if system:
        result["systemInstruction"] = {"parts": system}
    declarations = []
    for tool in request.client_tools:
        if constraints := unrepresentable_client_tool_constraints(tool):
            raise GeminiConversionError(
                f"Native Gemini cannot enforce {constraints} on tool {tool.name!r}"
            )
        declaration: JsonObject = {
            "name": tool.name,
            "parametersJsonSchema": tool.input_schema
            if tool.input_schema is not None
            else {"type": "object"},
        }
        if tool.description is not None:
            declaration["description"] = tool.description
        declarations.append(declaration)
    if declarations:
        result["tools"] = [{"functionDeclarations": declarations}]
    resolved = resolve_anthropic_tool_choice(request.client_tools, request.tool_choice)
    if resolved and request.tool_choice is not None:
        if resolved.get("disable_parallel_tool_use") and resolved.get("type") != "none":
            raise GeminiConversionError(
                "Native Gemini cannot enforce disable_parallel_tool_use"
            )
        choice = resolved if resolved.get("type") == "tool" else resolved["type"]
        if config_choice := _tool_config(choice, declarations, responses=True):
            result["toolConfig"] = config_choice
    return result


def responses_to_gemini(
    request: OpenAIResponsesRequest, *, allow_pending_calls: bool = False
) -> JsonObject:
    request = request.restrict_allowed_tools()
    body: JsonObject = request.model_dump(mode="json", exclude_none=True)
    if body.get("store") or body.get("previous_response_id"):
        raise GeminiConversionError(
            "Native Gemini cannot access stored Responses state"
        )
    value = body.get("input")
    items = (
        [{"role": "user", "content": value}]
        if isinstance(value, str)
        else value
        if isinstance(value, list)
        else [value]
    )
    contents: list[JsonObject] = []
    system: list[JsonObject] = (
        [{"text": body["instructions"]}] if body.get("instructions") else []
    )
    calls: dict[str, str] = {}
    latest = latest_user_input(items)
    for item in items:
        if isinstance(item, str):
            _append(contents, "user", [{"text": item}])
            continue
        if not isinstance(item, dict):
            raise GeminiConversionError("Responses input requires messages or items")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            parts = _text_parts(item.get("content"), responses=True)
            if item is latest and not parts:
                raise GeminiConversionError(
                    "The latest user message has no translatable content"
                )
            if role in ("system", "developer"):
                if any("text" not in part for part in parts):
                    raise GeminiConversionError("System instructions must be text")
                system.extend(parts)
            elif role in ("assistant", "user"):
                _append(contents, "model" if role == "assistant" else "user", parts)
            else:
                raise GeminiConversionError("Unsupported Responses role")
        elif kind == "reasoning":
            _append(contents, "model", _reasoning(item, item.get("encrypted_content")))
        elif kind == "function_call":
            if item.get("extra_content"):
                raise GeminiConversionError(
                    "Saved compatibility tool metadata cannot be replayed on native Gemini; start a fresh conversation"
                )
            if any(
                not isinstance(item.get(key), str) or not item[key]
                for key in ("call_id", "name", "arguments")
            ):
                raise GeminiConversionError(
                    "Function calls require an ID, name, and arguments"
                )
            call_id, name = json_text(item["call_id"]), json_text(item["name"])
            if call_id in calls:
                raise GeminiConversionError("Duplicate pending tool call ID")
            calls[call_id] = name
            args = tool_arguments({"function": {"arguments": item["arguments"]}})
            _append(
                contents,
                "model",
                [{"functionCall": {"id": call_id, "name": name, "args": args}}],
            )
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                raise GeminiConversionError("Function output requires a call ID")
            call_id = item["call_id"]
            if call_id not in calls:
                raise GeminiConversionError("Orphaned function_call_output")
            _append(
                contents,
                "user",
                [
                    _function_result(
                        call_id, calls.pop(call_id), item.get("output"), responses=True
                    )
                ],
            )
        elif kind in ("input_text", "output_text", "text", "input_image"):
            _append(contents, "user", _text_parts([item], responses=True))
        elif kind == "item_reference" or (
            is_hosted_history(kind)
            and item.get("status")
            not in (None, "completed", "failed", "incomplete", "interrupted")
        ):
            raise GeminiConversionError(f"Native Gemini cannot replay {kind!r}")
        elif not isinstance(kind, str):
            raise GeminiConversionError("Responses input item requires a type")
    if not contents and not allow_pending_calls:
        raise GeminiConversionError(
            "Responses request has no translatable conversation"
        )
    if calls and not allow_pending_calls:
        raise GeminiConversionError("Unresolved function calls in history")
    config: JsonObject = {}
    for source, target in {
        "temperature": "temperature",
        "top_p": "topP",
        "max_output_tokens": "maxOutputTokens",
    }.items():
        if body.get(source) is not None:
            config[target] = body[source]
    if (text := body.get("text")) is not None:
        if not isinstance(text, dict):
            raise GeminiConversionError(
                "Native Gemini cannot represent Responses text controls"
            )
        if (fmt := text.get("format")) is not None:
            _format(config, fmt)
    result: JsonObject = {
        "model": request.model,
        "contents": contents,
        "generationConfig": config,
    }
    if system:
        result["systemInstruction"] = {"parts": system}
    declarations = []
    for tool in json_objects(body.get("tools") or []):
        if tool.get("type") != "function":
            continue
        if tool.get("strict"):
            raise GeminiConversionError(
                "Native Gemini cannot enforce strict function tools"
            )
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise GeminiConversionError("Function declarations require a name")
        declarations.append(
            {
                "name": tool["name"],
                "parametersJsonSchema": tool.get("parameters", {"type": "object"}),
                **(
                    {"description": tool["description"]}
                    if tool.get("description")
                    else {}
                ),
            }
        )
    if declarations:
        result["tools"] = [{"functionDeclarations": declarations}]
    if config_choice := _tool_config(
        body.get("tool_choice"), declarations, responses=True
    ):
        result["toolConfig"] = config_choice
    if (
        declarations
        and body.get("tool_choice") != "none"
        and request.parallel_tool_calls is False
    ):
        raise GeminiConversionError(
            "Native Gemini cannot enforce parallel_tool_calls=false"
        )
    return result


def native_gemini_body(request: GeminiRequest) -> JsonObject:
    return request.model_dump(mode="json", by_alias=True, exclude_unset=True)
