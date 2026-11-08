"""Request construction for native GenerateContent, without a Chat intermediate."""

from copy import deepcopy
from typing import Any

from ..anthropic.image_sources import portable_anthropic_image_url
from ..anthropic.models import MessagesRequest
from ..anthropic.request_serialization import (
    dump_messages_request,
    is_noop_context_management,
)
from ..chat_completions.request import (
    ChatRequest,
    content_parts,
    function_calls,
    tool_arguments,
    translated_body,
    validate_message_fields,
)
from ..history_replay import (
    decode_replay,
    is_replay,
    protected_reasoning,
    readable_reasoning,
    resolve_messages_replay,
)
from ..json_types import JsonObject
from ..openai_responses import OpenAIResponsesRequest
from ..reasoning import ReasoningControl, ReasoningPolicy
from .errors import GeminiConversionError
from .models import GeminiRequest, InlineData, ThinkingConfig


def _image(url: str) -> dict[str, Any]:
    if not url.startswith("data:") or ";base64," not in url:
        raise GeminiConversionError("Native Gemini requires inline base64 images")
    media, data = url[5:].split(";base64,", 1)
    image = InlineData.model_validate({"mimeType": media, "data": data})
    return {"inlineData": image.model_dump(by_alias=True)}


def _text_parts(value: Any, *, responses: bool = False) -> list[dict[str, Any]]:
    parts = []
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
        else:
            raise GeminiConversionError(
                f"Native Gemini cannot represent content {kind!r}"
            )
    return parts


def _append(
    contents: list[dict[str, Any]], role: str, parts: list[dict[str, Any]]
) -> None:
    if not parts:
        return
    if contents and contents[-1]["role"] == role:
        contents[-1]["parts"].extend(parts)
    else:
        contents.append({"role": role, "parts": parts})


def _reasoning(item: dict[str, Any]) -> list[dict[str, Any]]:
    carrier = item.get("encrypted_content") or item.get("signature") or item.get("data")
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
    body: dict[str, Any], policy: ReasoningPolicy, *, native: bool = False
) -> None:
    config = body.setdefault("generationConfig", {})
    if native and policy == ReasoningPolicy.provider_default():
        return
    thinking = config.setdefault("thinkingConfig", {})
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


def _tool_config(choice: Any, *, responses: bool = False) -> dict[str, Any]:
    if isinstance(choice, str):
        mode = {"auto": "AUTO", "none": "NONE", "required": "ANY", "any": "ANY"}.get(
            choice
        )
        if mode is None:
            raise GeminiConversionError("Unsupported tool_choice")
        return {"functionCallingConfig": {"mode": mode}}
    if isinstance(choice, dict):
        name = (
            choice.get("name") if responses else choice.get("function", {}).get("name")
        )
        if isinstance(name, str):
            return {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}
            }
    raise GeminiConversionError("Unsupported tool_choice")


def _format(config: dict[str, Any], format_: dict[str, Any]) -> None:
    kind = format_.get("type")
    if kind == "text":
        return
    if kind == "json_object":
        config["responseMimeType"] = "application/json"
    elif kind == "json_schema":
        schema = format_.get("json_schema", format_)
        config.update(
            responseMimeType="application/json",
            responseJsonSchema=deepcopy(schema["schema"]),
        )
    else:
        raise GeminiConversionError("Unsupported structured output format")


def _function_result(
    call_id: str, name: str, value: Any, *, responses: bool = False
) -> dict[str, Any]:
    parts = _text_parts(value, responses=responses)
    text = "\n".join(part["text"] for part in parts if "text" in part)
    result: dict[str, Any] = {"id": call_id, "name": name, "response": {"output": text}}
    images = [part for part in parts if "inlineData" in part]
    if images:
        result["parts"] = images
    return {"functionResponse": result}


def chat_to_gemini(
    request: ChatRequest, *, allow_pending_calls: bool = False
) -> JsonObject:
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
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "reasoning_effort",
            "response_format",
        },
    )
    contents: list[dict[str, Any]] = []
    system: list[dict[str, Any]] = []
    calls: dict[str, str] = {}
    for message in body["messages"]:
        validate_message_fields(message)
        role = message["role"]
        if role in {"system", "developer"}:
            parts = _text_parts(message.get("content"))
            if any("text" not in part for part in parts):
                raise GeminiConversionError("System instructions must contain text")
            system.extend(parts)
            continue
        if role == "tool":
            call_id = message["tool_call_id"]
            if call_id not in calls:
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
                key: value for key, value in message.items() if key != "content"
            }
            parts.extend(_reasoning(reasoning))
            for detail in message.get("reasoning_details") or []:
                if detail.get("type") == "reasoning.encrypted":
                    parts.extend(_reasoning(detail))
                elif detail.get("type") not in {
                    "reasoning.text",
                    "reasoning.summary",
                } or detail.get("signature"):
                    raise GeminiConversionError(
                        "Unsupported protected reasoning detail"
                    )
        parts.extend(_text_parts(message.get("content")))
        if message.get("refusal"):
            parts.append({"text": message["refusal"]})
        for call in function_calls(message):
            name, call_id = call["function"]["name"], call["id"]
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
    config: dict[str, Any] = {}
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
    result: dict[str, Any] = {
        "model": request.model,
        "contents": contents,
        "generationConfig": config,
    }
    if system:
        result["systemInstruction"] = {"parts": system}
    declarations = []
    for tool in body.get("tools") or []:
        if tool.get("type") != "function":
            raise GeminiConversionError("Native Gemini supports function tools only")
        function = tool["function"]
        if function.get("strict"):
            raise GeminiConversionError(
                "Native Gemini cannot enforce strict function tools"
            )
        if set(function) - {"name", "description", "parameters", "strict"}:
            raise GeminiConversionError("Unsupported function declaration fields")
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
    if (choice := body.get("tool_choice")) is not None:
        result["toolConfig"] = _tool_config(choice)
    if body.get("parallel_tool_calls") is False:
        raise GeminiConversionError(
            "Native Gemini cannot enforce parallel_tool_calls=false"
        )
    return result


def messages_to_gemini(
    request: MessagesRequest, *, allow_pending_calls: bool = False
) -> JsonObject:
    body = dump_messages_request(request)
    unsupported = set(request.model_extra or {})
    unsupported.update(key for key in ("mcp_servers", "extra_body") if body.get(key))
    if not is_noop_context_management(request.context_management):
        unsupported.add("context_management")
    if unsupported:
        raise GeminiConversionError(
            f"Native Gemini cannot represent Messages fields: {sorted(unsupported)}"
        )
    contents: list[dict[str, Any]] = []
    system: list[dict[str, Any]] = []
    if body.get("system"):
        system = [{"text": part["text"]} for part in content_parts(body["system"])]
    calls: dict[str, str] = {}
    for message in body["messages"]:
        role = message["role"]
        parts: list[dict[str, Any]] = []
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
            elif kind in {"thinking", "redacted_thinking"}:
                parts.extend(_reasoning(block))
            elif kind == "tool_use":
                if block.get("extra_content"):
                    raise GeminiConversionError(
                        "Saved compatibility tool metadata cannot be replayed on native Gemini; start a fresh conversation"
                    )
                call_id, name = str(block["id"]), str(block["name"])
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
                value = block.get("content")
                output = [] if isinstance(value, dict) else content_parts(value)
                text = (
                    deepcopy(value)
                    if isinstance(value, dict)
                    else "\n".join(
                        part["text"] for part in output if part.get("type") == "text"
                    )
                )
                result: dict[str, Any] = {
                    "id": call_id,
                    "name": calls.pop(call_id),
                    "response": {"error" if block.get("is_error") else "output": text},
                }
                images = [
                    _image(portable_anthropic_image_url(part["source"]))
                    for part in output
                    if part.get("type") == "image"
                ]
                if any(part.get("type") not in {"image", "text"} for part in output):
                    raise GeminiConversionError("Unsupported tool result content")
                if images:
                    result["parts"] = images
                parts.append({"functionResponse": result})
            else:
                raise GeminiConversionError(f"Unsupported Messages content {kind!r}")
        if role == "system":
            system.extend(parts)
        else:
            _append(contents, "model" if role == "assistant" else "user", parts)
    if calls and not allow_pending_calls:
        raise GeminiConversionError("Unresolved tool calls in history")
    config: dict[str, Any] = {}
    for source, target in {
        "max_tokens": "maxOutputTokens",
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "stop_sequences": "stopSequences",
    }.items():
        if body.get(source) is not None:
            config[target] = body[source]
    output = body.get("output_config") or {}
    if set(output) - {"effort", "format"}:
        raise GeminiConversionError("Unsupported Messages output_config")
    if fmt := output.get("format"):
        _format(config, fmt)
    result: dict[str, Any] = {
        "model": request.model,
        "contents": contents,
        "generationConfig": config,
    }
    if system:
        result["systemInstruction"] = {"parts": system}
    if body.get("tools"):
        declarations = []
        for tool in body["tools"]:
            if tool.get("strict"):
                raise GeminiConversionError(
                    "Native Gemini cannot enforce strict function tools"
                )
            if tool.get("type") not in (None, "custom"):
                raise GeminiConversionError(
                    "Native Gemini cannot execute provider-hosted tools"
                )
            declarations.append(
                {
                    "name": tool["name"],
                    "parametersJsonSchema": tool.get(
                        "input_schema", {"type": "object"}
                    ),
                    **(
                        {"description": tool["description"]}
                        if tool.get("description")
                        else {}
                    ),
                }
            )
        result["tools"] = [{"functionDeclarations": declarations}]
    if body.get("tools") and (choice := body.get("tool_choice")):
        if choice.get("disable_parallel_tool_use"):
            raise GeminiConversionError(
                "Native Gemini cannot enforce disable_parallel_tool_use"
            )
        result["toolConfig"] = _tool_config(
            choice if choice.get("type") == "tool" else choice["type"], responses=True
        )
    return result


def responses_to_gemini(
    request: OpenAIResponsesRequest, *, allow_pending_calls: bool = False
) -> JsonObject:
    body = request.model_dump(mode="json", exclude_none=True)
    supported = {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "temperature",
        "top_p",
        "max_output_tokens",
        "reasoning",
        "store",
        "include",
        "text",
        "metadata",
    }
    if unsupported := set(body) - supported:
        raise GeminiConversionError(
            f"Native Gemini cannot represent Responses fields: {sorted(unsupported)}"
        )
    if body.get("store") or body.get("parallel_tool_calls") is False:
        raise GeminiConversionError(
            "Native Gemini cannot enforce storage or serial tool controls"
        )
    value = body.get("input")
    items = (
        [{"role": "user", "content": value}]
        if isinstance(value, str)
        else value
        if isinstance(value, list)
        else [value]
    )
    contents: list[dict[str, Any]] = []
    system = [{"text": body["instructions"]}] if body.get("instructions") else []
    calls: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise GeminiConversionError("Responses input requires messages or items")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            parts = _text_parts(item.get("content"), responses=True)
            if role in {"system", "developer"}:
                if any("text" not in part for part in parts):
                    raise GeminiConversionError("System instructions must be text")
                system.extend(parts)
            elif role in {"assistant", "user"}:
                _append(contents, "model" if role == "assistant" else "user", parts)
            else:
                raise GeminiConversionError("Unsupported Responses role")
        elif kind == "reasoning":
            _append(contents, "model", _reasoning(item))
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
            call_id, name = item["call_id"], item["name"]
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
        else:
            raise GeminiConversionError(f"Unsupported Responses input item {kind!r}")
    if calls and not allow_pending_calls:
        raise GeminiConversionError("Unresolved function calls in history")
    config: dict[str, Any] = {}
    for source, target in {
        "temperature": "temperature",
        "top_p": "topP",
        "max_output_tokens": "maxOutputTokens",
    }.items():
        if body.get(source) is not None:
            config[target] = body[source]
    text = body.get("text") or {}
    if set(text) - {"format"}:
        raise GeminiConversionError(
            "Native Gemini cannot represent Responses text controls"
        )
    if fmt := text.get("format"):
        _format(config, fmt)
    result: dict[str, Any] = {
        "model": request.model,
        "contents": contents,
        "generationConfig": config,
    }
    if system:
        result["systemInstruction"] = {"parts": system}
    declarations = []
    for tool in body.get("tools") or []:
        if tool.get("type") != "function":
            raise GeminiConversionError("Native Gemini supports function tools only")
        if tool.get("strict"):
            raise GeminiConversionError(
                "Native Gemini cannot enforce strict function tools"
            )
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise GeminiConversionError("Function declarations require a name")
        if set(tool) - {"type", "name", "description", "parameters", "strict"}:
            raise GeminiConversionError("Unsupported function declaration fields")
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
    if (choice := body.get("tool_choice")) is not None:
        result["toolConfig"] = _tool_config(choice, responses=True)
    return result


def native_gemini_body(request: GeminiRequest) -> JsonObject:
    return request.model_dump(mode="json", by_alias=True, exclude_unset=True)
