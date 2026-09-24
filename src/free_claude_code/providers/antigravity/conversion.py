"""Translate FCC's OpenAI-chat wire shape to and from Antigravity Gemini events."""

import base64
import json
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


def openai_chat_to_cloudcode(
    body: dict[str, Any], *, project_id: str
) -> dict[str, Any]:
    """Build one Cloud Code generateContent envelope from an OpenAI chat body."""

    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = []
    contents, system_instruction = _messages_to_gemini(messages)
    request: dict[str, Any] = {"contents": contents}
    if system_instruction is not None:
        request["systemInstruction"] = system_instruction

    tools = _tools_to_gemini(body.get("tools"))
    if tools:
        request["tools"] = tools
        tool_config = _tool_config(body.get("tool_choice"))
        if tool_config is not None:
            request["toolConfig"] = tool_config

    generation = _generation_config(body)
    if generation:
        request["generationConfig"] = generation

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Antigravity request is missing model")
    return {"model": model, "project": project_id, "request": request}


def _messages_to_gemini(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    names_by_id: dict[str, str] = {}
    for raw in messages:
        if not isinstance(raw, dict) or raw.get("role") != "assistant":
            continue
        for tool_call in raw.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            call_id = tool_call.get("id")
            if (
                isinstance(function, dict)
                and isinstance(function.get("name"), str)
                and isinstance(call_id, str)
            ):
                names_by_id[call_id.split("|", 1)[0]] = function["name"]

    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []
    pending_tool_parts: list[dict[str, Any]] = []

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user").lower()
        if role != "tool" and pending_tool_parts:
            contents.append({"role": "user", "parts": pending_tool_parts})
            pending_tool_parts = []

        if role in {"system", "developer"}:
            system_parts.extend(_content_parts(raw.get("content")))
            continue

        if role == "tool":
            call_id = str(raw.get("tool_call_id") or "")
            clean_id = call_id.split("|", 1)[0]
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                name = names_by_id.get(clean_id)
            if not name:
                raise ValueError("tool response could not resolve its function name")
            output = _tool_output_text(raw.get("content"))
            pending_tool_parts.append(
                {
                    "functionResponse": {
                        "id": clean_id,
                        "name": name,
                        "response": {"output": output},
                    }
                }
            )
            continue

        parts: list[dict[str, Any]] = []
        if role == "assistant":
            reasoning = raw.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = raw.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                parts.append({"thought": True, "text": reasoning})
        parts.extend(_content_parts(raw.get("content")))
        if role == "assistant":
            for tool_call in raw.get("tool_calls") or []:
                part = _assistant_tool_part(tool_call)
                if part is not None:
                    parts.append(part)
        if parts:
            contents.append(
                {"role": "model" if role == "assistant" else "user", "parts": parts}
            )

    if pending_tool_parts:
        contents.append({"role": "user", "parts": pending_tool_parts})

    system = {"role": "system", "parts": system_parts} if system_parts else None
    return contents, system


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for raw in content:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type")
        if kind == "text" and isinstance(raw.get("text"), str):
            parts.append({"text": raw["text"]})
        elif kind == "image_url":
            inline = _image_part(raw.get("image_url"))
            if inline is not None:
                parts.append({"inlineData": inline})
    return parts


def _image_part(value: Any) -> dict[str, str] | None:
    url = (
        value
        if isinstance(value, str)
        else value.get("url")
        if isinstance(value, dict)
        else None
    )
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    metadata, separator, data = url[5:].partition(",")
    if not separator:
        return None
    mime = metadata.split(";", 1)[0] or "image/jpeg"
    try:
        base64.b64decode(data, validate=True)
    except ValueError:
        return None
    return {"mimeType": mime, "data": data}


def _assistant_tool_part(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments")
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    call_id = str(value.get("id") or f"toolu_{uuid.uuid4()}")
    clean_id, separator, signature = call_id.partition("|")
    part: dict[str, Any] = {
        "functionCall": {"id": clean_id, "name": name, "args": args}
    }
    if separator and signature:
        part["thoughtSignature"] = signature
    return part


def _tool_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _tools_to_gemini(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    declarations: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or raw.get("type") != "function":
            continue
        function = raw.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        declaration: dict[str, Any] = {"name": function["name"]}
        if isinstance(function.get("description"), str):
            declaration["description"] = function["description"]
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            declaration["parameters"] = _clean_schema(parameters)
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def _tool_config(value: Any) -> dict[str, Any] | None:
    mode: str | None = None
    allowed_names: list[str] | None = None
    if value == "auto":
        mode = "AUTO"
    elif value == "none":
        mode = "NONE"
    elif value == "required":
        mode = "ANY"
    elif isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            mode = "ANY"
            allowed_names = [name]
    if mode is None:
        return None
    calling: dict[str, Any] = {"mode": mode}
    if allowed_names is not None:
        calling["allowedFunctionNames"] = allowed_names
    return {"functionCallingConfig": calling}


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    allowed = {"type", "description", "properties", "items", "required", "enum"}
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                str(name): _clean_schema(item)
                for name, item in value.items()
                if isinstance(item, dict)
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _generation_config(body: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    _copy_number(body, "temperature", config, "temperature")
    _copy_number(body, "top_p", config, "topP")
    _copy_number(body, "presence_penalty", config, "presencePenalty")
    _copy_number(body, "frequency_penalty", config, "frequencyPenalty")

    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if (
        isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens > 0
    ):
        config["maxOutputTokens"] = max_tokens

    stop = body.get("stop")
    if isinstance(stop, str) and stop:
        config["stopSequences"] = [stop]
    elif isinstance(stop, list):
        stops = [item for item in stop if isinstance(item, str) and item]
        if stops:
            config["stopSequences"] = stops

    seed = body.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        config["seed"] = seed

    effort = body.get("reasoning_effort")
    if isinstance(effort, str):
        normalized = effort.strip().lower()
        if normalized in {"none", "off"}:
            config["thinkingConfig"] = {"thinkingBudget": 0}
        elif normalized in {"low", "medium", "high"}:
            config["thinkingConfig"] = {"thinkingLevel": normalized.upper()}
    return config


def _copy_number(
    source: dict[str, Any], source_key: str, target: dict[str, Any], target_key: str
) -> None:
    value = source.get(source_key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        target[target_key] = value


@dataclass(slots=True)
class StreamState:
    model: str
    chat_id: str = ""
    created: int = 0
    sent_role: bool = False
    saw_tool: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    next_tool_index: int = 0
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.chat_id:
            self.chat_id = f"chatcmpl-{uuid.uuid4()}"
        if not self.created:
            self.created = int(time.time())


def gemini_event_chunks(event: dict[str, Any], state: StreamState) -> list[Any]:
    """Convert one Cloud Code SSE payload to OpenAI-SDK-like chunk objects."""

    response = event.get("response")
    payload: dict[str, Any] = response if isinstance(response, dict) else event
    chunks: list[Any] = []
    usage = payload.get("usageMetadata")
    if isinstance(usage, dict):
        state.prompt_tokens = _int(usage.get("promptTokenCount"))
        state.completion_tokens = _int(
            usage.get("candidatesTokenCount", usage.get("completionTokenCount"))
        )

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return chunks
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        finish_reason = candidate.get("finishReason") or candidate.get("stopReason")
        if isinstance(finish_reason, str) and finish_reason:
            state.finish_reason = finish_reason
        content = candidate.get("content")
        parts = (
            content.get("parts")
            if isinstance(content, dict)
            else candidate.get("parts")
        )
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                reasoning = text if part.get("thought") is True else None
                chunks.append(
                    _chunk(
                        state,
                        content=None if reasoning else text,
                        reasoning=reasoning,
                    )
                )
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                args = function_call.get("args")
                if not isinstance(args, dict):
                    args = {}
                call_id = function_call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"call_{uuid.uuid4()}"
                signature = part.get("thoughtSignature") or part.get(
                    "thought_signature"
                )
                if isinstance(signature, str) and signature:
                    call_id = f"{call_id}|{signature}"
                tool_index = state.next_tool_index
                state.next_tool_index += 1
                state.saw_tool = True
                chunks.append(
                    _chunk(
                        state,
                        tool_call=(call_id, name, args),
                        tool_index=tool_index,
                    )
                )
    return chunks


def final_chunk(state: StreamState) -> Any:
    usage = SimpleNamespace(
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        total_tokens=state.prompt_tokens + state.completion_tokens,
        completion_tokens_details=None,
    )
    choice = SimpleNamespace(
        finish_reason=_openai_finish_reason(state),
        delta=SimpleNamespace(
            role=None,
            content=None,
            reasoning_content=None,
            reasoning=None,
            tool_calls=None,
        ),
    )
    return SimpleNamespace(
        id=state.chat_id,
        object="chat.completion.chunk",
        created=state.created,
        model=state.model,
        choices=[choice],
        usage=usage,
    )


def _openai_finish_reason(state: StreamState) -> str:
    if state.saw_tool:
        return "tool_calls"
    reason = (state.finish_reason or "").upper()
    if reason == "MAX_TOKENS":
        return "length"
    if reason in {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "LANGUAGE",
    }:
        return "content_filter"
    return "stop"


def _chunk(
    state: StreamState,
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_call: tuple[str, str, dict[str, Any]] | None = None,
    tool_index: int = 0,
) -> Any:
    role = None if state.sent_role else "assistant"
    state.sent_role = True
    tools = None
    if tool_call is not None:
        call_id, name, args = tool_call
        tools = [
            SimpleNamespace(
                index=tool_index,
                id=call_id,
                type="function",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(
                        args, separators=(",", ":"), ensure_ascii=False
                    ),
                ),
            )
        ]
    delta = SimpleNamespace(
        role=role,
        content=content,
        reasoning_content=reasoning,
        reasoning=reasoning,
        tool_calls=tools,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
    return SimpleNamespace(
        id=state.chat_id,
        object="chat.completion.chunk",
        created=state.created,
        model=state.model,
        choices=[choice],
        usage=None,
    )


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
