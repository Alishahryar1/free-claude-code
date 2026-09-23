"""OpenAI-chat tool-call assembly helpers."""

import json
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from free_claude_code.core.anthropic.streaming import (
    ToolSchema,
    parse_complete_tool_input,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec

from .stream_output import ChatStreamOutput

RecordToolExtraContent = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class _CollectedToolCall:
    index: int
    tool_id: str | None = None
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)
    extra_content: dict[str, Any] | None = None


class _CompletedOpenAIToolFunction(TypedDict):
    name: str
    arguments: str


class CompletedOpenAIToolCall(TypedDict):
    """Schema-valid OpenAI tool-call payload collected for recovery emission."""

    index: int
    id: str | None
    function: _CompletedOpenAIToolFunction
    extra_content: NotRequired[JsonObject]


class OpenAIToolCallCollector:
    """Collect and validate one buffered OpenAI tool-call response."""

    def __init__(self) -> None:
        self._calls: dict[int, _CollectedToolCall] = {}

    @property
    def has_calls(self) -> bool:
        return bool(self._calls)

    def add(self, tool_call: Any) -> None:
        """Add one SDK tool-call delta without emitting downstream state."""
        raw_index = getattr(tool_call, "index", 0)
        index = raw_index if isinstance(raw_index, int) and raw_index >= 0 else 0
        state = self._calls.setdefault(index, _CollectedToolCall(index=index))

        tool_id = getattr(tool_call, "id", None)
        if tool_id:
            state.tool_id = str(tool_id)

        function = getattr(tool_call, "function", None)
        incoming_name = getattr(function, "name", None)
        if isinstance(incoming_name, str) and incoming_name:
            state.name = _merge_tool_name(state.name, incoming_name)

        arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str) and arguments:
            state.argument_parts.append(arguments)

        extra_content = tool_call_extra_content(tool_call)
        if extra_content:
            state.extra_content = extra_content

    def completed_calls(
        self,
        schemas: dict[str, ToolSchema],
        *,
        tool_names: OpenAIToolNameCodec | None = None,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
    ) -> tuple[CompletedOpenAIToolCall, ...] | None:
        """Return complete schema-valid calls, or None when output is incomplete."""
        completed: list[CompletedOpenAIToolCall] = []
        for index in sorted(self._calls):
            state = self._calls[index]
            wire_name = state.name.strip()
            name = tool_names.decode(wire_name) if tool_names is not None else wire_name
            if not name or name not in schemas:
                return None
            arguments = "".join(state.argument_parts)
            aliases = (
                tool_argument_aliases.get(name, {})
                if tool_argument_aliases is not None
                else {}
            )
            if aliases:
                restored = restore_tool_argument_aliases(arguments, aliases)
                if restored is None:
                    return None
                arguments = restored
            if parse_complete_tool_input(arguments, name, schemas) is None:
                return None

            call: CompletedOpenAIToolCall = {
                "index": index,
                "id": state.tool_id,
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
            if state.extra_content:
                call["extra_content"] = state.extra_content
            completed.append(call)
        return tuple(completed)


def iter_heuristic_tool_use_events(
    output: ChatStreamOutput,
    tool_use: dict[str, Any],
    *,
    tool_names: OpenAIToolNameCodec | None = None,
) -> Iterator[str]:
    """Emit SSE for one heuristic tool_use block."""
    name = tool_use.get("name")
    if tool_names is not None and isinstance(name, str):
        decoded_name = tool_names.decode(name)
        if decoded_name != name:
            tool_use = {**tool_use, "name": decoded_name}
    if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
        task_input = tool_use["input"]
        if task_input.get("run_in_background") is not False:
            task_input["run_in_background"] = False
    yield from output.close_content_blocks()
    tool_index = len(output.tool_states)
    output.ensure_tool_state(tool_index)
    output.register_tool_name(tool_index, str(tool_use["name"]))
    yield output.start_tool_block(
        tool_index,
        str(tool_use["id"]),
        str(tool_use["name"]),
    )
    yield output.emit_tool_delta(tool_index, json.dumps(tool_use["input"]))
    yield from output.stop_tool_block(tool_index)


def tool_call_extra_content(tool_call: Any) -> dict[str, Any] | None:
    """Return provider-specific extra tool-call metadata from OpenAI objects."""
    if isinstance(tool_call, dict):
        value = tool_call.get("extra_content")
        return value if isinstance(value, dict) else None

    value = getattr(tool_call, "extra_content", None)
    if isinstance(value, dict):
        return value

    model_extra = getattr(tool_call, "model_extra", None)
    if isinstance(model_extra, dict):
        value = model_extra.get("extra_content")
        if isinstance(value, dict):
            return value

    pydantic_extra = getattr(tool_call, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict):
        value = pydantic_extra.get("extra_content")
        if isinstance(value, dict):
            return value

    return None


class OpenAIToolCallAssembler:
    """Assemble OpenAI tool-call deltas into Anthropic SSE tool blocks."""

    def __init__(
        self,
        *,
        reserved_tool_ids: Iterable[str],
        record_extra_content: RecordToolExtraContent | None = None,
    ) -> None:
        self._record_extra_content = record_extra_content
        self._reserved_tool_ids = {tool_id for tool_id in reserved_tool_ids if tool_id}
        self._candidate_tool_ids: dict[int, str] = {}
        self._public_tool_ids: dict[int, str] = {}

    def process_tool_call(
        self,
        tc: Mapping[str, Any],
        output: ChatStreamOutput,
        *,
        tool_names: OpenAIToolNameCodec | None = None,
        tool_name_buffers: dict[int, str] | None = None,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[str]:
        """Process one tool-call delta and yield client-protocol events."""
        raw_index = tc.get("index", 0)
        tc_index = raw_index if isinstance(raw_index, int) else 0
        if tc_index < 0:
            tc_index = len(output.tool_states)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        arguments = fn_delta.get("arguments", "") or ""

        candidate_id = tc.get("id")
        if candidate_id is not None:
            output.ensure_tool_state(tc_index)
        if (
            tc_index not in self._public_tool_ids
            and isinstance(candidate_id, str)
            and candidate_id.strip()
        ):
            self._candidate_tool_ids[tc_index] = candidate_id

        raw_extra_content = tc.get("extra_content")
        extra_content = (
            raw_extra_content
            if isinstance(raw_extra_content, dict) and raw_extra_content
            else None
        )
        if extra_content:
            output.set_tool_extra_content(tc_index, extra_content)

        if isinstance(incoming_name, str) and incoming_name:
            resolved_name = _decode_streamed_tool_name(
                incoming_name,
                tool_index=tc_index,
                tool_names=tool_names,
                buffers=tool_name_buffers,
            )
            if resolved_name is not None:
                output.register_tool_name(tc_index, resolved_name)

        state = output.tool_states.get(tc_index)
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = self._assign_public_tool_id(tc_index)
                display_name = (resolved_name or "").strip() or "tool_call"
                start_extra_content = state.extra_content if state else extra_content
                if start_extra_content:
                    self._record_tool_call_extra_content(tool_id, start_extra_content)
                yield output.start_tool_block(
                    tc_index,
                    tool_id,
                    display_name,
                    extra_content=start_extra_content,
                )
                state = output.tool_states[tc_index]
                if state.pre_start_args:
                    pre = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(
                        output,
                        tc_index,
                        pre,
                        tool_argument_aliases=tool_argument_aliases,
                        tool_argument_alias_buffers=tool_argument_alias_buffers,
                    )

        state = output.tool_states.get(tc_index)
        if state is not None and state.tool_id and extra_content:
            self._record_tool_call_extra_content(state.tool_id, extra_content)
        if not arguments:
            return
        if state is None or not state.started:
            state = output.ensure_tool_state(tc_index)
            if not (resolved_name or "").strip():
                state.pre_start_args += arguments
                return

        yield from self._emit_tool_arg_delta(
            output,
            tc_index,
            arguments,
            tool_argument_aliases=tool_argument_aliases,
            tool_argument_alias_buffers=tool_argument_alias_buffers,
        )

    def _assign_public_tool_id(self, tool_index: int) -> str:
        assigned = self._public_tool_ids.get(tool_index)
        if assigned is not None:
            return assigned

        candidate = self._candidate_tool_ids.get(tool_index)
        if candidate is not None and candidate not in self._reserved_tool_ids:
            public_id = candidate
        else:
            public_id = f"tool_{uuid.uuid4()}"
            while public_id in self._reserved_tool_ids:
                public_id = f"tool_{uuid.uuid4()}"

        self._reserved_tool_ids.add(public_id)
        self._public_tool_ids[tool_index] = public_id
        return public_id

    def flush_task_arg_buffers(self, output: ChatStreamOutput) -> Iterator[str]:
        """Emit buffered Task args as a single JSON delta."""
        for tool_index, out in output.flush_task_arg_buffers():
            yield output.emit_tool_delta(tool_index, out)

    def flush_tool_name_buffers(
        self,
        output: ChatStreamOutput,
        *,
        tool_names: OpenAIToolNameCodec,
        tool_name_buffers: dict[int, str],
        tool_argument_aliases: dict[str, dict[str, str]],
        tool_argument_alias_buffers: dict[int, str],
    ) -> Iterator[str]:
        """Resolve names held only because they also prefix a generated alias."""
        for tool_index, name in list(tool_name_buffers.items()):
            tool_name_buffers.pop(tool_index, None)
            yield from self.process_tool_call(
                {
                    "index": tool_index,
                    "function": {"name": name, "arguments": ""},
                },
                output,
                tool_names=tool_names,
                tool_argument_aliases=tool_argument_aliases,
                tool_argument_alias_buffers=tool_argument_alias_buffers,
            )

    def flush_tool_argument_alias_buffers(
        self,
        output: ChatStreamOutput,
        tool_argument_aliases: dict[str, dict[str, str]],
        tool_argument_alias_buffers: dict[int, str],
    ) -> Iterator[str]:
        """Emit remaining aliased args without losing malformed JSON."""
        for tool_index, buffered_args in list(tool_argument_alias_buffers.items()):
            if not buffered_args:
                tool_argument_alias_buffers.pop(tool_index, None)
                continue
            state = output.tool_states.get(tool_index)
            if state is None or state.name == "Task":
                continue
            aliases = tool_argument_aliases.get(state.name, {})
            if not aliases:
                continue
            restored = self._restore_aliased_tool_arguments(buffered_args, aliases)
            yield output.emit_tool_delta(
                tool_index,
                restored if restored is not None else buffered_args,
            )
            tool_argument_alias_buffers.pop(tool_index, None)

    def _emit_tool_arg_delta(
        self,
        output: ChatStreamOutput,
        tc_index: int,
        args: str,
        *,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[str]:
        """Emit one argument fragment for a started tool block."""
        if not args:
            return
        state = output.tool_states.get(tc_index)
        if state is None:
            return
        if state.name == "Task":
            parsed = output.buffer_task_args(tc_index, args)
            if parsed is not None:
                yield output.emit_tool_delta(tc_index, json.dumps(parsed))
            return
        aliases = (
            tool_argument_aliases.get(state.name, {}) if tool_argument_aliases else {}
        )
        if aliases:
            if tool_argument_alias_buffers is None:
                restored = self._restore_aliased_tool_arguments(args, aliases)
                if restored is not None:
                    yield output.emit_tool_delta(tc_index, restored)
                return

            buffered_args = tool_argument_alias_buffers.get(tc_index, "") + args
            restored = self._restore_aliased_tool_arguments(buffered_args, aliases)
            if restored is None:
                tool_argument_alias_buffers[tc_index] = buffered_args
                return
            tool_argument_alias_buffers.pop(tc_index, None)
            yield output.emit_tool_delta(tc_index, restored)
            return
        yield output.emit_tool_delta(tc_index, args)

    def _restore_aliased_tool_arguments(
        self, argument_json: str, aliases: dict[str, str]
    ) -> str | None:
        return restore_tool_argument_aliases(argument_json, aliases)

    def _record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        if self._record_extra_content is not None:
            self._record_extra_content(tool_call_id, extra_content)


def restore_tool_argument_aliases(
    argument_json: str,
    aliases: dict[str, str],
) -> str | None:
    """Restore provider-private argument aliases in one complete JSON object."""
    try:
        parsed = json.loads(argument_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return argument_json
    return json.dumps(_restore_tool_argument_alias_value(parsed, aliases))


def _restore_tool_argument_alias_value(
    value: Any,
    aliases: dict[str, str],
) -> Any:
    if isinstance(value, dict):
        return {
            aliases.get(key, key): _restore_tool_argument_alias_value(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_tool_argument_alias_value(item, aliases) for item in value]
    return value


def _merge_tool_name(existing: str, incoming: str) -> str:
    if not existing or incoming.startswith(existing):
        return incoming
    if existing.startswith(incoming):
        return existing
    return "".join((existing, incoming))


def _decode_streamed_tool_name(
    incoming: str,
    *,
    tool_index: int,
    tool_names: OpenAIToolNameCodec | None,
    buffers: dict[int, str] | None,
) -> str | None:
    if tool_names is None or not tool_names.has_aliases:
        return incoming
    if buffers is None:
        return tool_names.decode(incoming)

    combined = _merge_tool_name(buffers.get(tool_index, ""), incoming)
    if tool_names.is_alias(combined):
        buffers.pop(tool_index, None)
        return tool_names.decode(combined)
    if tool_names.is_alias_prefix(combined):
        buffers[tool_index] = combined
        return None
    buffers.pop(tool_index, None)
    return tool_names.decode(combined)
