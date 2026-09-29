"""Project public Claude SDK messages onto FCC transcript identities."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import cast

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolResultBlock,
    UserMessage,
)
from claude_agent_sdk.types import Message, SessionMessage

from free_claude_code.application.code_sessions.models import (
    CodeValidationError,
    HarnessEvent,
    ItemUpdate,
    NativeThread,
    NativeTurn,
    PromptRequest,
)
from free_claude_code.core.json_types import JsonObject, JsonValue


def obj(value: object) -> JsonObject:
    return cast(JsonObject, value) if isinstance(value, dict) else {}


def pretty(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


@dataclass
class ClaudePrompt:
    id: str
    turn_id: str
    tool: str
    input: JsonObject

    def request(self) -> PromptRequest:
        if self.tool != "AskUserQuestion":
            return PromptRequest(
                self.id,
                "approval",
                {
                    "title": f"Allow {self.tool}?",
                    "detail": pretty(self.input),
                    "choices": [
                        {"id": "allow", "label": "Allow once"},
                        {"id": "deny", "label": "Deny"},
                    ],
                },
                self.input,
                self.turn_id,
                self.id,
            )
        questions = self.input.get("questions")
        if not isinstance(questions, list) or not questions:
            raise CodeValidationError("Claude returned an invalid question.")
        forms: list[JsonValue] = []
        for index, raw in enumerate(questions):
            question = obj(raw)
            text = question.get("question")
            if not isinstance(text, str) or not text:
                raise CodeValidationError("Claude returned a question without text.")
            forms.append(
                {
                    "id": str(index),
                    "label": text,
                    "header": question.get("header"),
                    "options": question.get("options", []),
                    "multiple": question.get("multiSelect") is True,
                    "allow_other": True,
                }
            )
        return PromptRequest(
            self.id,
            "questions",
            {"title": "Claude Code needs your input", "questions": forms},
            self.input,
            self.turn_id,
            self.id,
        )

    def answer(self, answer: JsonObject) -> JsonObject:
        if self.tool != "AskUserQuestion":
            if answer.get("choice") not in {"allow", "deny"}:
                raise CodeValidationError("Choose Allow once or Deny.")
            return {"allow": answer["choice"] == "allow", "updated_input": self.input}
        supplied = obj(answer.get("answers"))
        questions = self.request().form["questions"]
        assert isinstance(questions, list)
        if supplied.keys() != {str(i) for i in range(len(questions))}:
            raise CodeValidationError("Answer each question.")
        result: JsonObject = {}
        for index, raw in enumerate(questions):
            question = obj(raw)
            values = supplied[str(index)]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
            ):
                raise CodeValidationError("Enter a nonempty answer for each question.")
            if not question.get("multiple") and len(values) != 1:
                raise CodeValidationError("Choose one answer for this question.")
            result[str(question["label"])] = ", ".join(cast(list[str], values))
        return {"allow": True, "updated_input": {**self.input, "answers": result}}


class ClaudeProtocol:
    def __init__(self, generation: str, session_id: str) -> None:
        self.generation, self.session_id = generation, session_id
        self.pending: str | None = None
        self.context: str | None = None
        self.interrupted = False
        self.unattributed_result = False
        self._streams: dict[str | None, str] = {}
        self._messages: dict[str, str] = {}
        self._tools: dict[str, str] = {}
        self._tasks: dict[str, tuple[str, str]] = {}
        self._items: dict[str, ItemUpdate] = {}
        self._block_ids: dict[tuple[str, int], str] = {}
        self._latest_block: dict[str, int] = {}
        self._final_offsets: dict[str, int] = {}
        self._completed_messages: dict[str, int] = {}

    def begin(self, run_id: str) -> None:
        self.pending = run_id
        self.context = None
        self.interrupted = False
        self.unattributed_result = False

    def turn_for_tool(self, tool_id: str) -> str | None:
        return self._tools.get(tool_id) or self.context

    def event(self, kind, **kwargs) -> HarnessEvent:
        return HarnessEvent(self.generation, self.session_id, kind, **kwargs)

    def _item(
        self, turn: str, message: str, index: int, block: JsonObject, *, complete: bool
    ) -> ItemUpdate | None:
        kind = block.get("type")
        item_id = str(block.get("id")) if kind == "tool_use" else f"{message}:{index}"
        if kind == "text":
            item = ItemUpdate(
                turn,
                item_id,
                "text",
                text=str(block.get("text", "")),
                complete=complete,
            )
        elif kind == "thinking":
            item = ItemUpdate(
                turn,
                item_id,
                "reasoning",
                text=str(block.get("thinking", "")),
                complete=complete,
            )
        elif kind == "tool_use":
            self._tools[item_id] = turn
            item = ItemUpdate(
                turn,
                item_id,
                "tool",
                title=str(block.get("name", "Tool")),
                detail=pretty(block.get("input", {})),
                complete=False,
            )
        else:
            return None
        self._items[item_id] = item
        self._block_ids[message, index] = item_id
        return item

    def _tool_result(self, block: JsonObject) -> HarnessEvent | None:
        identity = str(block.get("tool_use_id", ""))
        previous = self._items.get(identity)
        turn = self._tools.get(identity)
        if turn is None:
            return None
        content = block.get("content", "")
        text = content if isinstance(content, str) else pretty(content)
        item = (
            replace(previous, text=text, complete=True)
            if previous
            else ItemUpdate(turn, identity, "tool", text=text, complete=True)
        )
        self._items[identity] = item
        return self.event("item", turn_id=turn, item=item)

    def feed(self, message: Message) -> tuple[HarnessEvent, ...]:
        if isinstance(message, UserMessage):
            if (
                message.uuid == self.pending
                and message.origin
                and message.origin.get("kind") == "human"
            ):
                self.context = self.pending
                return (self.event("turn_started", turn_id=self.pending),)
            if message.origin and message.origin.get("kind") != "human":
                self.context = None
                return ()
            if (
                isinstance(message.content, list)
                and message.content
                and all(isinstance(b, ToolResultBlock) for b in message.content)
            ):
                return tuple(
                    event
                    for b in message.content
                    if (event := self._tool_result(obj(asdict(b)))) is not None
                )
            self.context = None
            return ()
        if isinstance(message, StreamEvent):
            return self._stream(message)
        if isinstance(message, AssistantMessage):
            identity = message.message_id or message.uuid
            turn = self._messages.get(identity or "") or (
                self._tools.get(message.parent_tool_use_id)
                if message.parent_tool_use_id
                else self.context
            )
            if turn is None or identity is None:
                return (
                    self.event(
                        "session_notice",
                        message="Claude output could not be attributed to a saved turn.",
                    ),
                )
            self._messages[identity] = turn
            events = []
            receipt = message.uuid or identity
            offset = self._completed_messages.get(receipt)
            if offset is None:
                offset = (
                    self._latest_block[identity]
                    if len(message.content) == 1 and identity in self._latest_block
                    else self._final_offsets.get(identity, 0)
                )
                self._completed_messages[receipt] = offset
                self._final_offsets[identity] = offset + len(message.content)
            for index, block in enumerate(message.content, start=offset):
                raw = obj(asdict(block))
                # SDK content dataclasses omit the native type discriminator.
                native_types = {
                    "TextBlock": "text",
                    "ThinkingBlock": "thinking",
                    "ToolUseBlock": "tool_use",
                }
                raw["type"] = native_types.get(type(block).__name__, "unknown")
                item = self._item(turn, identity, index, raw, complete=True)
                if item is not None:
                    events.append(self.event("item", turn_id=turn, item=item))
            usage = message.usage or {}
            if type(usage.get("input_tokens")) is int:
                used = sum(
                    value
                    for key in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                    if type(value := usage.get(key, 0)) is int
                )
                events.append(self.event("context_usage", context_used_tokens=used))
            return tuple(events)
        if isinstance(message, ResultMessage):
            if not message.origin:
                self.unattributed_result = self.pending is not None
                return ()
            if message.origin.get("kind") != "human":
                return ()
            run = self.pending
            if run is None:
                return ()
            ambiguous = self.context != run
            self.unattributed_result = ambiguous
            status = (
                "interrupted"
                if self.interrupted
                else "failed"
                if message.is_error or ambiguous
                else "completed"
            )
            error = (
                "Claude's result could not be matched to its submitted input."
                if ambiguous
                else "\n".join(message.errors or [])
                or (message.result if message.is_error else None)
            )
            event = self.event(
                "turn_completed", turn_id=run, status=status, message=error
            )
            self.pending = self.context = None
            pending_tools = {tool for _, tool in self._tasks.values()}
            self._items = {
                identity: item
                for identity, item in self._items.items()
                if identity in pending_tools
            }
            self._tools = {
                identity: turn
                for identity, turn in self._tools.items()
                if identity in pending_tools
            }
            self._streams.clear()
            self._messages.clear()
            self._block_ids.clear()
            self._latest_block.clear()
            self._final_offsets.clear()
            self._completed_messages.clear()
            return (event,)
        if isinstance(message, SystemMessage):
            data = obj(message.data)
            task = str(data.get("task_id", ""))
            tool = str(data.get("tool_use_id", ""))
            if task and tool in self._tools:
                self._tasks[task] = self._tools[tool], tool
            if (
                message.subtype in {"task_notification", "task_updated"}
                and task in self._tasks
            ):
                status = data.get("status") or obj(data.get("patch")).get("status")
                if status in {"completed", "failed", "stopped"}:
                    _, completed_tool = self._tasks.pop(task)
                    if not any(
                        value[1] == completed_tool for value in self._tasks.values()
                    ):
                        self._tools.pop(completed_tool, None)
                        self._items.pop(completed_tool, None)
            if message.subtype == "compact_boundary":
                return (
                    self.event(
                        "session_notice",
                        message="Claude compacted its native context. Saved FCC history is retained.",
                    ),
                )
        return ()

    def _stream(self, message: StreamEvent) -> tuple[HarnessEvent, ...]:
        raw = obj(message.event)
        kind = raw.get("type")
        parent = message.parent_tool_use_id
        if kind == "message_start":
            native = obj(raw.get("message"))
            identity = native.get("id")
            turn = self._tools.get(parent) if parent else self.context
            if isinstance(identity, str) and turn is not None:
                self._streams[parent] = identity
                self._messages[identity] = turn
            return ()
        identity = self._streams.get(parent)
        if identity is None:
            return ()
        turn = self._messages[identity]
        index = raw.get("index")
        if type(index) is not int:
            return ()
        if kind == "content_block_start":
            self._latest_block[identity] = index
            item = self._item(
                turn, identity, index, obj(raw.get("content_block")), complete=False
            )
        else:
            item_id = self._block_ids.get((identity, index))
            item = self._items.get(item_id or "")
            if item is None:
                return ()
            if kind == "content_block_delta":
                delta = obj(raw.get("delta"))
                text = delta.get("text", delta.get("thinking"))
                if isinstance(text, str):
                    item = replace(item, text=item.text + text)
                elif isinstance(delta.get("partial_json"), str):
                    detail = "" if item.detail == "{}" else item.detail
                    item = replace(item, detail=detail + str(delta["partial_json"]))
                else:
                    return ()
            elif kind == "content_block_stop":
                item = replace(item, complete=item.kind != "tool")
            else:
                return ()
            self._items[item.item_id] = item
        return (self.event("item", turn_id=turn, item=item),) if item else ()

    @classmethod
    def history(
        cls,
        session_id: str,
        messages: Sequence[SessionMessage],
        submitted_run_ids: frozenset[str],
    ) -> NativeThread:
        protocol = cls("history", session_id)
        turns: dict[str, dict[str, ItemUpdate]] = {}
        current: str | None = None
        offsets: dict[str, int] = {}
        seen: set[str] = set()
        for message in messages:
            if message.uuid in seen:
                continue
            seen.add(message.uuid)
            native = obj(message.message)
            content = native.get("content", [])
            blocks = content if isinstance(content, list) else []
            if message.type == "user" and message.uuid in submitted_run_ids:
                current = message.uuid
                text = (
                    content
                    if isinstance(content, str)
                    else "\n".join(str(obj(b).get("text", "")) for b in blocks)
                )
                turns[current] = {
                    current: ItemUpdate(
                        current,
                        current,
                        "user",
                        text=text,
                        complete=True,
                        client_id=current,
                    )
                }
                continue
            if message.type == "user":
                if blocks and all(
                    obj(block).get("type") == "tool_result" for block in blocks
                ):
                    for block in blocks:
                        event = protocol._tool_result(obj(block))
                        if event and event.item and event.item.turn_id in turns:
                            turns[event.item.turn_id][event.item.item_id] = event.item
                else:
                    current = None
                continue
            if current is None:
                continue
            identity = str(native.get("id") or message.uuid)
            offset = offsets.get(identity, 0)
            offsets[identity] = offset + len(blocks)
            for index, block in enumerate(blocks, start=offset):
                item = protocol._item(
                    current, identity, index, obj(block), complete=True
                )
                if item:
                    turns[current][item.item_id] = item
        return NativeThread(
            session_id,
            tuple(
                NativeTurn(turn, tuple(items.values())) for turn, items in turns.items()
            ),
            history_notice=(
                "Some submitted inputs are unavailable in Claude's native history. "
                "Saved FCC history is retained. Input was not resent."
                if submitted_run_ids.difference(turns)
                else None
            ),
        )
