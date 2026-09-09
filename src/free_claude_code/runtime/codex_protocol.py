"""Codex app-server v2 item and interactive-request projection."""

import json
import math
from dataclasses import dataclass, field

from jsonschema import FormatChecker

from free_claude_code.application.code_sessions.models import (
    CodeValidationError,
    HarnessEvent,
    ItemUpdate,
    NativeThread,
    NativeTurn,
    PromptRequest,
    RunStatus,
)
from free_claude_code.core.json_types import JsonObject, JsonValue


def object_value(value: object) -> JsonObject:
    return dict(value) if isinstance(value, dict) else {}


def array_value(value: object) -> list[JsonValue]:
    return list(value) if isinstance(value, list | tuple) else []


def string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def pretty(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


@dataclass(slots=True)
class _Item:
    raw: JsonObject = field(default_factory=dict)
    text: str = ""
    output: str = ""
    summary: dict[int, str] = field(default_factory=dict)
    reasoning: dict[int, str] = field(default_factory=dict)


class CodexProtocol:
    def __init__(self, generation: str) -> None:
        self.generation = generation
        self._items: dict[tuple[str, str], _Item] = {}

    def notification(self, method: str, params: JsonObject) -> HarnessEvent | None:
        thread_id = string_value(params.get("threadId")) or None
        turn = object_value(params.get("turn"))
        turn_id = (
            string_value(params.get("turnId")) or string_value(turn.get("id")) or None
        )
        if method in {"turn/started", "turn/completed"}:
            status: RunStatus = "completed"
            if turn.get("status") == "interrupted":
                status = "interrupted"
            elif turn.get("status") == "failed":
                status = "failed"
            return HarnessEvent(
                self.generation,
                thread_id,
                "turn_started" if method == "turn/started" else "turn_completed",
                turn_id=turn_id,
                status=status,
                message=string_value(object_value(turn.get("error")).get("message"))
                or None,
                error_details=object_value(turn.get("error")),
            )
        if method == "error":
            return HarnessEvent(
                self.generation,
                thread_id,
                "error",
                turn_id=turn_id,
                message=string_value(object_value(params.get("error")).get("message")),
                will_retry=params.get("willRetry") is True,
                error_details=object_value(params.get("error")),
            )
        if method == "serverRequest/resolved":
            request_id = params.get("requestId")
            if isinstance(request_id, str | int) and not isinstance(request_id, bool):
                return HarnessEvent(
                    self.generation, thread_id, "resolved", request_id=request_id
                )
            return None
        if method == "guardianWarning" and thread_id:
            return HarnessEvent(
                self.generation,
                thread_id,
                "notice",
                message=string_value(params.get("message")),
                raw=dict(params),
            )
        if not thread_id or not turn_id:
            return None
        if method in {
            "item/autoApprovalReview/started",
            "item/autoApprovalReview/completed",
        }:
            review_id = string_value(params.get("reviewId"))
            if not review_id:
                return None
            review = object_value(params.get("review"))
            status = string_value(review.get("status"))
            label = {
                "inProgress": "Reviewing",
                "approved": "Approved",
                "denied": "Denied",
                "timedOut": "Timed out",
                "aborted": "Aborted",
            }.get(status, status)
            return HarnessEvent(
                self.generation,
                thread_id,
                "item",
                turn_id=turn_id,
                item=ItemUpdate(
                    turn_id,
                    f"auto-review:{review_id}",
                    "auto_review",
                    title=f"Auto-review: {label}",
                    text=string_value(review.get("rationale")),
                    detail=pretty({"action": params.get("action"), "review": review}),
                    raw=dict(params),
                    complete=method == "item/autoApprovalReview/completed",
                ),
            )
        raw = object_value(params.get("item"))
        item_id = string_value(params.get("itemId")) or string_value(raw.get("id"))
        if not item_id:
            return None
        item = self._items.setdefault((turn_id, item_id), _Item())
        complete = method == "item/completed"
        delta = string_value(params.get("delta"))
        if method in {"item/started", "item/completed"}:
            item.raw = {**item.raw, **raw}
        elif method == "item/agentMessage/delta":
            item.raw.setdefault("type", "agentMessage")
            item.text += delta
        elif method in {
            "item/reasoning/summaryTextDelta",
            "item/reasoning/textDelta",
            "item/reasoning/summaryPartAdded",
        }:
            item.raw.setdefault("type", "reasoning")
            summary = method != "item/reasoning/textDelta"
            index = params.get("summaryIndex" if summary else "contentIndex", 0)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                return None
            parts = item.summary if summary else item.reasoning
            parts[index] = parts.get(index, "") + delta
        elif method in {
            "item/commandExecution/outputDelta",
            "item/fileChange/outputDelta",
        }:
            item.raw.setdefault(
                "type", "fileChange" if "fileChange" in method else "commandExecution"
            )
            item.output += delta
        elif method == "item/fileChange/patchUpdated":
            item.raw["changes"] = params.get("changes", [])
        elif method == "item/mcpToolCall/progress":
            item.output += string_value(params.get("message")) + "\n"
        else:
            return None
        return HarnessEvent(
            self.generation,
            thread_id,
            "item",
            turn_id=turn_id,
            item=self._project(turn_id, item_id, item, complete),
        )

    def history(self, payload: JsonObject) -> NativeThread:
        thread = object_value(payload.get("thread"))
        thread_id = string_value(thread.get("id"))
        if not thread_id:
            raise CodeValidationError("Codex did not return a conversation ID.")
        turns: list[NativeTurn] = []
        for raw_turn in array_value(thread.get("turns")):
            turn = object_value(raw_turn)
            turn_id = string_value(turn.get("id"))
            items: list[ItemUpdate] = []
            for value in array_value(turn.get("items")):
                raw = object_value(value)
                item_id = string_value(raw.get("id"))
                if not turn_id or not item_id:
                    continue
                accumulated = self._items.setdefault((turn_id, item_id), _Item())
                accumulated.raw = {**accumulated.raw, **raw}
                items.append(self._project(turn_id, item_id, accumulated, True))
            if turn_id:
                error = object_value(turn.get("error"))
                status: RunStatus = (
                    "failed"
                    if turn.get("status") == "failed"
                    else "interrupted"
                    if turn.get("status") == "interrupted"
                    else "completed"
                )
                turns.append(
                    NativeTurn(
                        turn_id,
                        tuple(items),
                        status,
                        string_value(error.get("message")) or None,
                        error,
                    )
                )
        return NativeThread(thread_id, tuple(turns))

    @staticmethod
    def _project(turn_id: str, item_id: str, item: _Item, complete: bool) -> ItemUpdate:
        raw = item.raw
        native_kind = string_value(raw.get("type"))
        kind, text, title, detail = "tool", "", native_kind, ""
        if native_kind == "userMessage":
            kind = "user"
            text = "\n".join(
                string_value(object_value(value).get("text"))
                for value in array_value(raw.get("content"))
            )
        elif native_kind == "agentMessage":
            kind = "text"
            text = (
                string_value(raw.get("text"))
                if complete and "text" in raw
                else item.text or string_value(raw.get("text"))
            )
            questions = array_value(raw.get("questions"))
            if questions:
                detail = "\n\n".join(
                    string_value(object_value(question).get("title"))
                    for question in questions
                )
        elif native_kind == "reasoning":
            kind, title = "reasoning", "Thinking"
            summary = [string_value(value) for value in array_value(raw.get("summary"))]
            content = [string_value(value) for value in array_value(raw.get("content"))]
            if not any(summary):
                summary = [value for _, value in sorted(item.summary.items())]
            if not any(content):
                content = [value for _, value in sorted(item.reasoning.items())]
            text = "\n\n".join(value for value in [*summary, *content] if value)
        elif native_kind == "commandExecution":
            title = string_value(raw.get("command")) or "Command"
            output = string_value(raw.get("aggregatedOutput")) or item.output
            detail = "\n".join(
                value
                for value in [
                    string_value(raw.get("cwd")),
                    output,
                    f"Exit code: {raw['exitCode']}"
                    if raw.get("exitCode") is not None
                    else "",
                ]
                if value
            )
        elif native_kind == "fileChange":
            kind, title = "tool", "File changes"
            changes = []
            for value in array_value(raw.get("changes")):
                change = object_value(value)
                changes.append(
                    "\n".join(
                        [
                            string_value(change.get("path")),
                            string_value(change.get("diff")),
                        ]
                    )
                )
            detail = "\n\n".join(changes) or item.output
        elif native_kind == "mcpToolCall":
            title = ".".join(
                value
                for value in [
                    string_value(raw.get("server")),
                    string_value(raw.get("tool")),
                ]
                if value
            )
            detail = "\n\n".join(
                value
                for value in [
                    pretty(raw.get("arguments")),
                    pretty(raw.get("result")) if raw.get("result") is not None else "",
                    pretty(raw.get("error")) if raw.get("error") else "",
                    item.output,
                ]
                if value
            )
        else:
            title = native_kind or "Native item"
            detail = pretty(raw)
        source: JsonObject = {
            "item": dict(raw),
            "stream": {
                "text": item.text,
                "output": item.output,
                "summary": {
                    str(index): value for index, value in sorted(item.summary.items())
                },
                "reasoning": {
                    str(index): value for index, value in sorted(item.reasoning.items())
                },
            },
        }
        return ItemUpdate(
            turn_id,
            item_id,
            kind,
            text=text,
            title=title,
            detail=detail,
            raw=source,
            complete=complete,
            client_id=string_value(raw.get("clientId")) or None,
        )


class NativePrompt:
    def __init__(self, method: str, request_id: str | int, params: JsonObject) -> None:
        self.method, self.request_id, self.params = method, request_id, params
        self.decisions: list[JsonValue] = []
        self.grants: dict[str, tuple[str, JsonValue]] = {}
        self.kind, self.form = self._form()

    def request(self) -> PromptRequest:
        return PromptRequest(
            self.request_id,
            self.kind,
            self.form,
            dict(self.params),
            string_value(self.params.get("turnId")) or None,
            string_value(self.params.get("itemId")) or None,
        )

    def _form(self) -> tuple[str, JsonObject]:
        params = self.params
        if self.method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            offered = params.get("availableDecisions")
            self.decisions = (
                array_value(offered)
                if isinstance(offered, list)
                else ["accept", "acceptForSession", "decline", "cancel"]
            )
            labels = {
                "accept": "Allow once",
                "acceptForSession": "Allow for this session",
                "decline": "Decline",
                "cancel": "Cancel turn",
            }
            choices: list[JsonValue] = []
            for index, decision in enumerate(self.decisions):
                if isinstance(decision, str):
                    label = labels.get(decision, decision)
                elif "acceptWithExecpolicyAmendment" in object_value(decision):
                    amendment = object_value(
                        object_value(decision).get("acceptWithExecpolicyAmendment")
                    )
                    label = "Allow and remember command prefix: " + " ".join(
                        string_value(value)
                        for value in array_value(amendment.get("execpolicy_amendment"))
                    )
                else:
                    label = "Apply requested network rule: " + pretty(decision)
                choices.append({"id": str(index), "label": label})
            details = [
                string_value(params.get(key))
                for key in ("command", "cwd", "grantRoot", "reason")
            ]
            if params.get("additionalPermissions"):
                details.append(pretty(params["additionalPermissions"]))
            return "approval", {
                "title": "Permission requested",
                "detail": "\n".join(value for value in details if value),
                "choices": choices,
            }
        if self.method == "item/tool/requestUserInput":
            questions: list[JsonValue] = []
            for value in array_value(params.get("questions")):
                question = object_value(value)
                questions.append(
                    {
                        "id": question.get("id"),
                        "label": question.get("question"),
                        "header": question.get("header"),
                        "options": question.get("options"),
                        "allow_other": question.get("isOther", False),
                        "secret": question.get("isSecret", False),
                    }
                )
            return "questions", {
                "title": "Codex needs your input",
                "questions": questions,
            }
        if self.method == "item/permissions/requestApproval":
            permissions = object_value(params.get("permissions"))
            choices: list[JsonValue] = []
            if object_value(permissions.get("network")).get("enabled") is True:
                self.grants["network"] = ("network", {"enabled": True})
                choices.append({"id": "network", "label": "Network access"})
            files = object_value(permissions.get("fileSystem"))
            for access in ("read", "write", "entries"):
                for index, grant in enumerate(array_value(files.get(access))):
                    if (
                        access == "entries"
                        and object_value(grant).get("access") == "deny"
                    ):
                        continue
                    key = f"{access}:{index}"
                    self.grants[key] = (access, grant)
                    label = (
                        f"{access.title()}: {grant}"
                        if access != "entries"
                        else pretty(grant)
                    )
                    choices.append({"id": key, "label": label})
            return "permissions", {
                "title": "Additional permissions",
                "detail": string_value(params.get("reason")),
                "choices": choices,
            }
        if self.method == "mcpServer/elicitation/request":
            mode = params.get("mode")
            form: JsonObject = {
                "title": string_value(params.get("serverName")) or "Tool question",
                "detail": string_value(params.get("message")),
            }
            if mode == "url":
                form["url"] = params.get("url")
                return "url", form
            schema = object_value(params.get("requestedSchema"))
            fields: list[JsonValue] = []
            unsupported = mode != "form" or schema.get("type") != "object"
            required = array_value(schema.get("required"))
            for name, value in object_value(schema.get("properties")).items():
                definition = object_value(value)
                field_type = string_value(definition.get("type"))
                if field_type not in {
                    "string",
                    "integer",
                    "number",
                    "boolean",
                    "array",
                }:
                    unsupported = True
                choices = _enum_options(definition)
                if field_type == "array" and (
                    not choices
                    or any(
                        not isinstance(object_value(choice).get("value"), str)
                        for choice in choices
                    )
                ):
                    unsupported = True
                fields.append(
                    {
                        "id": name,
                        "label": definition.get("title", name),
                        "description": definition.get("description"),
                        "type": field_type,
                        "required": name in required,
                        "options": choices,
                        "default": definition.get("default"),
                        "minimum": definition.get("minimum"),
                        "maximum": definition.get("maximum"),
                        "minLength": definition.get("minLength"),
                        "maxLength": definition.get("maxLength"),
                    }
                )
            form.update({"fields": fields, "unsupported": unsupported})
            return "form", form
        raise CodeValidationError(f"Unsupported native interaction: {self.method}")

    def answer(self, answer: JsonObject) -> JsonObject:
        if self.kind == "approval":
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in {
                str(index) for index in range(len(self.decisions))
            }:
                raise CodeValidationError("Choose one of the offered decisions.")
            return {"decision": self.decisions[int(choice)]}
        if self.kind == "questions":
            values = object_value(answer.get("answers"))
            questions = [
                object_value(value)
                for value in array_value(self.params.get("questions"))
            ]
            if set(values) != {
                string_value(question.get("id")) for question in questions
            }:
                raise CodeValidationError("Answer each question before submitting.")
            result: JsonObject = {}
            for question in questions:
                key = string_value(question.get("id"))
                entries = array_value(values.get(key))
                if not entries or any(
                    not isinstance(value, str) or not value.strip() for value in entries
                ):
                    raise CodeValidationError("Answer each question before submitting.")
                options = {
                    string_value(object_value(value).get("label"))
                    for value in array_value(question.get("options"))
                }
                if (
                    options
                    and not question.get("isOther")
                    and any(value not in options for value in entries)
                ):
                    raise CodeValidationError("Choose an offered answer.")
                result[key] = {"answers": entries}
            return {"answers": result}
        if self.kind == "permissions":
            selected = array_value(answer.get("selected"))
            if any(
                not isinstance(value, str) or value not in self.grants
                for value in selected
            ) or answer.get("scope", "turn") not in {"turn", "session"}:
                raise CodeValidationError(
                    "Choose only the requested permissions and scope."
                )
            permissions: JsonObject = {}
            files: JsonObject = {}
            requested_files = object_value(
                object_value(self.params.get("permissions")).get("fileSystem")
            )
            denies = [
                value
                for value in array_value(requested_files.get("entries"))
                if object_value(value).get("access") == "deny"
            ]
            if denies:
                files["entries"] = denies
            for key in selected:
                access, value = self.grants[string_value(key)]
                if access == "network":
                    permissions["network"] = value
                else:
                    files[access] = [*array_value(files.get(access)), value]
            if files:
                if requested_files.get("globScanMaxDepth") is not None:
                    files["globScanMaxDepth"] = requested_files["globScanMaxDepth"]
                permissions["fileSystem"] = files
            return {"permissions": permissions, "scope": answer.get("scope", "turn")}
        action = answer.get("action")
        if action not in {"accept", "decline", "cancel"}:
            raise CodeValidationError(
                "Choose whether to answer or dismiss this prompt."
            )
        if action != "accept":
            return {"action": action, "content": None}
        if self.kind == "url":
            return {"action": "accept", "content": None}
        if self.form.get("unsupported"):
            raise CodeValidationError(
                "This tool's form is not supported. Decline or cancel it."
            )
        values = object_value(answer.get("values"))
        schema = object_value(self.params.get("requestedSchema"))
        properties = object_value(schema.get("properties"))
        if not set(values).issubset(properties) or any(
            name not in values for name in array_value(schema.get("required"))
        ):
            raise CodeValidationError("Complete the required fields.")
        for name, value in values.items():
            _validate_field(name, value, object_value(properties[name]))
        return {"action": "accept", "content": values}


def _enum_options(definition: JsonObject) -> list[JsonValue]:
    if definition.get("type") == "array":
        definition = object_value(definition.get("items"))
    values = array_value(definition.get("enum"))
    names = array_value(definition.get("enumNames"))
    if values:
        return [
            {"value": value, "label": names[index] if index < len(names) else value}
            for index, value in enumerate(values)
        ]
    return [
        {
            "value": object_value(value).get("const"),
            "label": object_value(value).get("title"),
        }
        for value in array_value(definition.get("oneOf", definition.get("anyOf")))
    ]


def _validate_field(name: str, value: JsonValue, definition: JsonObject) -> None:
    kind = definition.get("type")
    valid = True
    if kind == "boolean":
        valid = isinstance(value, bool)
    elif kind in {"number", "integer"}:
        valid = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        if valid and kind == "integer":
            valid = isinstance(value, int)
        if valid and isinstance(value, int | float):
            minimum, maximum = definition.get("minimum"), definition.get("maximum")
            valid = (not isinstance(minimum, int | float) or value >= minimum) and (
                not isinstance(maximum, int | float) or value <= maximum
            )
    elif kind == "string":
        valid = isinstance(value, str)
        if isinstance(value, str):
            minimum, maximum = definition.get("minLength"), definition.get("maxLength")
            valid = (not isinstance(minimum, int) or len(value) >= minimum) and (
                not isinstance(maximum, int) or len(value) <= maximum
            )
            format_name = definition.get("format")
            if isinstance(format_name, str):
                valid = valid and FormatChecker().conforms(value, format_name)
    elif kind == "array":
        valid = isinstance(value, list) and all(
            isinstance(entry, str) for entry in value
        )
        if valid and isinstance(value, list):
            minimum, maximum = definition.get("minItems"), definition.get("maxItems")
            valid = (not isinstance(minimum, int) or len(value) >= minimum) and (
                not isinstance(maximum, int) or len(value) <= maximum
            )
    else:
        valid = False
    options = [
        object_value(option).get("value") for option in _enum_options(definition)
    ]
    if options:
        candidates = array_value(value) if kind == "array" else [value]
        valid = valid and all(candidate in options for candidate in candidates)
    if not valid:
        raise CodeValidationError(f"Invalid value for {name}.")
