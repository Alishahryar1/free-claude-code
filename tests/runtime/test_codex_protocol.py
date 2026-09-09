import pytest

from free_claude_code.application.code_sessions import CodeValidationError
from free_claude_code.core.json_types import JsonObject
from free_claude_code.runtime.codex_protocol import CodexProtocol, NativePrompt


@pytest.mark.parametrize(
    "status,label",
    [
        ("approved", "Approved"),
        ("denied", "Denied"),
        ("timedOut", "Timed out"),
        ("aborted", "Aborted"),
    ],
)
def test_auto_review_has_own_identity_and_preserves_native_action_and_rationale(
    status, label
):
    protocol = CodexProtocol("generation")
    payload: JsonObject = {
        "threadId": "thread",
        "turnId": "turn",
        "reviewId": "review",
        "targetItemId": None,
        "action": {
            "type": "networkAccess",
            "host": "localhost",
            "port": 8000,
            "protocol": "http",
            "target": "http://localhost:8000",
        },
        "review": {"status": "inProgress"},
        "startedAtMs": 10,
    }
    started = protocol.notification("item/autoApprovalReview/started", payload)
    assert started is not None and started.item is not None
    assert started.item.item_id == "auto-review:review"
    assert not started.item.complete
    completed = protocol.notification(
        "item/autoApprovalReview/completed",
        {
            **payload,
            "review": {
                "status": status,
                "rationale": "Native explanation",
                "riskLevel": "low",
            },
            "completedAtMs": 20,
        },
    )
    assert completed is not None and completed.item is not None
    assert completed.item.item_id == started.item.item_id
    assert completed.item.title == f"Auto-review: {label}"
    assert completed.item.complete
    assert completed.item.text == "Native explanation"
    assert "localhost" in completed.item.detail
    assert completed.item.raw["review"] == {
        "status": status,
        "rationale": "Native explanation",
        "riskLevel": "low",
    }


def test_native_guardian_warning_is_a_notice_without_a_turn_id():
    event = CodexProtocol("generation").notification(
        "guardianWarning", {"threadId": "thread", "message": "Review is unavailable"}
    )
    assert event is not None
    assert event.kind == "notice" and event.turn_id is None
    assert event.message == "Review is unavailable"


def notification(protocol, method, **extra):
    return protocol.notification(
        method, {"threadId": "thread", "turnId": "turn", "itemId": "item", **extra}
    )


def test_reasoning_keeps_summary_and_content_indices_and_original_completion():
    protocol = CodexProtocol("generation")
    notification(
        protocol, "item/reasoning/summaryTextDelta", summaryIndex=1, delta="Second"
    )
    notification(
        protocol, "item/reasoning/summaryTextDelta", summaryIndex=0, delta="First"
    )
    notification(
        protocol,
        "item/reasoning/textDelta",
        contentIndex=0,
        delta="Available reasoning",
    )
    event = notification(
        protocol,
        "item/completed",
        item={
            "id": "item",
            "type": "reasoning",
            "summary": [],
            "content": [],
            "opaque": "keep",
        },
    )
    assert event.item.text == "First\n\nSecond\n\nAvailable reasoning"
    assert event.item.raw["item"]["opaque"] == "keep"
    assert event.item.raw["stream"]["summary"] == {"0": "First", "1": "Second"}


def test_text_final_replaces_partial_without_losing_source_metadata():
    protocol = CodexProtocol("generation")
    notification(protocol, "item/agentMessage/delta", delta="Hel")
    event = notification(
        protocol,
        "item/completed",
        item={
            "id": "item",
            "type": "agentMessage",
            "text": "Hello",
            "delivery": "async",
            "questions": [{"title": "Continue?"}],
        },
    )
    assert event.item.text == "Hello"
    assert event.item.raw["item"]["questions"] == [{"title": "Continue?"}]
    assert event.item.complete


def test_retryable_error_does_not_turn_into_completion():
    event = notification(
        CodexProtocol("generation"), "error", error={"message": "retry"}, willRetry=True
    )
    assert event.kind == "error"
    assert event.message == "retry"
    assert event.will_retry is True
    assert event.error_details == {"message": "retry"}


def test_command_completion_keeps_output_and_exit_code():
    protocol = CodexProtocol("generation")
    notification(protocol, "item/commandExecution/outputDelta", delta="output\n")
    event = notification(
        protocol,
        "item/completed",
        item={
            "id": "item",
            "type": "commandExecution",
            "command": "pytest",
            "cwd": "/work",
            "status": "completed",
            "exitCode": 1,
        },
    )
    assert "output" in event.item.detail
    assert "1" in event.item.detail
    assert event.item.title == "pytest"


def test_approval_can_only_select_exact_native_offered_decisions():
    amendment = {"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["pytest"]}}
    prompt = NativePrompt(
        "item/commandExecution/requestApproval",
        7,
        {
            "threadId": "thread",
            "turnId": "turn",
            "itemId": "item",
            "command": "pytest",
            "availableDecisions": ["accept", amendment, "decline", "cancel"],
        },
    )
    assert prompt.answer({"choice": "1"}) == {"decision": amendment}
    assert prompt.answer({"choice": "2"}) == {"decision": "decline"}
    with pytest.raises(CodeValidationError):
        prompt.answer({"choice": "acceptForSession"})


def test_questions_preserve_ids_and_free_text_and_need_an_answer():
    prompt = NativePrompt(
        "item/tool/requestUserInput",
        "7",
        {
            "threadId": "thread",
            "turnId": "turn",
            "itemId": "item",
            "isBlocking": False,
            "questions": [
                {
                    "id": "choice",
                    "header": "Choose",
                    "question": "Where?",
                    "options": [{"label": "Here", "description": "This folder"}],
                    "isOther": True,
                    "isSecret": False,
                }
            ],
        },
    )
    assert prompt.answer({"answers": {"choice": ["Elsewhere"]}}) == {
        "answers": {"choice": {"answers": ["Elsewhere"]}}
    }
    with pytest.raises(CodeValidationError):
        prompt.answer({"answers": {"unknown": ["x"]}})


def test_permission_grant_is_selected_subset_and_keeps_denies():
    prompt = NativePrompt(
        "item/permissions/requestApproval",
        8,
        {
            "threadId": "thread",
            "turnId": "turn",
            "itemId": "item",
            "permissions": {
                "network": {"enabled": True},
                "fileSystem": {
                    "read": ["/work"],
                    "entries": [
                        {"access": "deny", "path": {"type": "path", "path": "/secret"}}
                    ],
                },
            },
        },
    )
    response = prompt.answer({"selected": ["network"], "scope": "turn"})
    assert response == {
        "scope": "turn",
        "permissions": {
            "network": {"enabled": True},
            "fileSystem": {
                "entries": [
                    {"access": "deny", "path": {"type": "path", "path": "/secret"}}
                ]
            },
        },
    }
    with pytest.raises(CodeValidationError):
        prompt.answer({"selected": ["unoffered"], "scope": "session"})


def test_mcp_form_answers_use_schema_types_and_reject_invalid_values():
    prompt = NativePrompt(
        "mcpServer/elicitation/request",
        9,
        {
            "threadId": "thread",
            "turnId": None,
            "serverName": "local",
            "mode": "form",
            "message": "Choose a limit",
            "requestedSchema": {
                "type": "object",
                "required": ["limit"],
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 4}
                },
            },
        },
    )
    assert prompt.answer({"action": "accept", "values": {"limit": 2}}) == {
        "action": "accept",
        "content": {"limit": 2},
    }
    with pytest.raises(CodeValidationError):
        prompt.answer({"action": "accept", "values": {"limit": 7}})
    assert prompt.answer({"action": "decline"}) == {
        "action": "decline",
        "content": None,
    }


def test_native_form_rejects_invalid_documented_string_format():
    prompt = NativePrompt(
        "mcpServer/elicitation/request",
        1,
        {
            "threadId": "thread",
            "mode": "form",
            "serverName": "tool",
            "message": "Email",
            "requestedSchema": {
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
            },
        },
    )
    with pytest.raises(CodeValidationError):
        prompt.answer({"action": "accept", "values": {"email": "invalid"}})
    assert (
        prompt.answer({"action": "accept", "values": {"email": "user@example.com"}})[
            "action"
        ]
        == "accept"
    )


@pytest.mark.parametrize("choices,supported", [(["one", "two"], True), ([1, 2], False)])
def test_multiselect_projection_matches_the_negotiated_string_contract(
    choices, supported
):
    prompt = NativePrompt(
        "mcpServer/elicitation/request",
        1,
        {
            "threadId": "thread",
            "mode": "form",
            "serverName": "tool",
            "requestedSchema": {
                "type": "object",
                "properties": {
                    "selection": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"const": value, "title": str(value)}
                                for value in choices
                            ]
                        },
                    }
                },
            },
        },
    )
    assert prompt.form["unsupported"] is not supported
    if supported:
        assert prompt.answer({"action": "accept", "values": {"selection": choices}})[
            "content"
        ] == {"selection": choices}
    else:
        with pytest.raises(CodeValidationError, match="not supported"):
            prompt.answer({"action": "accept", "values": {"selection": choices}})
    for action in ("decline", "cancel"):
        assert prompt.answer({"action": action}) == {"action": action, "content": None}


def test_history_preserves_native_turns_client_ids_and_failure_details():
    native = CodexProtocol("g").history(
        {
            "thread": {
                "id": "thread",
                "turns": [
                    {
                        "id": "first",
                        "status": "failed",
                        "error": {
                            "message": "Rejected",
                            "additionalDetails": "details",
                        },
                        "items": [
                            {
                                "id": "user",
                                "type": "userMessage",
                                "clientId": "operation",
                                "content": [{"type": "text", "text": "Hello"}],
                            }
                        ],
                    }
                ],
            }
        }
    )
    assert native.turns[0].items[0].client_id == "operation"
    assert native.turns[0].status == "failed"
    assert native.turns[0].error_details == {
        "message": "Rejected",
        "additionalDetails": "details",
    }
