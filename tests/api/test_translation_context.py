"""Verify which parts of a request or reply can cross a provider boundary."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.core.json_types import JsonObject
from tests.api.test_four_protocol_matrix import (
    ingress_request,
    matrix_app,
    sent_request,
    translation_error,
)
from tests.providers.test_gemini import _sse


@pytest.mark.parametrize("wire", ["chat", "messages", "responses"])
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_file_reference_with_text_requires_its_provider(
    wire: str, protocol: str
) -> None:
    attachment: JsonObject
    if wire == "chat":
        attachment = {"type": "file", "file": {"file_id": "file-example"}}
    elif wire == "messages":
        attachment = {
            "type": "document",
            "source": {"type": "file", "file_id": "file-example"},
        }
    else:
        attachment = {"type": "input_file", "file_id": "file-example"}
    text: JsonObject = {
        "type": "input_text" if wire == "responses" else "text",
        "text": "Summarize the attachment",
    }
    path, body = ingress_request(
        wire, True, [{"role": "user", "content": [attachment, text]}]
    )
    if wire == protocol:
        assert "file-example" in json.dumps(sent_request(protocol, path, body))
    else:
        assert "file" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_messages_thread_continuation_requires_its_provider(protocol: str) -> None:
    path, body = ingress_request(
        "messages",
        False,
        [{"role": "user", "content": "What was my earlier question?"}],
    )
    body.pop("tools")
    body["thread"] = {"type": "continue", "previous_message_id": "msg_earlier"}
    if protocol == "messages":
        assert sent_request(protocol, path, body)["thread"] == body["thread"]
    else:
        assert "thread" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_program_intermediate_results_stay_with_the_messages_executor(
    protocol: str,
) -> None:
    path, body = ingress_request(
        "messages",
        False,
        [
            {"role": "user", "content": "Count the rows"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "name": "code_execution",
                        "id": "srv1",
                        "input": {"code": "print(len(rows))"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read",
                        "input": {},
                        "caller": {
                            "type": "code_execution_20260120",
                            "tool_id": "srv1",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "intermediate-row",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "code_execution_tool_result",
                        "tool_use_id": "srv1",
                        "content": {
                            "type": "code_execution_result",
                            "stdout": "1",
                            "stderr": "",
                            "return_code": 0,
                            "content": [],
                        },
                    }
                ],
            },
            {"role": "user", "content": "Explain the count"},
        ],
    )
    if protocol == "messages":
        upstream = json.dumps(sent_request(protocol, path, body))
        assert "intermediate-row" in upstream and "code_execution_20260120" in upstream
    else:
        assert "program" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("execution", ["client", "server"])
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_discovery_configuration_never_becomes_conversation_text(
    execution: str, protocol: str
) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Look up available tools"},
            {
                "type": "tool_search_call",
                "execution": execution,
                "call_id": "s1",
                "status": "completed",
                "arguments": {"query": "catalog"},
            },
            {
                "type": "tool_search_output",
                "execution": execution,
                "call_id": "s1",
                "status": "completed",
                "tools": [
                    {
                        "type": "mcp",
                        "server_label": "example",
                        "server_url": "https://connector.example.invalid",
                        "headers": {"X-Example-Setting": "connector-only-value"},
                    },
                    {
                        "type": "function",
                        "name": "inspect_catalog",
                        "parameters": {"type": "object"},
                        "provider_option": "registry-only-value",
                    },
                ],
            },
            {"role": "user", "content": "Continue"},
        ],
    )
    sent = sent_request(protocol, path, body)
    captured = json.dumps(sent)
    assert "inspect_catalog" in captured
    assert ("connector-only-value" in captured) == (protocol == "responses")
    assert ("registry-only-value" in captured) == (protocol == "responses")
    if execution == "client" and protocol != "responses":
        assert "inspect_catalog" in json.dumps(sent.get("tools"))
        body["tool_choice"] = {"type": "function", "name": "inspect_catalog"}
        sent_request(protocol, path, body)


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini"])
def test_extension_with_an_explicit_user_turn_preserves_the_question(
    protocol: str,
) -> None:
    last: JsonObject = {
        "type": "future_message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Actual question"}],
    }
    path, body = ingress_request(
        "responses", True, [{"role": "user", "content": "Older question"}, last]
    )
    assert "Actual question" in json.dumps(sent_request(protocol, path, body))
    last["content"] = [{"type": "future_media"}]
    assert "latest user" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_computer_approval_must_travel_with_the_screenshot(protocol: str) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Read the screen"},
            {
                "type": "computer_call_output",
                "call_id": "c1",
                "acknowledged_safety_checks": [
                    {"id": "approved-item", "code": "example", "message": "Reviewed"}
                ],
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                },
            },
            {"role": "user", "content": "What was visible?"},
        ],
    )
    if protocol == "responses":
        assert "approved-item" in json.dumps(sent_request(protocol, path, body))
    else:
        assert "safety" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini", "responses"])
def test_responses_compatibility_signature_stays_with_its_owner(protocol: str) -> None:
    call: JsonObject = {
        "type": "function_call",
        "call_id": "r",
        "name": "read",
        "arguments": "{}",
        "extra_content": {"google": {"thought_signature": "opaque-call-state"}},
    }
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Read the file"},
            call,
            {"type": "function_call_output", "call_id": "r", "output": "file value"},
            {"role": "user", "content": "Explain"},
        ],
    )
    if protocol == "responses":
        assert "opaque-call-state" in json.dumps(sent_request(protocol, path, body))
    else:
        assert "signed" in translation_error(protocol, path, body).lower()
    call["extra_content"] = {"optional_note": "metadata-only"}
    assert ("metadata-only" in json.dumps(sent_request(protocol, path, body))) == (
        protocol == "responses"
    )


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini"])
@pytest.mark.parametrize("batched", [False, True])
def test_completed_computer_actions_keep_the_action_and_image(
    protocol: str, batched: bool
) -> None:
    action: JsonObject = {
        "type": "click",
        "button": "left",
        "x": 19,
        "y": 27,
        "vendor_option": "internal-action-option",
    }
    call: JsonObject = {
        "type": "computer_call",
        "id": "item_c",
        "call_id": "c",
        "pending_safety_checks": [],
        "status": "completed",
    }
    if batched:
        call["actions"] = [action, {"type": "screenshot"}]
    else:
        call["action"] = action
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Read the screen"},
            call,
            {
                "type": "computer_call_output",
                "call_id": "c",
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                },
            },
            {"role": "user", "content": "Where did you click?"},
        ],
    )
    sent = json.dumps(sent_request(protocol, path, body))
    assert "click" in sent and "19" in sent and "iVBORw0KGgo=" in sent
    assert "internal-action-option" not in sent


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
@pytest.mark.parametrize("inside_result", [False, True])
def test_nested_document_keeps_stored_image_with_native_messages(
    protocol: str, inside_result: bool
) -> None:
    document: JsonObject = {
        "type": "document",
        "source": {
            "type": "content",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "file", "file_id": "file-nested-photo"},
                }
            ],
        },
    }
    history: list[JsonObject] = [
        {
            "role": "user",
            "content": [document, {"type": "text", "text": "Describe that image"}],
        }
    ]
    if inside_result:
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "read", "id": "r", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "r",
                        "content": [
                            document,
                            {"type": "text", "text": "Image from the report"},
                        ],
                    }
                ],
            },
            {"role": "user", "content": "Describe that image"},
        ]
    path, body = ingress_request("messages", True, history)
    if protocol == "messages":
        assert "file-nested-photo" in json.dumps(sent_request(protocol, path, body))
    else:
        assert "file" in translation_error(protocol, path, body).lower()
    document.update(
        title="Report title",
        source={
            "type": "content",
            "content": [
                {"type": "text", "text": "Report caption"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "Y2F0",
                    },
                },
            ],
        },
    )
    portable = json.dumps(sent_request(protocol, path, body))
    assert all(
        value in portable for value in ("Report title", "Report caption", "Y2F0")
    )


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_completed_web_history_only_exposes_public_results(protocol: str) -> None:
    path, body = ingress_request(
        "messages",
        True,
        [
            {"role": "user", "content": "Search the catalog"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "name": "web_search",
                        "id": "s",
                        "input": {"query": "catalog"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "s",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Public result",
                                "url": "https://example.invalid/catalog",
                                "encrypted_content": "private-search-state",
                                "vendor_option": "private-search-option",
                            }
                        ],
                    },
                ],
            },
            {"role": "user", "content": "Which source?"},
        ],
    )
    body.update(
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        tool_choice={"type": "none"},
    )
    sent = json.dumps(sent_request(protocol, path, body))
    assert "Public result" in sent and "https://example.invalid/catalog" in sent
    assert ("private-search-state" in sent) == (protocol == "messages")
    assert ("private-search-option" in sent) == (protocol == "messages")


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
def test_completed_web_fetch_preserves_text_and_reports_native_only_documents(
    protocol: str,
) -> None:
    document: JsonObject = {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": "Previously fetched page",
        },
    }
    path, body = ingress_request(
        "messages",
        True,
        [
            {"role": "user", "content": "Read the page"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "fetch1",
                        "name": "web_fetch",
                        "input": {"url": "https://example.invalid/page"},
                    },
                    {
                        "type": "web_fetch_tool_result",
                        "tool_use_id": "fetch1",
                        "content": {
                            "type": "web_fetch_result",
                            "url": "https://example.invalid/page",
                            "content": document,
                        },
                    },
                ],
            },
            {"role": "user", "content": "Which page was that?"},
        ],
    )
    assert "Previously fetched page" in json.dumps(sent_request(protocol, path, body))
    document["source"] = {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "JVBERi0xLjQK",
    }
    if protocol == "messages":
        assert "JVBERi0xLjQK" in json.dumps(sent_request(protocol, path, body))
    else:
        assert "native Messages" in translation_error(protocol, path, body)


@pytest.mark.parametrize(
    ("wire", "streaming"),
    [
        ("chat", False),
        ("chat", True),
        ("messages", False),
        ("messages", True),
        ("responses", True),
        ("gemini", False),
        ("gemini", True),
    ],
)
def test_gemini_native_only_reply_cannot_finish_as_an_empty_answer(
    wire: str, streaming: bool
) -> None:
    def reply(request: httpx.Request):
        return _sse(
            [
                {"executableCode": {"language": "PYTHON", "code": "value = 6"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "6"}},
            ]
        )

    path, body = ingress_request(wire, streaming)
    with TestClient(matrix_app("gemini", native_reply=reply, max_attempts=1)) as client:
        response = client.post(path, json=body)
    if wire == "gemini":
        assert response.status_code == 200 and "codeExecutionResult" in response.text
    else:
        assert response.status_code == 502, response.text
        assert "unsupported model output" in response.text
