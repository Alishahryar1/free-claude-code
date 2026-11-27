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
    captured = json.dumps(sent_request(protocol, path, body))
    assert "inspect_catalog" in captured
    assert ("connector-only-value" in captured) == (protocol == "responses")
    assert ("registry-only-value" in captured) == (protocol == "responses")


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
