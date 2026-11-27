"""Inline attachments must reach a capable provider as actual file inputs."""

import pytest

from free_claude_code.core.json_types import JsonObject, JsonValue
from tests.api.test_four_protocol_matrix import (
    ingress_request,
    sent_request,
    translation_error,
)
from tests.wire_payload import object_array, object_payload

PDF = "data:application/pdf;base64,JVBERi0xLjQK"


@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize("target", ["chat", "responses", "messages", "gemini"])
@pytest.mark.parametrize("with_text", [True, False])
def test_user_file_is_delivered_in_order_or_rejected(
    wire: str, target: str, with_text: bool
) -> None:
    file: JsonObject = {"file_data": PDF, "filename": "Report.PDF"}
    part: JsonObject = (
        {"type": "file", "file": file}
        if wire == "chat"
        else {"type": "input_file", **file}
    )
    part["prompt_cache_breakpoint"] = {"ttl": "1h"}
    content: list[JsonValue] = [part]
    if with_text:
        content.append(
            {
                "type": "text" if wire == "chat" else "input_text",
                "text": "What is on page one?",
            }
        )
    path, body = ingress_request(wire, True, [{"role": "user", "content": content}])
    if target not in ("chat", "responses"):
        assert "file" in translation_error(target, path, body).lower()
        return
    sent = sent_request(target, path, body)
    message = next(
        item
        for item in object_array(sent["messages" if target == "chat" else "input"])
        if item["role"] == "user"
    )
    parts = object_array(message["content"])
    assert parts[0]["prompt_cache_breakpoint"] == part["prompt_cache_breakpoint"]
    if target == "chat":
        assert parts[0]["type"] == "file" and parts[0]["file"] == file
    else:
        assert (
            parts[0]["type"] == "input_file"
            and parts[0]["file_data"] == PDF
            and parts[0]["filename"] == "Report.PDF"
        )
    assert [entry["text"] for entry in parts[1:]] == (
        ["What is on page one?"] if with_text else []
    )


@pytest.mark.parametrize("custom", [False, True])
def test_tool_output_files_reach_chat_after_the_parallel_tool_results(
    custom: bool,
) -> None:
    call: JsonObject = (
        {
            "type": "custom_tool_call",
            "name": "inspect_pdf",
            "call_id": "a",
            "input": "report",
        }
        if custom
        else {
            "type": "function_call",
            "name": "inspect_pdf",
            "call_id": "a",
            "arguments": "{}",
        }
    )
    file: JsonObject = {"file_data": "JVBERi0xLjQK", "filename": "Report.PDF"}
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Get both items"},
            call,
            {
                "type": "function_call",
                "name": "count",
                "call_id": "b",
                "arguments": "{}",
            },
            {
                "type": "custom_tool_call_output" if custom else "function_call_output",
                "call_id": "a",
                "output": [
                    {"type": "input_text", "text": "Report follows"},
                    {"type": "input_file", **file},
                ],
            },
            {"type": "function_call_output", "call_id": "b", "output": "1"},
            {"role": "user", "content": "Read the attachment"},
        ],
    )
    sent = sent_request("chat", path, body)
    messages = object_array(sent["messages"])
    results = [
        index for index, message in enumerate(messages) if message["role"] == "tool"
    ]
    assert [messages[index]["tool_call_id"] for index in results] == ["a", "b"]
    attached = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "file" for part in object_array(message["content"]))
    )
    assert max(results) < attached < len(messages) - 1
    assert messages[attached]["role"] == "user"
    label, caption, document = object_array(messages[attached]["content"])
    assert '"a"' in str(label["text"])
    assert (
        caption["text"] == "Report follows" and object_payload(document["file"]) == file
    )


@pytest.mark.parametrize(
    "part",
    [
        {"file_url": "https://example.invalid/sample.pdf"},
        {"file_data": "data:text/plain;base64,YQ==", "filename": "note.txt"},
        {"file_data": PDF, "detail": "low"},
    ],
)
def test_chat_reports_unsupported_responses_file_inputs(part: JsonObject) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", **part},
                    {"type": "input_text", "text": "Summarize this"},
                ],
            }
        ],
    )
    assert "file" in translation_error("chat", path, body).lower()
