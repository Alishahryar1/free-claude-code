"""Verify which parts of a request or reply can cross a provider boundary."""

import json
from unittest.mock import patch

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject, JsonValue, load_json_value
from free_claude_code.providers.deepseek import DeepSeekProvider
from tests.api.support import create_test_app
from tests.api.test_four_protocol_matrix import (
    ingress_request,
    matrix_app,
    matrix_reply,
    sent_request,
    translation_error,
)
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_gemini import _sse
from tests.wire_payload import decode_object, object_array, object_payload


@pytest.mark.parametrize(
    ("wire", "target"),
    [
        (wire, target)
        for wire in ("chat", "messages", "responses", "gemini")
        for target in ("chat", "messages", "responses", "gemini")
        if wire != target
    ],
)
@pytest.mark.parametrize("choice", ["named", "none"])
def test_foreign_reply_cannot_call_a_function_the_caller_excluded(
    wire: str, target: str, choice: str
) -> None:
    path, body = ingress_request(wire, wire == "responses")
    if wire == "gemini":
        body["tools"] = [
            {"functionDeclarations": [{"name": "read"}, {"name": "allowed"}]}
        ]
        config: JsonObject = (
            {"mode": "NONE"}
            if choice == "none"
            else {"mode": "ANY", "allowedFunctionNames": ["allowed"]}
        )
        body["toolConfig"] = {"functionCallingConfig": config}
    elif wire == "messages":
        body["tools"] = [
            {"name": name, "input_schema": {"type": "object"}}
            for name in ("read", "allowed")
        ]
        body["tool_choice"] = (
            {"type": "none"}
            if choice == "none"
            else {"type": "tool", "name": "allowed"}
        )
    else:

        def declaration(name: str) -> JsonObject:
            return (
                {"type": "function", "function": {"name": name}}
                if wire == "chat"
                else {"type": "function", "name": name}
            )

        body["tools"] = [declaration("read"), declaration("allowed")]
        body["tool_choice"] = "none" if choice == "none" else declaration("allowed")
    _assert_read_is_rejected(wire, target, path, body)


def _assert_read_is_rejected(
    wire: str, target: str, path: str, body: JsonObject
) -> None:
    attempts: list[str] = []

    def wrong_tool(request: httpx.Request | httpx2.Request):
        attempts.append(request.url.path)
        return matrix_reply(target)

    with TestClient(
        matrix_app(
            target, native_reply=wrong_tool, sdk_reply=wrong_tool, max_attempts=1
        )
    ) as client:
        result = client.post(path, json=body)
    assert attempts
    if result.status_code == 200:
        assert wire == "responses"
        events = parse_sse_text(result.text)
        assert events[-1].event == "response.failed"
        assert not any(
            isinstance(item := event.data.get("item"), dict)
            and item.get("type")
            in ("function_call", "custom_tool_call", "tool_search_call")
            for event in events
        )
    else:
        assert result.status_code == 502


@pytest.mark.parametrize("target", ["chat", "messages", "responses", "gemini"])
def test_legacy_chat_disable_is_still_an_explicit_tool_choice(target: str) -> None:
    path, body = ingress_request("chat", False)
    body.pop("tools")
    body.update(
        functions=[{"name": "read", "parameters": {"type": "object"}}],
        function_call="none",
    )
    if target == "chat":
        native = sent_request(target, path, body)
        assert (
            native["function_call"] == "none"
            and native["functions"] == body["functions"]
        )
    else:
        _assert_read_is_rejected("chat", target, path, body)


@pytest.mark.parametrize("target", ["chat", "messages", "responses", "gemini"])
def test_vendor_chat_action_and_its_result_are_kept_together(target: str) -> None:
    path, body = ingress_request(
        "chat",
        False,
        [
            {"role": "user", "content": "Get the invoice amount"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "vendor_invoice",
                        "id": "v",
                        "vendor_invoice": {"invoice": "123"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "v",
                "content": "The verified invoice is exactly $470.",
            },
        ],
    )
    if target == "chat":
        assert sent_request(target, path, body)["messages"] == body["messages"]
    else:
        assert "vendor_invoice" in translation_error(target, path, body)


@pytest.mark.parametrize(
    ("original", "offered", "choice", "expected"),
    [
        ("old", "read", None, 200),
        ("read", "old", None, 502),
        (None, "old", None, 502),
        ("read", "read", "none", 502),
        (
            "read",
            "read",
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "function", "function": {"name": "old"}}],
                },
            },
            502,
        ),
    ],
)
def test_chat_provider_checks_the_tool_selection_on_the_wire(
    original: str | None, offered: str, choice: JsonValue, expected: int
) -> None:
    sent: list[JsonObject] = []

    def reply(request: httpx2.Request):
        sent.append(decode_object(request.content))
        response = matrix_reply("chat")
        assert isinstance(response, httpx2.Response)
        return response

    sdk = AsyncOpenAI(
        api_key="test",
        base_url="https://upstream.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", return_value=sdk
    ):
        provider = DeepSeekProvider(
            make_provider_config(
                api_key="test", base_url="https://upstream.invalid/v1"
            ),
            admission=immediate_admission(max_attempts=1),
        )
    path, body = ingress_request("messages", False)
    if original:
        body["tools"] = [{"name": original, "input_schema": {"type": "object"}}]
    else:
        body.pop("tools")
    offered_names = (offered, "old") if isinstance(choice, dict) else (offered,)
    extra: JsonObject = {
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in offered_names
        ]
    }
    if choice is not None:
        extra["tool_choice"] = choice
    body["extra_body"] = extra
    with TestClient(
        create_test_app(
            Settings(MODEL="deepseek/test"), providers={"deepseek": provider}
        )
    ) as client:
        response = client.post(path, json=body)
    assert len(sent) == 1 and sent[0]["tools"] == extra["tools"]
    if choice is not None:
        assert sent[0]["tool_choice"] == choice
    assert response.status_code == expected, response.text
    assert ('"tool_use"' in response.text) == (expected == 200)


@pytest.mark.parametrize("target", ["chat", "responses", "gemini"])
@pytest.mark.parametrize("use_local_search", [False, True])
def test_program_only_tool_cannot_be_returned_as_a_direct_call(
    target: str, use_local_search: bool
) -> None:
    path, body = ingress_request("messages", False)
    tools: list[JsonObject] = [
        {
            "name": "read",
            "input_schema": {"type": "object"},
            "allowed_callers": ["code_execution_20250825"],
        }
    ]
    if use_local_search:
        tools.append({"type": "web_search_20250305", "name": "web_search"})
    body["tools"] = tools
    _assert_read_is_rejected("messages", target, path, body)


@pytest.mark.parametrize("target", ["chat", "messages", "gemini"])
def test_past_custom_tool_cannot_authorize_a_new_call(target: str) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Read the file"},
            {
                "type": "custom_tool_call",
                "name": "read",
                "call_id": "old",
                "input": "file",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "old",
                "output": "I have read it.",
            },
            {"role": "user", "content": "What did the tool return?"},
        ],
    )
    body.pop("tools")
    _assert_read_is_rejected("responses", target, path, body)


@pytest.mark.parametrize("target", ["chat", "messages", "responses", "gemini"])
def test_cited_answer_reaches_the_next_turn_with_its_source(target: str) -> None:
    citation: JsonObject = {
        "type": "url_citation",
        "title": "2025 budget",
        "url": "https://example.invalid/report",
        "start_index": 8,
        "end_index": 14,
        "vendor_note": "private citation metadata",
    }
    path, body = ingress_request(
        "responses",
        True,
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "See the report.",
                        "annotations": [citation],
                    }
                ],
            },
            {"role": "user", "content": "What was the URL for the report?"},
        ],
    )
    sent = json.dumps(sent_request(target, path, body))
    for value in (
        "See the report.",
        "2025 budget",
        "https://example.invalid/report",
        "What was the URL",
    ):
        assert value in sent
    assert ("private citation metadata" in sent) == (target == "responses")
    citation.update(
        type="file_citation", file_id="file-001", filename="report.pdf", index=8
    )
    if target != "responses":
        assert "file_citation" in translation_error(target, path, body)


@pytest.mark.parametrize("target", ["chat", "messages", "responses", "gemini"])
@pytest.mark.parametrize("status", ["in_progress", "incomplete"])
def test_incomplete_responses_tool_result_never_becomes_a_completed_reply(
    target: str, status: str
) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Read both pages."},
            {
                "type": "function_call",
                "name": "read",
                "call_id": "r",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "r",
                "status": status,
                "output": "first page only",
            },
            {"role": "user", "content": "Summarize both pages."},
        ],
    )
    if target == "responses":
        turns = object_array(sent_request(target, path, body)["input"])
        assert (
            next(
                turn["status"]
                for turn in turns
                if turn.get("type") == "function_call_output"
            )
            == status
        )
    else:
        assert status in translation_error(target, path, body)


@pytest.mark.parametrize("target", ["chat", "messages", "responses", "gemini"])
def test_metadata_on_a_noop_edit_and_a_past_tool_call_is_optional(target: str) -> None:
    path, body = ingress_request(
        "messages",
        False,
        [
            {"role": "user", "content": "Read"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "r",
                        "name": "read",
                        "input": {},
                        "extra_content": {"vendor_note": "call trace"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "r",
                        "content": "Public result",
                    }
                ],
            },
        ],
    )
    body["context_management"] = {
        "edits": [
            {
                "type": "clear_thinking_20251015",
                "keep": "all",
                "vendor_option": "edit trace",
            }
        ]
    }
    sent = json.dumps(sent_request(target, path, body))
    assert "Public result" in sent
    assert ("edit trace" in sent) == (target == "messages")
    assert ("call trace" in sent) == (target == "messages")


@pytest.mark.parametrize(
    ("wire", "target"), [("chat", "responses"), ("responses", "chat")]
)
@pytest.mark.parametrize("role", ["user", "developer", "assistant"])
def test_openai_cache_policy_and_exact_text_breakpoint_cross_together(
    wire: str, target: str, role: str
) -> None:
    marker: JsonObject = {"mode": "explicit", "ttl": "30m"}
    kind = "text" if wire == "chat" else "input_text"
    path, body = ingress_request(
        wire,
        True,
        [
            {
                "role": role,
                "content": [
                    {
                        "type": kind,
                        "text": "Reusable prefix",
                        "prompt_cache_breakpoint": marker,
                    },
                    {"type": kind, "text": "After the boundary"},
                ],
            },
            {"role": "user", "content": "Continue"},
        ],
    )
    options: JsonObject = {"mode": "explicit", "ttl": "30m"}
    body.update(
        prompt_cache_key="cache-group",
        prompt_cache_options=options,
        prompt_cache_retention="in_memory",
        safety_identifier="stable-user",
    )
    sent = sent_request(target, path, body)
    for key in (
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "safety_identifier",
    ):
        assert sent[key] == body[key]
    messages = object_array(sent["messages" if target == "chat" else "input"])
    first = object_array(
        next(message["content"] for message in messages if message.get("role") == role)
    )
    assert (
        first[0]["text"] == "Reusable prefix"
        and first[0]["prompt_cache_breakpoint"] == marker
    )
    assert (
        first[1]["text"] == "After the boundary"
        and "prompt_cache_breakpoint" not in first[1]
    )
    if target == "responses":
        assert [part["type"] for part in first] == ["input_text", "input_text"]


def test_server_only_search_choice_cannot_enable_the_client_executor() -> None:
    path, body = ingress_request("responses", True)
    body["tools"] = [
        {
            "type": "tool_search",
            "execution": "client",
            "parameters": {"type": "object"},
        },
        {"type": "tool_search", "execution": "server"},
    ]
    allowed: JsonObject = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "tool_search", "execution": "server"}],
    }
    body["tool_choice"] = allowed
    sent = sent_request("chat", path, body)
    assert "fcc_tool_search" not in json.dumps(sent.get("tools"))
    allowed["mode"] = "required"
    assert translation_error("chat", path, body)
    allowed["tools"] = [{"type": "tool_search", "execution": "client"}]
    assert "fcc_tool_search" in json.dumps(sent_request("chat", path, body)["tools"])


@pytest.mark.parametrize(
    ("wire", "target"), [("chat", "responses"), ("responses", "chat")]
)
def test_openai_breakpoints_stay_on_the_image_and_tool_result(
    wire: str, target: str
) -> None:
    marker: JsonObject = {"mode": "explicit", "ttl": "12h"}
    uri = "data:image/png;base64,iVBORw0KGgo="
    if wire == "chat":
        image: JsonObject = {"type": "image_url", "image_url": {"url": uri}}
        assistant: JsonObject = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "r",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        }
        result: JsonObject = {"role": "tool", "tool_call_id": "r"}
        result["content"] = [
            {"type": "text", "text": "found", "prompt_cache_breakpoint": marker}
        ]
    else:
        image = {"type": "input_image", "image_url": uri}
        assistant = {
            "type": "function_call",
            "call_id": "r",
            "name": "read",
            "arguments": "{}",
        }
        result = {
            "type": "function_call_output",
            "call_id": "r",
            "output": [
                {
                    "type": "input_text",
                    "text": "found",
                    "prompt_cache_breakpoint": marker,
                }
            ],
        }
    image["prompt_cache_breakpoint"] = {**marker, "vendor_note": "private"}
    path, body = ingress_request(
        wire, True, [{"role": "user", "content": [image]}, assistant, result]
    )
    body["prompt_cache_options"] = {
        "mode": "explicit",
        "ttl": "12h" if wire == "responses" else "30m",
        "vendor_note": "private",
    }
    sent = sent_request(target, path, body)
    assert sent["prompt_cache_options"] == (
        {"mode": "explicit"} if target == "chat" else {"mode": "explicit", "ttl": "30m"}
    )
    turns = object_array(sent["messages" if target == "chat" else "input"])
    picture = object_array(turns[0]["content"])[0]
    output = object_array(turns[-1]["content" if target == "chat" else "output"])[0]
    assert (
        picture["prompt_cache_breakpoint"]
        == output["prompt_cache_breakpoint"]
        == marker
    )
    assert output["text"] == "found"


@pytest.mark.parametrize("target", ["gemini", "chat", "messages", "responses"])
def test_gemini_resume_token_requires_the_original_protocol(target: str) -> None:
    path, body = ingress_request("gemini", False)
    body["continuationToken"] = "saved-turn"
    if target == "gemini":
        assert sent_request(target, path, body)["continuationToken"] == "saved-turn"
    else:
        assert "continuationToken" in translation_error(target, path, body)


@pytest.mark.parametrize("target", ["gemini", "chat", "messages", "responses"])
@pytest.mark.parametrize("kind", ["function", "custom"])
def test_responses_async_functions_require_a_capable_route(
    target: str, kind: str
) -> None:
    path, body = ingress_request("responses", True)
    asynchronous: JsonObject = {
        "type": kind,
        "name": "background_lookup",
        "async": True,
    }
    body["tools"] = [asynchronous, {"type": "function", "name": "read"}]
    optional = sent_request(target, path, body)
    assert ("background_lookup" in json.dumps(optional["tools"])) == (
        target == "responses"
    )
    assert "read" in json.dumps(optional["tools"])
    body["tool_choice"] = {"type": kind, "name": "background_lookup"}
    if target == "responses":
        assert sent_request(target, path, body)["tool_choice"] == body["tool_choice"]
    else:
        assert translation_error(target, path, body)


@pytest.mark.parametrize("target", ["gemini", "chat", "messages", "responses"])
@pytest.mark.parametrize(
    "part",
    [
        {"fileData": {"fileUri": "files/manual", "mimeType": "application/pdf"}},
        {"executableCode": {"language": "PYTHON", "code": "value = 1"}},
        {"toolResponse": {"id": "s", "response": {"value": 1}}},
    ],
)
def test_gemini_system_instruction_only_accepts_known_text(
    target: str, part: JsonObject
) -> None:
    path, body = ingress_request("gemini", False)
    body["systemInstruction"] = {"parts": [{"text": "Use this reference"}, part]}
    assert "systemInstruction" in translation_error(target, path, body)


@pytest.mark.parametrize("target", ["gemini", "chat", "messages"])
@pytest.mark.parametrize(
    ("call", "result_kind"),
    [
        (
            {"type": "custom_tool_call", "name": "lookup", "call_id": "r"},
            "custom_tool_call_output",
        ),
        (
            {
                "type": "function_call",
                "name": "lookup",
                "call_id": "r",
                "arguments": "{}",
            },
            "custom_tool_call_output",
        ),
        (
            {"type": "custom_tool_call", "name": "lookup", "call_id": "r", "input": ""},
            "function_call_output",
        ),
    ],
)
def test_translation_checks_call_and_result_before_rewriting_their_types(
    target: str, call: JsonObject, result_kind: str
) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Run the lookup"},
            call,
            {"type": result_kind, "call_id": "r", "output": "result"},
        ],
    )
    body.pop("tools")
    assert "tool" in translation_error(target, path, body).lower()


@pytest.mark.parametrize("target", ["gemini", "chat", "messages"])
def test_empty_custom_input_is_a_valid_call(target: str) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Run lookup"},
            {"type": "custom_tool_call", "name": "lookup", "call_id": "r", "input": ""},
            {
                "type": "custom_tool_call_output",
                "call_id": "r",
                "output": "default entry",
            },
        ],
    )
    body["tools"] = [{"type": "custom", "name": "lookup"}]
    sent = sent_request(target, path, body)
    turns = object_array(sent["contents" if target == "gemini" else "messages"])
    assistant = turns[1]
    if target == "gemini":
        call = object_payload(object_array(assistant["parts"])[0]["functionCall"])
        args = call["args"]
    elif target == "messages":
        args = object_array(assistant["content"])[0]["input"]
    else:
        call = object_payload(object_array(assistant["tool_calls"])[0]["function"])
        raw_args = call["arguments"]
        assert isinstance(raw_args, str)
        args = load_json_value(raw_args)
    assert args == {"input": ""}
    assert "default entry" in json.dumps(turns[-1])


def test_client_search_cannot_complete_an_ordinary_function_call() -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Read"},
            {
                "type": "function_call",
                "call_id": "r",
                "name": "read",
                "arguments": "{}",
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "r",
                "tools": [],
            },
        ],
    )
    assert "tool" in translation_error("chat", path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
@pytest.mark.parametrize("via_tool", [False, True])
def test_messages_search_results_preserve_public_evidence(
    protocol: str, via_tool: bool
) -> None:
    hit: JsonObject = {
        "type": "search_result",
        "source": "https://example.invalid/report",
        "title": "Seat report",
        "content": [
            {
                "type": "text",
                "text": "Available seats: 14",
                "vendor_note": "private-note",
            }
        ],
        "extra_data": "private-note",
    }
    history: list[JsonObject] = []
    if via_tool:
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "lookup",
                            "name": "read",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "lookup",
                            "content": [hit],
                        }
                    ],
                },
            ]
        )
    else:
        history.append({"role": "user", "content": [hit]})
    history.append({"role": "user", "content": "How many seats are available?"})
    path, body = ingress_request("messages", True, history)
    actual = json.dumps(sent_request(protocol, path, body))
    for value in (
        "Seat report",
        "https://example.invalid/report",
        "Available seats: 14",
        "How many seats",
    ):
        assert value in actual
    assert ("private-note" in actual) == (protocol == "messages")
    hit["citations"] = {"enabled": True}
    if protocol != "messages":
        assert "citations" in translation_error(protocol, path, body)


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
def test_deferred_messages_web_search_waits_for_discovery(protocol: str) -> None:
    path, body = ingress_request("messages", True)
    hosted: JsonObject = {
        "name": "web_search",
        "type": "web_search_20250305",
        "defer_loading": True,
    }
    body["tools"] = [
        hosted,
        {"name": "tool_search_tool_regex", "type": "tool_search_tool_regex_20251119"},
        {"name": "read", "input_schema": {"type": "object"}},
    ]
    offered = sent_request(protocol, path, body)["tools"]
    assert "fcc_web_search" not in json.dumps(offered)
    assert "read" in json.dumps(offered)
    if protocol == "messages":
        assert isinstance(offered, list) and hosted in offered


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini"])
def test_flattened_tool_keeps_its_namespace_instructions(protocol: str) -> None:
    history: list[JsonObject] = [
        {"role": "user", "content": "Find billing tools"},
        {
            "type": "tool_search_call",
            "execution": "client",
            "call_id": "search",
            "arguments": {},
        },
        {
            "type": "tool_search_output",
            "execution": "client",
            "call_id": "search",
            "tools": [
                {
                    "type": "namespace",
                    "name": "billing",
                    "description": "Only read monthly statements.",
                    "tools": [{"type": "function", "name": "statement"}],
                }
            ],
        },
        {"role": "user", "content": "What tools were found?"},
    ]
    path, body = ingress_request("responses", True, history)
    body["tools"] = [
        {
            "type": "namespace",
            "name": "billing",
            "description": "Ask the owner before reading an account.",
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "description": "Read today's balance",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "custom",
                    "name": "edit",
                    "description": "Edit a balance",
                    "format": {"type": "text"},
                },
            ],
        }
    ]
    sent = sent_request(protocol, path, body)
    tools = json.dumps(sent["tools"])
    assert tools.count("Ask the owner before reading an account.") == 3
    assert "Read today's balance" in tools and "Edit a balance" in tools
    transcript = json.dumps(sent.get("messages", sent.get("contents")))
    assert "Only read monthly statements." in transcript
    assert "Ask the owner before reading an account." not in transcript


@pytest.mark.parametrize("role", ["assistant", "developer", "system"])
@pytest.mark.parametrize("protocol", ["chat", "messages"])
def test_reference_image_in_a_text_only_role_cannot_disappear(
    role: str, protocol: str
) -> None:
    path, body = ingress_request(
        "responses",
        True,
        [
            {
                "role": role,
                "content": [
                    {"type": "input_text", "text": "Reference"},
                    {"type": "input_image", "image_url": "data:image/png;base64,YQ=="},
                ],
            },
            {"role": "user", "content": "What does the reference show?"},
        ],
    )
    assert "image" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
def test_forced_messages_tool_survives_an_unavailable_sibling(protocol: str) -> None:
    path, body = ingress_request("messages", True)
    schema: JsonObject = {"type": "object", "properties": {}}
    body["tools"] = [
        {"name": "read", "input_schema": schema},
        {"name": "catalogued_lookup", "input_schema": schema, "defer_loading": True},
        {"name": "strict_lookup", "input_schema": schema, "strict": True},
    ]
    body["tool_choice"] = {"type": "tool", "name": "read"}
    observed = sent_request(protocol, path, body)
    declarations = json.dumps(observed["tools"])
    assert "read" in declarations
    assert ("catalogued_lookup" in declarations) == (protocol == "messages")
    assert ("strict_lookup" in declarations) == (protocol != "gemini")
    body["tool_choice"] = {"type": "tool", "name": "catalogued_lookup"}
    if protocol != "messages":
        assert "available" in translation_error(protocol, path, body)


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
def test_chat_audio_input_keeps_its_recording_or_fails_before_inference(
    protocol: str,
) -> None:
    path, body = ingress_request(
        "chat",
        True,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "wav", "data": "UklGRg=="},
                    },
                    {"type": "text", "text": "Transcribe this clip"},
                ],
            }
        ],
    )
    if protocol == "chat":
        assert "UklGRg==" in json.dumps(sent_request(protocol, path, body))
    else:
        assert "audio" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
@pytest.mark.parametrize("in_tool_result", [False, True])
def test_messages_document_is_never_replaced_by_its_caption(
    protocol: str, in_tool_result: bool
) -> None:
    content: list[JsonObject] = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQK",
            },
        },
        {"type": "text", "text": "Read the attached report"},
    ]
    history: list[JsonObject] = []
    if in_tool_result:
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "read", "id": "r", "input": {}}
                ],
            }
        )
        content = [{"type": "tool_result", "tool_use_id": "r", "content": content}]
    history.append({"role": "user", "content": content})
    path, body = ingress_request("messages", True, history)
    if protocol == "messages":
        captured = json.dumps(sent_request(protocol, path, body))
        assert "JVBERi0xLjQK" in captured and "Read the attached report" in captured
    else:
        assert "document" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
def test_cache_prewarming_cannot_turn_into_inference(protocol: str) -> None:
    path, body = ingress_request("responses", True)
    body["prompt_cache_options"] = {"prewarm": True}
    if protocol == "responses":
        assert sent_request(protocol, path, body)["prompt_cache_options"] == {
            "prewarm": True
        }
    else:
        assert "prewarm" in translation_error(protocol, path, body)
        body["prompt_cache_options"] = {"prewarm": False}
        assert "hello" in json.dumps(sent_request(protocol, path, body))


@pytest.mark.parametrize(
    ("wire", "field", "control"),
    [
        ("responses", "background", True),
        (
            "responses",
            "moderation",
            {"model": "omni-moderation-latest", "policy": {"input": {"mode": "block"}}},
        ),
        (
            "chat",
            "moderation",
            {
                "model": "omni-moderation-latest",
                "policy": {"output": {"mode": "score"}},
            },
        ),
    ],
)
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages", "gemini"])
def test_request_execution_controls_require_a_capable_route(
    wire: str, field: str, control: JsonValue, protocol: str
) -> None:
    path, body = ingress_request(wire, True)
    body[field] = control
    if wire == protocol:
        assert sent_request(protocol, path, body)[field] == control
    else:
        assert field in translation_error(protocol, path, body)


def test_chat_classifies_a_deferred_hosted_result_as_a_client_error() -> None:
    path, body = ingress_request(
        "messages",
        True,
        [
            {"role": "user", "content": "Read both sources"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "read", "id": "r", "input": {}},
                    {
                        "type": "server_tool_use",
                        "name": "web_search",
                        "id": "s",
                        "input": {"query": "sources"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "s",
                        "content": [{"type": "future_search_result"}],
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "r",
                        "content": "Read completed",
                    }
                ],
            },
        ],
    )
    assert "web search result" in translation_error("chat", path, body)


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
