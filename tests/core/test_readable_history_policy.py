"""Readable history remains reasoning until the destination accepts a scope."""

from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.history_replay import (
    HistoryMode,
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    prepare_history,
)


def origin(protocol="chat"):
    return ReplayOrigin(
        "provider", protocol, "https://provider.test/v1", "key", "model"
    )


def history():
    return {
        "messages": [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "First answer",
                "reasoning_content": "Old",
            },
            {"role": "assistant", "content": "No thoughts"},
            {"role": "user", "content": "Next"},
            {"role": "assistant", "content": "Next answer", "reasoning_content": "New"},
            {"role": "assistant", "content": "Also no thoughts"},
        ],
    }


def test_default_all_retains_old_reasoning_and_adds_empty_assistant_fields():
    body = history()
    original = deepcopy(body)
    result = cast(dict[str, Any], prepare_history(body, origin()))
    assistants = [m for m in result["messages"] if m["role"] == "assistant"]
    assert [m.get("reasoning_content") for m in assistants] == ["Old", "", "New", ""]
    assert [m["content"] for m in assistants] == [
        "First answer",
        "No thoughts",
        "Next answer",
        "Also no thoughts",
    ]
    assert body == original


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_empty_placeholder_uses_destination_native_field(field):
    result = prepare_history(
        {"messages": [{"role": "assistant", "content": "Answer"}]},
        origin(),
        reasoning_field=field,
    )
    assert result["messages"] == [{"role": "assistant", "content": "Answer", field: ""}]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("explicit_empty", [False, True])
@pytest.mark.parametrize("mode", list(HistoryMode))
def test_discarded_protected_history_does_not_remove_included_empty_slot(
    field, explicit_empty, mode
):
    carrier = encode_replay(
        ReplayRecord(
            origin("responses"),
            {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
        )
    )
    assistant = {
        "role": "assistant",
        "content": "Answer",
        "reasoning_details": [{"type": "reasoning.encrypted", "data": carrier}],
        **({field: ""} if explicit_empty else {}),
    }
    source = {"messages": [{"role": "user", "content": "Question"}, assistant]}
    original = deepcopy(source)
    result = prepare_history(
        source,
        origin(),
        scope=mode,
        reasoning_field=field,
        structured_details=False,
    )
    assert result["messages"] == [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "Answer",
            **({field: ""} if mode is not HistoryMode.NONE else {}),
        },
    ]
    assert source == original


@pytest.mark.parametrize("explicit_empty", [False, True])
@pytest.mark.parametrize("mode", list(HistoryMode))
def test_retained_native_details_preserve_explicit_empty_without_inventing_it(
    explicit_empty, mode
):
    details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    source = {
        "messages": [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_details": details,
                **({"reasoning_content": ""} if explicit_empty else {}),
            }
        ]
    }
    original = deepcopy(source)
    result = prepare_history(source, origin(), scope=mode)
    assert result["messages"] == [
        {
            "role": "assistant",
            "content": "Answer",
            "reasoning_details": details,
            **(
                {"reasoning_content": ""}
                if explicit_empty and mode is not HistoryMode.NONE
                else {}
            ),
        }
    ]
    assert source == original


def test_user_and_tool_result_messages_never_get_placeholders():
    body = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "tool", "content": "Result", "tool_call_id": "call"},
        ]
    }
    assert prepare_history(body, origin()) == body


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("all", ["Old", "", "New", ""]),
        ("current", [None, None, "New", ""]),
        ("none", [None, None, None, None]),
    ],
)
def test_three_scope_definitions_preserve_source(mode, expected):
    from free_claude_code.core.history_replay import HistoryMode

    body = history()
    original = deepcopy(body)
    result = cast(
        dict[str, Any], prepare_history(body, origin(), scope=HistoryMode(mode))
    )
    assert [
        m.get("reasoning_content")
        for m in result["messages"]
        if m["role"] == "assistant"
    ] == expected
    assert body == original


@pytest.mark.parametrize("mixed", [False, True])
def test_messages_tool_result_envelopes_only_start_user_turn_when_mixed(mixed):
    from free_claude_code.core.history_replay import HistoryMode

    content = [
        {
            "type": "tool_result",
            "tool_use_id": "call",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.test/image"},
                }
            ],
        }
    ]
    if mixed:
        content.append({"type": "text", "text": "New request"})
    body = {
        "messages": [
            {"role": "user", "content": "Begin"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Plan"},
                    {"type": "tool_use", "id": "call", "name": "look", "input": {}},
                ],
            },
            {"role": "user", "content": content},
        ]
    }
    current = prepare_history(body, origin("messages"), scope=HistoryMode.CURRENT)
    assert ("Plan" in str(current)) is not mixed
    assert "https://example.test/image" in str(current)


def test_current_without_user_includes_existing_assistant():
    from free_claude_code.core.history_replay import HistoryMode

    body = {"messages": [{"role": "assistant", "content": "Answer"}]}
    assert prepare_history(
        body, origin(), scope=HistoryMode.CURRENT
    ) == prepare_history(body, origin(), scope=HistoryMode.ALL)


@pytest.mark.parametrize("mode", ["all", "current", "none"])
def test_native_records_stay_exact_in_all_modes(mode):
    from free_claude_code.core.history_replay import HistoryMode

    native = {
        "type": "thinking",
        "thinking": "Signed text",
        "signature": "real-signature",
    }
    source = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    native,
                    {"type": "thinking", "thinking": "Editable"},
                    {"type": "text", "text": "Answer"},
                ],
            },
            {"role": "user", "content": "Next"},
        ]
    }
    projected = cast(
        dict[str, Any],
        prepare_history(source, origin("messages"), scope=HistoryMode(mode)),
    )
    assert projected["messages"][0]["content"][0] == native
    assert ("Editable" in str(projected)) == (mode == "all")


def test_unsigned_response_summary_is_messages_thinking_without_prose():
    from free_claude_code.core.history_replay import responses_reasoning_blocks

    assert responses_reasoning_blocks(
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Summary"}],
        }
    ) == [{"type": "thinking", "thinking": "Summary"}]


def test_native_messages_readable_foreign_carrier_reaches_shared_projection():
    from free_claude_code.core.anthropic.passthrough import restore_native_history
    from free_claude_code.core.history_replay import (
        HistoryMode,
        ReplayRecord,
        encode_replay,
    )

    carrier = encode_replay(
        ReplayRecord(origin("chat"), {"reasoning_content": "Portable thought"})
    )
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": carrier},
                    {"type": "text", "text": "Answer"},
                ],
            }
        ]
    }
    restored = restore_native_history(body, origin("messages"))
    projected = prepare_history(restored, origin("messages"), scope=HistoryMode.ALL)
    assert "Portable thought" in str(projected)
    none = prepare_history(restored, origin("messages"), scope=HistoryMode.NONE)
    assert "Portable thought" not in str(none)
    assert "Answer" in str(none)
