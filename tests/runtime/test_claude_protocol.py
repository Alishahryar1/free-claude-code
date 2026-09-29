from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    UserMessage,
)

from free_claude_code.runtime.claude_protocol import ClaudePrompt, ClaudeProtocol


def test_native_per_block_messages_keep_live_and_history_indices():
    from claude_agent_sdk.types import SessionMessage

    protocol = ClaudeProtocol("generation", "session")
    protocol.begin("run")
    protocol.feed(UserMessage("hi", uuid="run", origin={"kind": "human"}))
    protocol.feed(
        StreamEvent(
            "start", "session", {"type": "message_start", "message": {"id": "msg"}}
        )
    )
    finals = []
    history = [SessionMessage("user", "run", "session", {"content": "hi"})]
    for index, text in enumerate(("First block.", "Second block.")):
        protocol.feed(
            StreamEvent(
                "event",
                "session",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        complete = AssistantMessage(
            [TextBlock(text)], "model", message_id="msg", uuid=f"uuid-{index}"
        )
        finals.extend(event.item for event in protocol.feed(complete) if event.item)
        history.append(
            SessionMessage(
                "assistant",
                f"uuid-{index}",
                "session",
                {"id": "msg", "content": [{"type": "text", "text": text}]},
            )
        )
    assert [item.item_id for item in finals] == ["msg:0", "msg:1"]
    recovered = ClaudeProtocol.history("session", history, frozenset({"run"}))
    assert [
        (item.item_id, item.text)
        for item in recovered.turns[0].items
        if item.kind == "text"
    ] == [(item.item_id, item.text) for item in finals]


def test_missing_history_anchor_reports_recovery_limit():
    recovered = ClaudeProtocol.history("session", [], frozenset({"submitted"}))
    assert recovered.turns == ()
    assert recovered.history_notice
    assert ClaudeProtocol.history("session", [], frozenset()).history_notice is None


def test_partial_and_complete_text_share_identity():
    protocol = ClaudeProtocol("generation", "session")
    protocol.begin("run")
    protocol.feed(UserMessage("hi", uuid="run", origin={"kind": "human"}))

    def stream(event):
        return protocol.feed(StreamEvent("event", "session", event))

    stream({"type": "message_start", "message": {"id": "msg"}})
    stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    partial = stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
    )
    final = protocol.feed(
        AssistantMessage(
            [TextBlock("Hello world")], "model", message_id="msg", uuid="uuid"
        )
    )
    assert partial[0].item is not None
    assert final[0].item is not None
    assert partial[0].item.item_id == final[0].item.item_id
    assert partial[0].item.text == "Hello"
    assert final[0].item.text == "Hello world"
    assert final[0].item.complete


def test_background_result_does_not_finish_human_turn():
    protocol = ClaudeProtocol("g", "s")
    protocol.begin("run")
    protocol.feed(UserMessage("hi", uuid="run", origin={"kind": "human"}))
    result = ResultMessage(
        "success", 1, 1, False, 1, "s", origin={"kind": "task-notification"}
    )
    assert not any(e.kind == "turn_completed" for e in protocol.feed(result))
    result.origin = {"kind": "human"}
    assert protocol.feed(result)[-1].turn_id == "run"


def test_unknown_subagent_output_cannot_be_attached_to_current_human():
    protocol = ClaudeProtocol("g", "s")
    protocol.begin("new")
    protocol.feed(UserMessage("hi", uuid="new", origin={"kind": "human"}))
    events = protocol.feed(
        AssistantMessage(
            [TextBlock("old child")],
            "model",
            parent_tool_use_id="unknown-old-tool",
            message_id="old",
            uuid="old",
        )
    )
    assert not any(event.kind == "item" for event in events)


def test_question_answers_preserve_input_and_validate_selection():
    prompt = ClaudePrompt(
        "tool",
        "run",
        "AskUserQuestion",
        {
            "questions": [
                {
                    "question": "Pick",
                    "header": "Choice",
                    "multiSelect": True,
                    "options": [{"label": "A"}, {"label": "B"}],
                }
            ]
        },
    )
    questions = prompt.request().form["questions"]
    assert isinstance(questions, list)
    assert isinstance(questions[0], dict)
    assert questions[0]["multiple"] is True
    response = prompt.answer({"answers": {"0": ["A", "B"]}})
    updated = response["updated_input"]
    assert isinstance(updated, dict)
    assert updated["answers"] == {"Pick": "A, B"}
    assert updated["questions"] == prompt.input["questions"]
