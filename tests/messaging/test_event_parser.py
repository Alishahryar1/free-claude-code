from free_claude_code.messaging.event_parser import parse_cli_event


def test_parse_cli_event_assistant_content():
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "Internal thought"},
                {"type": "text", "text": "Hello user"},
            ]
        },
    }
    results = parse_cli_event(event)
    assert len(results) == 2
    assert results[0] == {"type": "thinking_chunk", "text": "Internal thought"}
    assert results[1] == {"type": "text_chunk", "text": "Hello user"}


def test_parse_cli_event_assistant_tools():
    event = {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": "ls", "input": {"path": "."}}]
        },
    }
    results = parse_cli_event(event)
    assert len(results) == 1
    assert results[0]["type"] == "tool_use"
    assert results[0]["name"] == "ls"
    assert results[0]["input"] == {"path": "."}


def test_parse_cli_event_assistant_subagent():
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "Task",
                    "input": {"description": "Fix bug"},
                }
            ]
        },
    }
    results = parse_cli_event(event)
    assert len(results) == 1
    assert results[0]["type"] == "tool_use"
    assert results[0]["name"] == "Task"
    assert results[0]["input"] == {"description": "Fix bug"}


def test_parse_cli_event_content_block_delta():
    # Text delta
    event_text = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " more"},
    }
    results_text = parse_cli_event(event_text)
    assert results_text == [{"type": "text_delta", "index": 0, "text": " more"}]

    # Thinking delta
    event_think = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "thinking_delta", "thinking": " more thought"},
    }
    results_think = parse_cli_event(event_think)
    assert results_think == [
        {"type": "thinking_delta", "index": 1, "text": " more thought"}
    ]


def test_parse_cli_event_content_block_start():
    event = {
        "type": "content_block_start",
        "index": 2,
        "content_block": {
            "type": "tool_use",
            "name": "Task",
            "input": {"description": "deploy"},
        },
    }
    results = parse_cli_event(event)
    assert results == [
        {
            "type": "tool_use_start",
            "index": 2,
            "id": "",
            "name": "Task",
            "input": {"description": "deploy"},
        }
    ]


def test_parse_cli_event_error():
    event = {"type": "error", "error": {"message": "something failed"}}
    results = parse_cli_event(event)
    assert results == [{"type": "error", "message": "something failed"}]


def test_parse_cli_event_user_tool_result():
    event = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "ok",
                    "is_error": False,
                }
            ]
        },
    }
    results = parse_cli_event(event)
    assert results == [
        {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "ok",
            "is_error": False,
        }
    ]


def test_parse_cli_event_exit_success():
    event = {"type": "exit", "code": 0}
    results = parse_cli_event(event)
    assert results == [{"type": "complete", "status": "success"}]


def test_parse_cli_event_exit_failure():
    event = {"type": "exit", "code": 1, "stderr": "fatal error"}
    results = parse_cli_event(event)
    assert results == [
        {
            "type": "error",
            "message": "fatal error",
            "source": "exit",
            "exit_code": 1,
        }
    ]


def test_parse_cli_event_invalid_input():
    assert parse_cli_event(None) == []
    assert parse_cli_event("not a dict") == []
    assert parse_cli_event({"type": "unknown"}) == []


def test_parse_cli_event_system_ignored():
    assert parse_cli_event({"type": "system", "foo": "bar"}) == []


def test_parse_cli_event_result_with_content_directly():
    event = {"type": "result", "content": [{"type": "text", "text": "hi"}]}
    assert parse_cli_event(event) == [{"type": "text_chunk", "text": "hi"}]


def test_parse_cli_event_result_with_result_content_directly():
    event = {"type": "result", "result": {"content": [{"type": "text", "text": "hi"}]}}
    assert parse_cli_event(event) == [{"type": "text_chunk", "text": "hi"}]


def test_parse_cli_event_content_block_unknown_type_skipped():
    """Content block with unknown type is skipped; known blocks still parsed."""
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "visible"},
                {"type": "unknown", "data": "ignored"},
                {"type": "thinking", "thinking": "thought"},
            ]
        },
    }
    results = parse_cli_event(event)
    assert len(results) == 2
    assert results[0] == {"type": "text_chunk", "text": "visible"}
    assert results[1] == {"type": "thinking_chunk", "text": "thought"}


def test_parse_cli_event_error_non_dict():
    """Error event with error as string (not dict) is handled."""
    event = {"type": "error", "error": "plain string error"}
    results = parse_cli_event(event)
    assert results == [{"type": "error", "message": "plain string error"}]


def test_parse_cli_event_exit_code_none():
    """Exit event with no code defaults to success."""
    event = {"type": "exit"}
    results = parse_cli_event(event)
    assert results == [{"type": "complete", "status": "success"}]
