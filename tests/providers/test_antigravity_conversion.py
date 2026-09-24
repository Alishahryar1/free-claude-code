"""OpenAI-chat/Gemini conversion contracts for Antigravity."""

from types import SimpleNamespace

from free_claude_code.providers.antigravity.conversion import (
    StreamState,
    final_chunk,
    gemini_event_chunks,
    openai_chat_to_cloudcode,
)


def test_chat_body_converts_system_tools_and_generation_config() -> None:
    envelope = openai_chat_to_cloudcode(
        {
            "model": "gemini-test",
            "messages": [
                {"role": "system", "content": "Be useful."},
                {"role": "developer", "content": "Follow project rules."},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look something up",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"},
            },
            "temperature": 0.25,
            "top_p": 0.8,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "stop": ["END"],
            "seed": 42,
            "max_tokens": 1234,
            "reasoning_effort": "high",
        },
        project_id="project-1",
    )

    assert envelope["project"] == "project-1"
    assert envelope["model"] == "gemini-test"
    request = envelope["request"]
    assert request["systemInstruction"]["parts"] == [
        {"text": "Be useful."},
        {"text": "Follow project rules."},
    ]
    assert request["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    declaration = request["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "lookup"
    assert "additionalProperties" not in declaration["parameters"]
    assert request["toolConfig"] == {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": ["lookup"],
        }
    }
    assert request["generationConfig"] == {
        "temperature": 0.25,
        "topP": 0.8,
        "presencePenalty": 0.1,
        "frequencyPenalty": 0.2,
        "maxOutputTokens": 1234,
        "stopSequences": ["END"],
        "seed": 42,
        "thinkingConfig": {"thinkingLevel": "HIGH"},
    }


def test_tool_choice_modes_map_to_gemini() -> None:
    base = {
        "model": "gemini-test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    }
    expected = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
    for choice, mode in expected.items():
        request = openai_chat_to_cloudcode(
            {**base, "tool_choice": choice}, project_id="project-1"
        )["request"]
        assert request["toolConfig"]["functionCallingConfig"] == {"mode": mode}


def test_assistant_reasoning_replays_as_gemini_thought() -> None:
    envelope = openai_chat_to_cloudcode(
        {
            "model": "gemini-test",
            "messages": [
                {
                    "role": "assistant",
                    "reasoning_content": "prior reasoning",
                    "content": "prior answer",
                }
            ],
        },
        project_id="project-1",
    )
    assert envelope["request"]["contents"] == [
        {
            "role": "model",
            "parts": [
                {"thought": True, "text": "prior reasoning"},
                {"text": "prior answer"},
            ],
        }
    ]


def test_thought_signature_round_trips_through_tool_call_id() -> None:
    state = StreamState(model="gemini-test")
    chunks = gemini_event_chunks(
        {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call-1",
                                        "name": "lookup",
                                        "args": {"q": "weather"},
                                    },
                                    "thoughtSignature": "sig-123",
                                }
                            ]
                        }
                    }
                ]
            }
        },
        state,
    )

    tool = chunks[0].choices[0].delta.tool_calls[0]
    assert tool.id == "call-1|sig-123"

    envelope = openai_chat_to_cloudcode(
        {
            "model": "gemini-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool.id,
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"weather"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": "sunny",
                },
            ],
        },
        project_id="project-1",
    )

    model_part = envelope["request"]["contents"][0]["parts"][0]
    assert model_part["thoughtSignature"] == "sig-123"
    assert model_part["functionCall"]["id"] == "call-1"
    tool_part = envelope["request"]["contents"][1]["parts"][0]
    assert tool_part["functionResponse"] == {
        "id": "call-1",
        "name": "lookup",
        "response": {"output": "sunny"},
    }


def test_gemini_text_reasoning_usage_and_finish_chunks() -> None:
    state = StreamState(model="gemini-test")
    chunks = gemini_event_chunks(
        {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 4,
                },
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"thought": True, "text": "thinking"},
                                {"text": "answer"},
                            ]
                        },
                    }
                ],
            }
        },
        state,
    )

    assert chunks[0].choices[0].delta.reasoning_content == "thinking"
    assert chunks[1].choices[0].delta.content == "answer"
    final = final_chunk(state)
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.prompt_tokens == 10
    assert final.usage.completion_tokens == 4


def test_multiple_tool_calls_keep_distinct_stream_indexes() -> None:
    state = StreamState(model="gemini-test")
    chunks = gemini_event_chunks(
        {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call-1",
                                        "name": "first",
                                        "args": {"x": 1},
                                    }
                                },
                                {
                                    "functionCall": {
                                        "id": "call-2",
                                        "name": "second",
                                        "args": {"y": 2},
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        },
        state,
    )

    assert [chunk.choices[0].delta.tool_calls[0].index for chunk in chunks] == [0, 1]
    assert [chunk.choices[0].delta.tool_calls[0].id for chunk in chunks] == [
        "call-1",
        "call-2",
    ]
    assert final_chunk(state).choices[0].finish_reason == "tool_calls"


def test_max_tokens_finish_reason_maps_to_openai_length() -> None:
    state = StreamState(model="gemini-test")
    assert not gemini_event_chunks(
        {"response": {"candidates": [{"finishReason": "MAX_TOKENS"}]}},
        state,
    )
    assert final_chunk(state).choices[0].finish_reason == "length"


def test_safety_finish_reason_maps_to_content_filter() -> None:
    state = StreamState(model="gemini-test")
    assert not gemini_event_chunks(
        {"response": {"candidates": [{"finishReason": "SAFETY"}]}},
        state,
    )
    assert final_chunk(state).choices[0].finish_reason == "content_filter"


def test_final_chunk_uses_tool_calls_finish_reason() -> None:
    state = StreamState(model="gemini-test", saw_tool=True)
    assert final_chunk(state).choices[0].finish_reason == "tool_calls"


def test_chunk_shape_matches_fcc_attribute_access() -> None:
    state = StreamState(model="gemini-test")
    chunk = gemini_event_chunks(
        {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}, state
    )[0]
    assert isinstance(chunk.choices[0].delta, SimpleNamespace)
    assert chunk.model == "gemini-test"
