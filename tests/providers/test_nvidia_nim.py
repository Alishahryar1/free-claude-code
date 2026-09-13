import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx2 import Request, Response

from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import NVIDIA_NIM_DEFAULT_BASE
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.nvidia_nim.client import _PROFILE as NIM_PROFILE
from free_claude_code.providers.nvidia_nim.tool_schema import (
    NIM_TOOL_ARGUMENT_ALIASES_KEY,
)
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    reasoning_for,
)
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply


def message(role, content):
    return {"role": role, "content": content}


def tool(name, description, input_schema):
    return {"name": name, "description": description, "input_schema": input_schema}


def block(**fields):
    return fields


def make_request(**overrides):
    model = overrides.pop("model", "test-model")
    overrides.setdefault("stop_sequences", ["STOP"])
    return make_messages_request(model, **overrides)


def _alias_provider():
    return NvidiaNimProvider(
        make_provider_config(api_key="a", base_url="https://provider.invalid/v1"),
        nim_settings=NimSettings(chat_template="custom_template"),
        admission=immediate_admission(max_attempts=4),
    )


def _alias_request(wire="messages", *, reasoning_history=False):
    schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "type": {"type": "string"}},
        "required": ["pattern", "type"],
        "additionalProperties": False,
    }
    history = [{"role": "user", "content": "Search"}]
    if reasoning_history:
        carrier = encode_replay(
            ReplayRecord(
                ReplayOrigin(
                    "source", "chat", "https://source.invalid/v1", "a", "test"
                ),
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "Need the tool.",
                            "index": 0,
                        }
                    ]
                },
            )
        )
        history = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Need the tool.",
                        "signature": carrier,
                    },
                    {
                        "type": "tool_use",
                        "name": "Grep",
                        "id": "prior",
                        "input": {"pattern": "needle", "type": "py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "prior",
                        "content": "result",
                    },
                ],
            },
        ]
    if wire == "messages":
        return make_request(
            system=None,
            messages=history,
            tools=[tool("Grep", "Search file contents", schema)],
        )
    return OpenAIResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": history,
            "tools": [{"type": "function", "name": "Grep", "parameters": schema}],
        }
    )


def _alias_events():
    template = _events_for("chat")[0]
    return [
        {
            **template,
            "choices": [
                {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": finish}
            ],
        }
        for call, finish in [
            (
                {
                    "index": 0,
                    "id": "call_grep",
                    "type": "function",
                    "function": {
                        "name": "Grep",
                        "arguments": '{"pattern":"needle","_fcc_arg_',
                    },
                },
                None,
            ),
            ({"index": 0, "function": {"arguments": 'type":"py"}'}}, "tool_calls"),
        ]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire,correction",
    [
        ("messages", None),
        ("responses", None),
        ("messages", "chat_template"),
        ("responses", "chat_template"),
        ("messages", "reasoning_budget"),
        ("messages", "reasoning_content"),
    ],
)
async def test_argument_aliases_survive_request_corrections(wire, correction):
    request = _alias_request(wire, reasoning_history=correction == "reasoning_content")
    original = deepcopy(request.model_dump())

    def responder(bodies):
        if len(bodies) == 1 and correction:
            return 400, {"message": f"Unsupported field: {correction}"}
        return 200, _alias_events()

    async with _harness("chat", responder, chat_provider_factory=_alias_provider) as (
        _,
        bodies,
        provider,
    ):
        stream = (
            provider.stream_messages
            if wire == "messages"
            else provider.stream_responses
        )
        saved = await _saved_reply(
            stream(request, reasoning=ReasoningPolicy.on(budget_tokens=4096)), wire
        )
    if wire == "messages":
        call = next(
            block for block in saved[0]["content"] if block["type"] == "tool_use"
        )
        assert call["input"] == {"pattern": "needle", "type": "py"}
        assert call["id"] == "call_grep"
    else:
        call = next(item for item in saved if item["type"] == "function_call")
        assert json.loads(call["arguments"]) == {"pattern": "needle", "type": "py"}
        assert call["call_id"] == "call_grep"
    assert len(bodies) == (2 if correction else 1)
    assert all(NIM_TOOL_ARGUMENT_ALIASES_KEY not in body for body in bodies)
    if correction == "reasoning_content":
        assert "reasoning_content" not in bodies[-1]["messages"][0]
        assert json.dumps(bodies[-1]).count("Need the tool.") == 1
    assert request.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("early_sse", [False, True])
async def test_argument_aliases_survive_shared_history_correction(early_sse):
    # Test adapter exercises the shared history path; native NIM disables details.
    def provider_factory():
        with patch(
            "free_claude_code.providers.nvidia_nim.client._PROFILE",
            replace(NIM_PROFILE, structured_reasoning_details=True),
        ):
            return _alias_provider()

    request = _alias_request()
    request.messages.insert(
        0,
        type(request.messages[0]).model_validate(
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "opaque-original"},
                ],
            }
        ),
    )

    def responder(bodies):
        if len(bodies) == 1:
            error = {
                "code": "invalid_encrypted_content",
                "message": "The encrypted content could not be verified.",
            }
            return (200, [{"error": error}]) if early_sse else (400, error)
        return 200, _alias_events()

    async with _harness("chat", responder, chat_provider_factory=provider_factory) as (
        _,
        bodies,
        provider,
    ):
        saved = await _saved_reply(provider.stream_messages(request), "messages")
    call = next(block for block in saved[0]["content"] if block["type"] == "tool_use")
    assert call["input"] == {"pattern": "needle", "type": "py"}
    assert len(bodies) == 2
    assert "opaque-original" not in json.dumps(bodies[-1])
    assert all(NIM_TOOL_ARGUMENT_ALIASES_KEY not in body for body in bodies)


def _input_json_deltas(events):
    deltas = []
    for event in events:
        if "event: content_block_delta" not in event:
            continue
        for line in event.splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            delta = payload.get("delta", {})
            if delta.get("type") == "input_json_delta":
                deltas.append(delta.get("partial_json", ""))
    return deltas


def _tool_call_chunk(
    *,
    name,
    arguments,
    tool_id="call_1",
    index=0,
    finish_reason=None,
):
    mock_tc = MagicMock()
    mock_tc.index = index
    mock_tc.id = tool_id
    mock_tc.function.name = name
    mock_tc.function.arguments = arguments

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=finish_reason,
        )
    ]
    mock_chunk.usage = None
    return mock_chunk


def _content_chunk(
    content,
    *,
    finish_reason=None,
    reasoning_content=None,
):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=None,
            ),
            finish_reason=finish_reason,
        )
    ]
    mock_chunk.usage = None
    return mock_chunk


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(
        status_code=400,
        request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
    )
    body = {"error": {"message": message, "type": "BadRequestError", "code": 400}}
    return openai.BadRequestError(message, response=response, body=body)


def _make_internal_server_error(message: str) -> openai.InternalServerError:
    response = Response(
        status_code=500,
        request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
    )
    body = {
        "error": {
            "message": message,
            "type": "internal_server_error",
            "code": 500,
        }
    }
    return openai.InternalServerError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = NvidiaNimProvider(
            provider_config,
            nim_settings=NimSettings(),
            admission=immediate_admission(),
        )
        assert provider._api_key == "test_key"
        assert provider._base_url == "https://test.api.nvidia.com/v1"
        mock_openai.assert_called_once()


@pytest.mark.asyncio
async def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""

    config = make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        NvidiaNimProvider(
            config, nim_settings=NimSettings(), admission=immediate_admission()
        )
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_build_request_body(provider_config):
    """Test request body construction."""
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_request()
    body = provider._chat._build_request_body(req, reasoning=reasoning_for(req))

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    ctk = body["extra_body"]["chat_template_kwargs"]
    assert ctk["thinking"] is True
    assert ctk["enable_thinking"] is True
    assert "reasoning_budget" not in ctk
    assert "reasoning_budget" not in body["extra_body"]


def test_responses_request_uses_nim_chat_policy(provider_config):
    original_name = "mcp__responses__" + "x" * 80
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(max_tokens=64, parallel_tool_calls=False),
        admission=immediate_admission(),
    )
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "Hello",
            "max_output_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "name": original_name,
                    "parameters": {
                        "type": "object",
                        "properties": {"type": {"type": "string"}},
                        "required": ["type"],
                    },
                }
            ],
        }
    )

    translated = provider._chat._build_responses_request_body(
        request, reasoning=REASONING_ON
    )

    body = translated.body
    tools = body["tools"]
    assert isinstance(tools, list)
    function = tools[0]["function"]
    assert isinstance(function, dict)
    wire_name = function["name"]
    assert wire_name == translated.tool_names.encode(original_name)
    assert wire_name != original_name
    assert body["max_tokens"] == 64
    assert body["top_p"] == 0.95
    assert body["parallel_tool_calls"] is False
    assert body[NIM_TOOL_ARGUMENT_ALIASES_KEY] == {
        original_name: {"_fcc_arg_type": "type"}
    }
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["properties"] == {"_fcc_arg_type": {"type": "string"}}
    extra_body = body["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
    }


@pytest.mark.asyncio
async def test_build_request_body_encodes_explicit_reasoning_off(
    provider_config,
):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_request()
    body = provider._chat._build_request_body(req, reasoning=REASONING_OFF)

    extra = body.get("extra_body", {})
    assert extra["chat_template_kwargs"] == {
        "thinking": False,
        "enable_thinking": False,
    }
    assert "reasoning_budget" not in extra


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    provider_config,
):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_request()
    req.thinking.enabled = False
    body = provider._chat._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


def test_preflight_and_build_request_issue_206_post_tool_text(nim_provider):
    """Regression: assistant message with tool_use then text plus tool results (GitHub #206)."""
    tool_id = "toolu_issue_206"
    req = make_request(
        messages=[
            message("user", "Use echo once."),
            message(
                "assistant",
                [
                    block(
                        type="tool_use",
                        id=tool_id,
                        name="echo_smoke",
                        input={"value": "FCC_206"},
                    ),
                    block(
                        type="text",
                        text="Commentary after the tool row.",
                    ),
                ],
            ),
            message(
                "user",
                [
                    block(type="tool_result", tool_use_id=tool_id, content="FCC_206"),
                    block(type="text", text="What was echoed?"),
                ],
            ),
        ],
    )
    nim_provider.preflight_messages(req, reasoning=REASONING_OFF)
    body = nim_provider._chat._build_request_body(req, reasoning=REASONING_OFF)
    assert "messages" in body
    assert any(m.get("role") == "tool" for m in body["messages"])


@pytest.mark.asyncio
async def test_stream_messages_text(nim_provider):
    """Test streaming text response."""
    req = make_request()

    # Create mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=""), finish_reason=None
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_messages_thinking_reasoning_content(nim_provider):
    """Test streaming with native reasoning_content."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None
    stop_chunk = MagicMock()
    stop_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    stop_chunk.usage = None

    async def mock_stream():
        yield mock_chunk
        yield stop_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

        # Check for thinking_delta
        found_thinking = False
        for e in events:
            if (
                "event: content_block_delta" in e
                and '"thinking_delta"' in e
                and "Thinking..." in e
            ):
                found_thinking = True
        assert found_thinking


@pytest.mark.asyncio
async def test_stream_messages_suppresses_thinking_when_disabled(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="<think>secret</think>Answer", reasoning_content="Thinking..."
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            e async for e in provider.stream_messages(req, reasoning=REASONING_OFF)
        ]

    event_text = "".join(events)
    assert "thinking_delta" not in event_text
    assert "Thinking..." not in event_text
    assert "secret" not in event_text
    assert "Answer" in event_text


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(status_code=400, request=Request("POST", "http://test"))
    body = {"error": {"message": message}}
    return openai.BadRequestError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_stream_messages_retries_without_chat_template(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(chat_template="custom_template"),
        admission=immediate_admission(),
    )
    req = make_request(model="mistralai/mixtral-8x7b-instruct-v0.1")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="OK", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    first_error = _make_bad_request_error(
        "chat_template is not supported for Mistral tokenizers."
    )

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [first_error, mock_stream()]

        events = [
            e async for e in provider.stream_messages(req, reasoning=REASONING_ON)
        ]

    assert mock_create.await_count == 2

    first_extra = mock_create.call_args_list[0].kwargs["extra_body"]
    second_extra = mock_create.call_args_list[1].kwargs["extra_body"]

    assert first_extra["chat_template"] == "custom_template"
    assert first_extra["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
    }
    assert "reasoning_budget" not in first_extra

    assert "chat_template" not in second_extra
    assert "chat_template_kwargs" not in second_extra
    assert "reasoning_budget" not in second_extra

    event_text = "".join(events)
    assert "event: error" not in event_text
    assert "OK" in event_text


@pytest.mark.asyncio
async def test_stream_messages_retries_without_chat_template_kwargs_issue_993(
    provider_config,
):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_request(model="mistralai/mistral-small-4-119b-2603")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="OK", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    first_error = _make_bad_request_error(
        "chat_template is not supported for Mistral tokenizers."
    )

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [first_error, mock_stream()]

        events = [
            e async for e in provider.stream_messages(req, reasoning=REASONING_ON)
        ]

    assert mock_create.await_count == 2

    first_extra = mock_create.call_args_list[0].kwargs["extra_body"]
    second_kwargs = mock_create.call_args_list[1].kwargs

    assert "chat_template" not in first_extra
    assert first_extra["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
    }
    second_extra = second_kwargs.get("extra_body") or {}
    assert "chat_template" not in second_extra
    assert "chat_template_kwargs" not in second_extra

    event_text = "".join(events)
    assert "event: error" not in event_text
    assert "OK" in event_text


@pytest.mark.asyncio
async def test_stream_messages_does_not_retry_unrelated_bad_request(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(chat_template="custom_template"),
        admission=immediate_admission(),
    )
    req = make_request(model="mistralai/mixtral-8x7b-instruct-v0.1")

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = _make_bad_request_error("unrelated bad request")

        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in provider.stream_messages(req)]

    assert mock_create.await_count == 1
    assert "Invalid request sent to provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls."""
    req = make_request()

    # Mock tool call delta
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]


@pytest.mark.asyncio
async def test_native_minimax_tool_markup_becomes_anthropic_tool_use(nim_provider):
    req = make_request(
        tools=[
            tool(
                "Read",
                "Read one file",
                {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            )
        ]
    )
    namespace = "]<]minimax[>["
    raw = (
        "I will read it."
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Read">'
        f"{namespace}<file_path>README.md{namespace}</file_path>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )

    async def mock_stream():
        yield _content_chunk(raw, finish_reason="stop")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in nim_provider.stream_messages(req)]

    event_text = "".join(events)
    assert namespace not in event_text
    assert "I will read it." in event_text
    assert '"type": "tool_use"' in event_text
    assert '"name": "Read"' in event_text
    assert json.loads(_input_json_deltas(events)[0]) == {"file_path": "README.md"}


@pytest.mark.asyncio
async def test_native_minimax_reasoning_markup_becomes_anthropic_tool_use(
    nim_provider,
):
    req = make_request(
        tools=[
            tool(
                "Write",
                "Write one file",
                {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            )
        ]
    )
    namespace = "]<]minimax[>["
    raw = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Write">'
        f"{namespace}<file_path>output.md{namespace}</file_path>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )

    async def mock_stream():
        yield _content_chunk(
            None,
            reasoning_content=raw,
            finish_reason="tool_calls",
        )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in nim_provider.stream_messages(req)]

    event_text = "".join(events)
    assert namespace not in event_text
    assert '"type": "tool_use"' in event_text
    assert '"name": "Write"' in event_text
    assert json.loads(_input_json_deltas(events)[0]) == {"file_path": "output.md"}


@pytest.mark.asyncio
async def test_native_minimax_markup_without_tools_retries_without_leaking(
    nim_provider,
):
    req = make_request(tools=None)
    namespace = "]<]minimax[>["
    raw = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Write">'
        f"{namespace}<file_path>output.md{namespace}</file_path>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )

    async def native_stream():
        yield _content_chunk(raw, finish_reason="tool_calls")

    async def recovered_stream():
        yield _content_chunk("Recovered safely.", finish_reason="stop")

    attempts = [native_stream() for _ in range(UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS - 1)]
    attempts.append(recovered_stream())
    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = attempts

        events = [event async for event in nim_provider.stream_messages(req)]

    event_text = "".join(events)
    assert mock_create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert namespace not in event_text
    assert event_text.count("Recovered safely.") == 1
    assert '"type": "tool_use"' not in event_text


@pytest.mark.asyncio
async def test_native_minimax_tool_markup_restores_nim_argument_aliases(nim_provider):
    req = make_request(
        tools=[
            tool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern", "type"],
                },
            )
        ]
    )
    namespace = "]<]minimax[>["
    raw = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Grep">'
        f"{namespace}<pattern>needle{namespace}</pattern>"
        f"{namespace}<_fcc_arg_type>py{namespace}</_fcc_arg_type>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )

    async def mock_stream():
        yield _content_chunk(raw, finish_reason="tool_calls")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in nim_provider.stream_messages(req)]

    assert json.loads(_input_json_deltas(events)[0]) == {
        "pattern": "needle",
        "type": "py",
    }


@pytest.mark.asyncio
async def test_malformed_native_minimax_tool_call_retries_without_leaking(
    nim_provider,
):
    req = make_request(
        tools=[
            tool(
                "Read",
                "Read one file",
                {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            )
        ]
    )
    namespace = "]<]minimax[>["
    malformed = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="">'
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )
    recovered = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Read">'
        f"{namespace}<file_path>README.md{namespace}</file_path>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )

    async def malformed_stream():
        yield _content_chunk(malformed, finish_reason="stop")

    async def recovered_stream():
        yield _content_chunk(recovered, finish_reason="tool_calls")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [malformed_stream(), recovered_stream()]

        events = [event async for event in nim_provider.stream_messages(req)]

    assert mock_create.await_count == 2
    event_text = "".join(events)
    assert namespace not in event_text
    assert '"name": "Read"' in event_text
    assert json.loads(_input_json_deltas(events)[0]) == {"file_path": "README.md"}


@pytest.mark.asyncio
async def test_midstream_native_tool_suffix_failure_recovers_without_duplication(
    nim_provider,
):
    req = make_request(
        tools=[
            tool(
                "Read",
                "Read one file",
                {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            )
        ]
    )
    namespace = "]<]minimax[>["
    malformed = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Read">'
        f"{namespace}<file_path>IGNORED.md{namespace}</file_path>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
        "unexpected trailing content"
    )
    recovered = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="Read">'
        f"{namespace}<file_path>README.md{namespace}</file_path>"
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )

    async def malformed_stream():
        yield _content_chunk("Starting the read. ")
        yield _content_chunk(malformed, finish_reason="stop")

    async def recovered_stream():
        yield _content_chunk(recovered, finish_reason="tool_calls")

    def immediate_holdback():
        return RecoveryHoldbackBuffer(holdback_seconds=0.0)

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create,
        patch(
            "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
            side_effect=immediate_holdback,
        ),
    ):
        mock_create.side_effect = [malformed_stream(), recovered_stream()]

        events = [event async for event in nim_provider.stream_messages(req)]

    assert mock_create.await_count == 2
    event_text = "".join(events)
    assert namespace not in event_text
    assert event_text.count("Starting the read.") == 1
    assert '"name": "Read"' in event_text
    assert json.loads(_input_json_deltas(events)[0]) == {"file_path": "README.md"}
    assert "event: error" not in event_text


@pytest.mark.asyncio
async def test_stream_messages_restores_aliased_tool_arguments(nim_provider):
    """NIM-safe argument aliases are restored before Anthropic SSE emission."""
    req = make_request(
        tools=[
            tool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "-A": {"type": "number"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="Grep",
        arguments=json.dumps({"pattern": "needle", "-A": 2, "_fcc_arg_type": "py"}),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

    await_args = mock_create.await_args
    assert await_args is not None
    create_kwargs = await_args.kwargs
    assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in create_kwargs
    properties = create_kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert "-A" in properties
    assert "type" not in properties
    assert "_fcc_arg_A" not in properties
    assert "_fcc_arg_type" in properties

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"pattern": "needle", "-A": 2, "type": "py"}
    assert "_fcc_arg_type" not in deltas[0]


@pytest.mark.asyncio
async def test_stream_messages_buffers_chunked_aliased_tool_arguments(nim_provider):
    """Chunked aliased args are emitted once as restored Claude Code args."""
    req = make_request(
        tools=[
            tool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ]
    )
    first_chunk = _tool_call_chunk(
        name="Grep",
        arguments='{"pattern": "needle", ',
        tool_id="call_chunked",
    )
    second_chunk = _tool_call_chunk(
        name=None,
        arguments='"_fcc_arg_type": "py"}',
        tool_id="call_chunked",
    )

    async def mock_stream():
        yield first_chunk
        yield second_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"pattern": "needle", "type": "py"}


@pytest.mark.asyncio
async def test_stream_messages_restores_nested_aliased_tool_arguments(nim_provider):
    req = make_request(
        tools=[
            tool(
                "NotionLike",
                "Nested type schema",
                {
                    "type": "object",
                    "properties": {
                        "parent": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "id": {"type": "string"},
                            },
                            "required": ["type", "id"],
                        }
                    },
                    "required": ["parent"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="NotionLike",
        arguments=json.dumps(
            {"parent": {"_fcc_arg_type": "page_id", "id": "page_123"}}
        ),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"parent": {"type": "page_id", "id": "page_123"}}


@pytest.mark.asyncio
async def test_stream_messages_task_tool_still_forces_background_false(nim_provider):
    req = make_request(
        tools=[
            tool(
                "Task",
                "Run a subagent",
                {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "run_in_background": {"type": "boolean"},
                    },
                    "required": ["description", "prompt"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="Task",
        arguments=json.dumps(
            {
                "description": "Inspect",
                "prompt": "Read the marker",
                "run_in_background": True,
            }
        ),
        tool_id="call_task",
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_messages(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0])["run_in_background"] is False


@pytest.mark.asyncio
async def test_stream_messages_retries_without_reasoning_budget(nim_provider):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_budget")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [
            e
            async for e in nim_provider.stream_messages(
                req,
                reasoning=ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
            )
        ]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["extra_body"]["chat_template_kwargs"]["reasoning_budget"] == 4096
    assert "reasoning_budget" not in second_call["extra_body"]
    assert "reasoning_budget" not in second_call["extra_body"]["chat_template_kwargs"]
    assert second_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_messages_retries_without_budget_for_thinking_token_error(
    nim_provider,
):
    req = make_request(model="meta/llama-3.3-70b-instruct")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_internal_server_error(
        "ValueError: thinking_token_budget is set but reasoning_config is not "
        "configured. Please set --reasoning-config to use thinking_token_budget."
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [
            e
            async for e in nim_provider.stream_messages(
                req, reasoning=ReasoningPolicy.on(budget_tokens=77)
            )
        ]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["extra_body"]["chat_template_kwargs"]["reasoning_budget"] == 77
    assert "reasoning_budget" not in second_call["extra_body"]
    assert "reasoning_budget" not in second_call["extra_body"]["chat_template_kwargs"]
    assert second_call["extra_body"]["chat_template_kwargs"]["thinking"] is True
    assert second_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_messages_retries_without_reasoning_content(nim_provider):
    req = make_request(
        system=None,
        messages=[
            message(
                "assistant",
                [
                    block(type="thinking", thinking="Need the tool."),
                    block(
                        type="tool_use",
                        id="toolu_reasoning",
                        name="echo_smoke",
                        input={"value": "FCC_TOOL"},
                    ),
                ],
            ),
            message(
                "user",
                [
                    block(
                        type="tool_result",
                        tool_use_id="toolu_reasoning",
                        content="result",
                    )
                ],
            ),
        ],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_content")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_messages(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["messages"][0]["reasoning_content"] == "Need the tool."
    assert "reasoning_content" not in second_call["messages"][0]
    assert (
        "[Earlier reasoning]\nNeed the tool." in second_call["messages"][0]["content"]
    )
    assert second_call["messages"][0]["tool_calls"][0]["id"] == "toolu_reasoning"
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_messages_bad_request_without_reasoning_budget_does_not_retry(
    nim_provider,
):
    req = make_request()
    error = _make_bad_request_error("Unsupported field: top_k")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in nim_provider.stream_messages(req)]

    assert mock_create.await_count == 1
    assert "Invalid request sent to provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_messages_unrelated_internal_error_does_not_downgrade(
    nim_provider,
):
    req = make_request()
    error = _make_internal_server_error("unrelated internal provider failure")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in nim_provider.stream_messages(req)]

    assert mock_create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert all(
        call.kwargs == mock_create.await_args_list[0].kwargs
        for call in mock_create.await_args_list
    )
    assert "Provider API request failed" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_messages_internal_reasoning_content_error_does_not_downgrade(
    nim_provider,
):
    req = make_request()
    error = _make_internal_server_error(
        "reasoning_content could not be processed by the upstream model"
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in nim_provider.stream_messages(req)]

    assert mock_create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert all(
        call.kwargs == mock_create.await_args_list[0].kwargs
        for call in mock_create.await_args_list
    )
    assert "Provider API request failed" in exc_info.value.message
