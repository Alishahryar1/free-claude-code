import json
from types import SimpleNamespace

import pytest

from free_claude_code.providers.nvidia_nim import native_tool_stream
from free_claude_code.providers.nvidia_nim.native_tool_stream import (
    NimNativeToolProtocolError,
    normalize_nim_native_tool_stream,
)

NS = "]<]minimax[>["
TOOL_START = f"{NS}<tool_call>"
TOOL_END = f"{NS}</tool_call>"
INVOKE_END = f"{NS}</invoke>"


def _element(name: str, value: str) -> str:
    return f"{NS}<{name}>{value}{NS}</{name}>"


def _invoke(name: str, arguments: str) -> str:
    return f'{NS}<invoke name="{name}">{arguments}{INVOKE_END}'


def _tool_block(*invokes: str) -> str:
    return f"{TOOL_START}\n{'\n'.join(invokes)}\n{TOOL_END}"


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _structured_call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        id="call_structured",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _stream(chunks: list[SimpleNamespace]):
    for chunk in chunks:
        yield chunk


async def _normalize(
    chunks: list[SimpleNamespace], body: dict
) -> list[SimpleNamespace]:
    return [
        chunk async for chunk in normalize_nim_native_tool_stream(_stream(chunks), body)
    ]


def _body(*functions: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]
    }


def _function(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _content(chunks: list[SimpleNamespace]) -> str:
    return "".join(
        content
        for chunk in chunks
        if (content := getattr(chunk.choices[0].delta, "content", None)) is not None
    )


def _reasoning(chunks: list[SimpleNamespace]) -> str:
    return "".join(
        reasoning
        for chunk in chunks
        if (
            reasoning := getattr(
                chunk.choices[0].delta,
                "reasoning_content",
                None,
            )
        )
        is not None
    )


def _tool_calls(chunks: list[SimpleNamespace]) -> list[SimpleNamespace]:
    calls: list[SimpleNamespace] = []
    for chunk in chunks:
        value = getattr(chunk.choices[0].delta, "tool_calls", None)
        if isinstance(value, list):
            calls.extend(call for call in value if isinstance(call, SimpleNamespace))
    return calls


def _nested_argument(depth: int) -> tuple[str, dict]:
    value = "leaf"
    schema: dict = {"type": "string"}
    for index in reversed(range(depth)):
        name = f"level_{index}"
        value = _element(name, value)
        schema = {
            "type": "object",
            "properties": {name: schema},
            "required": [name],
        }
    return value, schema


@pytest.mark.asyncio
async def test_normalizer_preserves_ordinary_text_incrementally() -> None:
    chunks = [
        _chunk(content="Hello "),
        _chunk(content="world", finish_reason="stop"),
    ]

    normalized = await _normalize(chunks, _body())

    assert _content(normalized) == "Hello world"
    assert not _tool_calls(normalized)
    assert normalized[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_normalizer_preserves_ordinary_stream_without_tool_schemas() -> None:
    source = _chunk(
        content="Answer",
        reasoning_content="Thinking",
        finish_reason="stop",
    )

    normalized = await _normalize([source], {})

    assert normalized == [source]
    assert normalized[0] is source


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
async def test_normalizer_decodes_tool_block_from_each_text_field(field: str) -> None:
    raw = "Preparing. " + _tool_block(
        _invoke("Read", _element("file_path", "README.md"))
    )
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    normalized = await _normalize(
        [
            _chunk(
                content=raw if field == "content" else None,
                reasoning_content=raw if field == "reasoning_content" else None,
                finish_reason="tool_calls",
            )
        ],
        body,
    )

    assert NS not in _content(normalized)
    assert NS not in _reasoning(normalized)
    assert (_content(normalized) if field == "content" else _reasoning(normalized)) == (
        "Preparing. "
    )
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert calls[0].function.name == "Read"
    assert json.loads(calls[0].function.arguments) == {"file_path": "README.md"}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
async def test_normalizer_rejects_native_calls_without_tool_schemas(
    field: str,
) -> None:
    raw = _tool_block(_invoke("Write", _element("file_path", "output.md")))

    with pytest.raises(NimNativeToolProtocolError, match="undeclared tool"):
        await _normalize(
            [
                _chunk(
                    content=raw if field == "content" else None,
                    reasoning_content=raw if field == "reasoning_content" else None,
                    finish_reason="tool_calls",
                )
            ],
            {},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
async def test_normalizer_rejects_unframed_native_namespace(field: str) -> None:
    raw = f'{NS}<invoke name="Write">'

    with pytest.raises(NimNativeToolProtocolError, match="malformed"):
        await _normalize(
            [
                _chunk(
                    content=raw if field == "content" else None,
                    reasoning_content=raw if field == "reasoning_content" else None,
                    finish_reason="stop",
                )
            ],
            {},
        )


@pytest.mark.asyncio
async def test_normalizer_rejects_native_calls_in_multiple_text_fields() -> None:
    raw = _tool_block(_invoke("Read", _element("file_path", "README.md")))
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    with pytest.raises(NimNativeToolProtocolError, match="multiple output fields"):
        await _normalize(
            [
                _chunk(
                    content=raw,
                    reasoning_content=raw,
                    finish_reason="tool_calls",
                )
            ],
            body,
        )


@pytest.mark.asyncio
async def test_normalizer_decodes_tool_block_split_at_every_character() -> None:
    raw = "Let me check. " + _tool_block(
        _invoke(
            "get_weather",
            _element("city", "Seattle") + _element("days", "5"),
        )
    )
    chunks = [_chunk(content=character) for character in raw]
    chunks.append(_chunk(finish_reason="stop"))
    body = _body(
        _function(
            "get_weather",
            {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            ["city", "days"],
        )
    )

    normalized = await _normalize(chunks, body)

    assert _content(normalized) == "Let me check. "
    assert NS not in _content(normalized)
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert calls[0].index == 0
    assert calls[0].function.name == "get_weather"
    assert json.loads(calls[0].function.arguments) == {
        "city": "Seattle",
        "days": 5,
    }
    assert normalized[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_normalizer_decodes_parallel_recursive_arguments_in_order() -> None:
    shipping = _element(
        "shipping",
        _element("city", "Singapore") + _element("zip", "018956"),
    )
    items = _element(
        "items",
        _element(
            "item",
            _element("sku", "book-001") + _element("qty", "2"),
        )
        + _element(
            "item",
            _element("sku", "pen-007") + _element("qty", "5"),
        ),
    )
    raw = _tool_block(
        _invoke(
            "create_order",
            _element("urgent", "true") + shipping + items,
        ),
        _invoke("ping", _element("message", "ready")),
    )
    body = _body(
        _function(
            "create_order",
            {
                "urgent": {"type": "boolean"},
                "shipping": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "integer"},
                    },
                    "required": ["city", "zip"],
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"},
                        },
                        "required": ["sku", "qty"],
                    },
                },
            },
            ["urgent", "shipping", "items"],
        ),
        _function("ping", {"message": {"type": "string"}}, ["message"]),
    )

    normalized = await _normalize(
        [_chunk(content=raw, finish_reason="tool_calls")],
        body,
    )

    calls = _tool_calls(normalized)
    assert [call.index for call in calls] == [0, 1]
    assert [call.function.name for call in calls] == ["create_order", "ping"]
    assert json.loads(calls[0].function.arguments) == {
        "urgent": True,
        "shipping": {"city": "Singapore", "zip": 18956},
        "items": [
            {"sku": "book-001", "qty": 2},
            {"sku": "pen-007", "qty": 5},
        ],
    }
    assert json.loads(calls[1].function.arguments) == {"message": "ready"}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
async def test_normalizer_prefers_structured_calls_and_strips_raw_duplicate(
    field: str,
) -> None:
    raw = "Checking. " + _tool_block(f'{NS}<invoke name="">{INVOKE_END}')
    structured = _structured_call("Read", '{"file_path":"README.md"}')
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    normalized = await _normalize(
        [
            _chunk(
                content=raw if field == "content" else None,
                reasoning_content=raw if field == "reasoning_content" else None,
                tool_calls=[structured],
            ),
            _chunk(finish_reason="tool_calls"),
        ],
        body,
    )

    assert (_content(normalized) if field == "content" else _reasoning(normalized)) == (
        "Checking. "
    )
    assert _tool_calls(normalized) == [structured]
    assert NS not in _content(normalized)
    assert NS not in _reasoning(normalized)


@pytest.mark.asyncio
async def test_normalizer_preserves_literal_xml_like_string_content() -> None:
    literal = 'C:\\work\\a<b & "quoted">.txt'
    raw = _tool_block(_invoke("Read", _element("file_path", literal)))
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    normalized = await _normalize(
        [_chunk(content=raw, finish_reason="tool_calls")],
        body,
    )

    calls = _tool_calls(normalized)
    assert json.loads(calls[0].function.arguments) == {"file_path": literal}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        TOOL_START,
        f'{TOOL_START}{NS}<invoke name="Read">',
        f'{TOOL_START}{NS}<invoke name="">{INVOKE_END}{TOOL_END}',
        _tool_block(_invoke("Unknown", _element("path", "x"))),
        _tool_block(_invoke("Read", _element("wrong", "x"))),
        _tool_block(
            _invoke(
                "Read",
                f"{NS}<file_path>x{NS}</wrong>",
            )
        ),
    ],
)
async def test_normalizer_rejects_incomplete_or_invalid_native_calls(raw: str) -> None:
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    with pytest.raises(NimNativeToolProtocolError):
        await _normalize([_chunk(content=raw, finish_reason="stop")], body)


@pytest.mark.asyncio
async def test_normalizer_enforces_native_tool_block_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke = _invoke("Read", _element("file_path", "README.md"))
    block_body = f"\n{invoke}\n"
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )
    monkeypatch.setattr(
        native_tool_stream,
        "_MAX_NATIVE_TOOL_BLOCK_CHARS",
        len(block_body),
    )

    normalized = await _normalize(
        [
            _chunk(
                content=f"{TOOL_START}{block_body}{TOOL_END}",
                finish_reason="tool_calls",
            )
        ],
        body,
    )

    assert len(_tool_calls(normalized)) == 1
    with pytest.raises(NimNativeToolProtocolError, match="oversized"):
        await _normalize(
            [
                _chunk(
                    content=f"{TOOL_START}{block_body}x{TOOL_END}",
                    finish_reason="tool_calls",
                )
            ],
            body,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("split_suffix", [False, True])
async def test_normalizer_accepts_whitespace_after_native_tool_block(
    split_suffix: bool,
) -> None:
    raw = _tool_block(_invoke("Read", _element("file_path", "README.md")))
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )
    suffix = "\n \t"
    chunks = (
        [_chunk(content=raw), _chunk(content=suffix, finish_reason="tool_calls")]
        if split_suffix
        else [_chunk(content=f"{raw}{suffix}", finish_reason="tool_calls")]
    )

    normalized = await _normalize(chunks, body)

    assert len(_tool_calls(normalized)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("split_suffix", [False, True])
@pytest.mark.parametrize(
    "suffix",
    [
        "unexpected",
        _tool_block(_invoke("Read", _element("file_path", "SECOND.md"))),
    ],
)
async def test_normalizer_rejects_content_after_native_tool_block(
    suffix: str,
    split_suffix: bool,
) -> None:
    raw = _tool_block(_invoke("Read", _element("file_path", "README.md")))
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )
    chunks = (
        [_chunk(content=raw), _chunk(content=suffix, finish_reason="tool_calls")]
        if split_suffix
        else [_chunk(content=f"{raw}{suffix}", finish_reason="tool_calls")]
    )

    with pytest.raises(NimNativeToolProtocolError, match="content after"):
        await _normalize(chunks, body)


@pytest.mark.asyncio
async def test_normalizer_accepts_native_argument_nesting_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = 3
    arguments, parameters = _nested_argument(depth)
    monkeypatch.setattr(
        native_tool_stream,
        "_MAX_NATIVE_ARGUMENT_DEPTH",
        depth,
    )
    raw = _tool_block(_invoke("deep_tool", arguments))
    body = _body(
        {
            "name": "deep_tool",
            "parameters": parameters,
        }
    )

    normalized = await _normalize(
        [_chunk(content=raw, finish_reason="tool_calls")],
        body,
    )

    assert len(_tool_calls(normalized)) == 1


@pytest.mark.asyncio
async def test_normalizer_rejects_native_argument_nesting_above_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 3
    arguments, parameters = _nested_argument(limit + 1)
    monkeypatch.setattr(
        native_tool_stream,
        "_MAX_NATIVE_ARGUMENT_DEPTH",
        limit,
    )
    raw = _tool_block(_invoke("deep_tool", arguments))
    body = _body(
        {
            "name": "deep_tool",
            "parameters": parameters,
        }
    )

    with pytest.raises(NimNativeToolProtocolError, match="excessively nested"):
        await _normalize(
            [_chunk(content=raw, finish_reason="tool_calls")],
            body,
        )
