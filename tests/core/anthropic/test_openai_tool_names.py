import re

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec

_PORTABLE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _request(*names: str) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": name,
                    "description": "test",
                    "input_schema": {"type": "object"},
                }
                for name in names
            ],
        }
    )


def test_portable_names_at_boundaries_remain_unchanged() -> None:
    names = ("x", "A" * 64, "read_file-2")
    codec = OpenAIToolNameCodec.from_request(_request(*names))

    assert codec.has_aliases is False
    assert [codec.encode(name) for name in names] == list(names)
    assert [codec.decode(name) for name in names] == list(names)
    assert all(codec.is_unchanged_name(name) for name in names)


def test_long_and_invalid_names_round_trip_through_portable_aliases() -> None:
    names = (
        "x" * 65,
        "mcp__server__" + "tool" * 23,
        "mcp.server/tool name",
        "工具名称",
    )
    codec = OpenAIToolNameCodec.from_request(_request(*names))

    assert codec.has_aliases is True
    for name in names:
        alias = codec.encode(name)
        assert alias != name
        assert _PORTABLE_NAME.fullmatch(alias)
        assert codec.decode(alias) == name


def test_reported_102_character_name_becomes_portable() -> None:
    name = "mcp__issue_1307__" + "x" * 85
    assert len(name) == 102

    alias = OpenAIToolNameCodec.from_request(_request(name)).encode(name)

    assert len(alias) <= 64
    assert _PORTABLE_NAME.fullmatch(alias)


def test_aliases_are_unique_for_names_with_the_same_long_prefix() -> None:
    common = "mcp__one_very_long_server_namespace__" + "x" * 50
    names = (f"{common}_first", f"{common}_second")
    codec = OpenAIToolNameCodec.from_request(_request(*names))

    assert codec.encode(names[0]) != codec.encode(names[1])


def test_aliases_are_order_independent() -> None:
    names = (
        "mcp__server_a__" + "x" * 70,
        "mcp__server_b__" + "y" * 70,
        "normal_tool",
    )
    forward = OpenAIToolNameCodec.from_request(_request(*names))
    reverse = OpenAIToolNameCodec.from_request(_request(*reversed(names)))

    assert {name: forward.encode(name) for name in names} == {
        name: reverse.encode(name) for name in names
    }


def test_alias_does_not_shadow_an_existing_portable_name() -> None:
    long_name = "mcp__collision_candidate__" + "x" * 70
    first = OpenAIToolNameCodec.from_request(_request(long_name))
    first_alias = first.encode(long_name)

    codec = OpenAIToolNameCodec.from_request(_request(long_name, first_alias))

    assert codec.encode(first_alias) == first_alias
    assert codec.encode(long_name) != first_alias
    assert codec.decode(codec.encode(long_name)) == long_name


def test_codec_collects_forced_choice_and_history_names_without_mutation() -> None:
    declared = "safe_tool"
    forced = "mcp__forced__" + "f" * 70
    historical = "mcp__historical__" + "h" * 70
    request = MessagesRequest.model_validate(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": historical,
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "done",
                        }
                    ],
                },
            ],
            "tools": [{"name": declared, "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": forced},
        }
    )
    snapshot = request.model_dump()

    codec = OpenAIToolNameCodec.from_request(request)

    assert codec.encode(declared) == declared
    assert codec.decode(codec.encode(forced)) == forced
    assert codec.decode(codec.encode(historical)) == historical
    assert request.model_dump() == snapshot


def test_only_generated_aliases_are_decoded_and_prefixes_are_detected() -> None:
    name = "mcp__fragmented__" + "x" * 70
    codec = OpenAIToolNameCodec.from_request(_request(name))
    alias = codec.encode(name)

    assert codec.is_alias(alias) is True
    assert codec.is_alias(alias[:20]) is False
    assert codec.is_unchanged_name(alias) is False
    assert codec.is_alias_prefix(alias[:20]) is True
    assert codec.is_alias_prefix(alias) is False
    assert codec.is_alias_prefix("unrelated") is False
    assert codec.decode("unknown_tool") == "unknown_tool"


def test_empty_name_retains_existing_invalid_identity() -> None:
    codec = OpenAIToolNameCodec.from_request(_request(""))

    assert codec.encode("") == ""
    assert codec.decode("") == ""
