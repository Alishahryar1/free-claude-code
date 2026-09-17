import pytest

from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
)
from free_claude_code.core.openai_responses.web_request import (
    prepare_web_request,
    validate_action,
)


def request(tool, **extra):
    return OpenAIResponsesRequest(model="test", input="hi", tools=[tool], **extra)


@pytest.mark.parametrize("types", [None, ["text"], ["text", "image"]])
def test_normal_codex_text_declarations(types):
    tool = {"type": "web_search", "external_web_access": False}
    if types is not None:
        tool["search_content_types"] = types
    original = request(tool)
    prepared, spec = prepare_web_request(original)
    assert spec is not None and not spec.live
    assert prepared.tools is not None
    assert prepared.tools[0]["type"] == "function"
    assert original.tools == [tool]


@pytest.mark.parametrize(
    "extra",
    [
        {"search_content_types": ["image"]},
        {"search_content_types": []},
        {"external_web_access": "false"},
        {"indexed_web_access": True},
        {"user_location": {"type": "approximate", "city": "Paris"}},
        {"filters": {"blocked_domains": ["example.org"]}},
        {"search_context_size": "unlimited"},
        {"return_token_budget": 100},
    ],
)
def test_unsupported_constraints_are_not_silently_dropped(extra):
    with pytest.raises(ResponsesConversionError):
        prepare_web_request(request({"type": "web_search", **extra}))


@pytest.mark.parametrize("choice", ["auto", "none", "required", {"type": "web_search"}])
def test_choice_mapping(choice):
    prepared, spec = prepare_web_request(
        request({"type": "web_search"}, tool_choice=choice)
    )
    assert spec is not None
    if choice == "none":
        assert prepared.tools == [] and not spec.allowed
    elif isinstance(choice, dict):
        assert prepared.tool_choice == {"type": "function", "name": spec.name}
    else:
        assert prepared.tool_choice == choice


def test_ordinary_same_name_and_discovered_names_do_not_collide():
    original = request({"type": "web_search"})
    original.tools += [
        {"type": "function", "name": "fcc_web", "parameters": {"type": "object"}}
    ]
    original.input = [
        {"role": "user", "content": "hi"},
        {
            "type": "tool_search_call",
            "call_id": "search",
            "execution": "client",
            "arguments": {},
        },
        {
            "type": "tool_search_output",
            "call_id": "search",
            "tools": [
                {
                    "type": "function",
                    "name": "fcc_web_1",
                    "parameters": {"type": "object"},
                }
            ],
        },
    ]
    _, spec = prepare_web_request(original)
    assert spec is not None
    assert spec.name == "fcc_web_2"


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"action": "other"},
        {"action": "search", "query": ""},
        {"action": "search", "query": "docs", "url": "https://example.org"},
        {"action": "find_in_page", "url": "url"},
    ],
)
def test_model_arguments_are_validated_before_io(value):
    with pytest.raises(ValueError):
        validate_action(value)


@pytest.mark.parametrize(
    "choice", [{"type": []}, {"type": {}}, {"type": "web_search", "mode": "other"}]
)
def test_malformed_web_choice_is_a_request_error(choice):
    with pytest.raises(ResponsesConversionError):
        prepare_web_request(request({"type": "web_search"}, tool_choice=choice))


def test_web_declaration_preserves_a_named_client_namespace_choice():
    original = request({"type": "web_search"})
    original.tools.append(
        {
            "type": "namespace",
            "name": "files",
            "tools": [
                {"type": "function", "name": "read", "parameters": {"type": "object"}}
            ],
        }
    )
    original.tool_choice = {"type": "tool", "namespace": "files", "name": "read"}
    prepared, spec = prepare_web_request(original)
    assert prepared.tool_choice == original.tool_choice
    assert spec is not None and not spec.allowed
