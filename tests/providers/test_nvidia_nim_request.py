"""Tests for NVIDIA NIM request policy helpers."""

import re
from copy import deepcopy
from typing import Any

import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import set_if_not_none
from free_claude_code.core.anthropic.models import MessagesRequest, Tool
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.nvidia_nim.request_options import (
    _set_extra,
)
from free_claude_code.providers.nvidia_nim.request_options import (
    build_nim_request_body as build_request_body,
)
from free_claude_code.providers.nvidia_nim.retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_content,
)
from free_claude_code.providers.nvidia_nim.tool_schema import (
    NIM_TOOL_ARGUMENT_ALIASES_KEY,
    body_without_nim_tool_argument_aliases,
    nim_tool_argument_aliases_from_body,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import REASONING_OFF, REASONING_ON

GREP_SCHEMA_FROM_SERVER_LOG: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "The regular expression"},
        "path": {"type": "string", "description": "File or directory to search"},
        "glob": {"type": "string", "description": "Glob to filter files"},
        "output_mode": {
            "type": "string",
            "enum": ["content", "files_with_matches", "count"],
        },
        "-A": {"type": "number", "description": "Lines after match"},
        "-B": {"type": "number", "description": "Lines before match"},
        "-C": {"type": "number", "description": "Lines around match"},
        "-i": {"type": "boolean", "description": "Case insensitive"},
        "-n": {"type": "boolean", "description": "Show line numbers"},
        "type": {"type": "string", "description": "File type to search"},
    },
    "additionalProperties": False,
    "required": ["pattern"],
}


@pytest.fixture
def req() -> MessagesRequest:
    return make_messages_request(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        system=None,
        temperature=None,
        top_p=None,
        stop_sequences=None,
        tools=None,
        extra_body=None,
        top_k=None,
        thinking=None,
    )


class TestSetIfNotNone:
    def test_value_not_none_sets(self):
        body = {}
        set_if_not_none(body, "key", "value")
        assert body["key"] == "value"

    def test_value_none_skips(self):
        body = {}
        set_if_not_none(body, "key", None)
        assert "key" not in body


class TestSetExtra:
    def test_key_in_extra_body_skips(self):
        extra = {"top_k": 42}
        _set_extra(extra, "top_k", 10)
        assert extra["top_k"] == 42

    def test_value_none_skips(self):
        extra = {}
        _set_extra(extra, "top_k", None)
        assert "top_k" not in extra

    def test_value_equals_ignore_value_skips(self):
        extra = {}
        _set_extra(extra, "top_k", -1, ignore_value=-1)
        assert "top_k" not in extra

    def test_value_set_when_valid(self):
        extra = {}
        _set_extra(extra, "top_k", 10, ignore_value=-1)
        assert extra["top_k"] == 10


class TestBuildRequestBody:
    def test_absent_sampling_defaults_are_not_injected(self, req):
        body = build_request_body(req, reasoning=REASONING_OFF)

        assert "temperature" not in body
        assert "top_p" not in body
        assert "parallel_tool_calls" not in body
        assert "extra_body" not in body

    def test_explicit_sampling_and_extras_survive(self, req):
        req.temperature = 0.3
        req.top_p = 1.0
        req.top_k = 15
        req.extra_body = {"top_k": 20, "ignore_eos": True, "seed": 7}
        original = deepcopy(req.model_dump())

        body = build_request_body(req, reasoning=REASONING_OFF)

        assert body["temperature"] == 0.3
        assert body["top_p"] == 1.0
        assert body["extra_body"] == req.extra_body
        assert req.model_dump() == original

    @pytest.mark.parametrize("keep_custom", (False, True))
    def test_reasoning_extras_are_cleaned_without_mutating_request(
        self, req, keep_custom
    ):
        template_kwargs: dict[str, Any] = {"thinking": True, "reasoning_budget": 100}
        if keep_custom:
            template_kwargs["custom"] = "value"
        req.extra_body = {
            "reasoning_budget": 99,
            "chat_template_kwargs": template_kwargs,
        }
        if keep_custom:
            req.extra_body["ignore_eos"] = True
        original = deepcopy(req.model_dump())

        body = build_request_body(req, reasoning=REASONING_OFF)

        if keep_custom:
            assert body["extra_body"] == {
                "chat_template_kwargs": {"custom": "value"},
                "ignore_eos": True,
            }
        else:
            assert "extra_body" not in body
        assert req.model_dump() == original

    @pytest.mark.parametrize("effort", list(ReasoningEffort))
    def test_named_effort_passes_through(self, req, effort):
        body = build_request_body(req, reasoning=ReasoningPolicy(effort=effort))
        assert body["reasoning_effort"] == effort.value
        assert "chat_template_kwargs" not in body.get("extra_body", {})

    def test_named_effort_replaces_client_reasoning_budgets(self):
        req = make_messages_request(
            model="test",
            thinking=None,
            extra_body={
                "reasoning_budget": 99,
                "chat_template_kwargs": {
                    "reasoning_budget": 100,
                    "custom": "value",
                },
            },
        )

        body = build_request_body(
            req,
            reasoning=ReasoningPolicy(effort=ReasoningEffort.HIGH),
        )

        extra_body = body["extra_body"]
        assert "reasoning_budget" not in extra_body
        assert body["reasoning_effort"] == "high"
        assert extra_body["chat_template_kwargs"] == {"custom": "value"}

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            (None, ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS),
            (100, 100),
            (
                ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS + 1,
                ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            ),
        ],
    )
    def test_max_tokens_preserves_fcc_output_limit(self, req, requested, expected):
        req.max_tokens = requested
        body = build_request_body(req, reasoning=REASONING_ON)
        assert body["max_tokens"] == expected

    @pytest.mark.parametrize("client_top_p", (None, 0.0, 0.5, 0.95, 1.0))
    def test_top_p_preserves_client_value_or_omission(self, req, client_top_p):
        req.top_p = client_top_p

        body = build_request_body(req, reasoning=REASONING_ON)

        if client_top_p is None:
            assert "top_p" not in body
        else:
            assert body["top_p"] == client_top_p

    def test_include_stop_str_in_output_not_sent(self, req):
        body = build_request_body(req, reasoning=REASONING_ON)
        assert "include_stop_str_in_output" not in body.get("extra_body", {})

    def test_tool_schema_boolean_subschemas_are_removed_without_mutating_request(
        self, req
    ):
        tool_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": False},
                "blocked": False,
                "nested": {"type": "object", "additionalProperties": False},
                "choice": {"anyOf": [False, {"type": "string"}]},
            },
            "additionalProperties": False,
            "required": ["query"],
        }
        req.tools = [
            Tool(
                name="search",
                description="search",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, reasoning=REASONING_OFF)

        parameters = body["tools"][0]["function"]["parameters"]
        properties = parameters["properties"]
        assert "additionalProperties" not in parameters
        assert "blocked" not in properties
        assert "additionalProperties" not in properties["nested"]
        assert properties["choice"]["anyOf"] == [{"type": "string"}]
        assert properties["query"]["default"] is False
        assert tool_schema["additionalProperties"] is False
        assert tool_schema["properties"]["nested"]["additionalProperties"] is False

    def test_grep_schema_type_parameter_is_aliased_without_mutating_request(self, req):
        tool_schema = deepcopy(GREP_SCHEMA_FROM_SERVER_LOG)
        tool_schema["properties"]["_fcc_arg_type"] = {
            "type": "string",
            "description": "Existing safe property that collides with the alias",
        }
        tool_schema["required"] = ["pattern", "-A", "_fcc_arg_type"]
        original_schema = deepcopy(tool_schema)
        req.tools = [
            Tool(
                name="Grep",
                description="Search file contents",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, reasoning=REASONING_OFF)

        parameters = body["tools"][0]["function"]["parameters"]
        properties = parameters["properties"]
        aliases = body[NIM_TOOL_ARGUMENT_ALIASES_KEY]["Grep"]
        assert "additionalProperties" not in parameters
        assert properties["-A"] == original_schema["properties"]["-A"]
        assert properties["-B"] == original_schema["properties"]["-B"]
        assert properties["-C"] == original_schema["properties"]["-C"]
        assert properties["-i"] == original_schema["properties"]["-i"]
        assert properties["-n"] == original_schema["properties"]["-n"]
        assert "type" not in properties
        assert properties["pattern"] == original_schema["properties"]["pattern"]
        assert properties["output_mode"]["enum"] == [
            "content",
            "files_with_matches",
            "count",
        ]
        assert (
            properties["_fcc_arg_type"]
            == original_schema["properties"]["_fcc_arg_type"]
        )
        assert aliases == {"_fcc_arg_type_2": "type"}
        assert properties["_fcc_arg_type_2"] == original_schema["properties"]["type"]
        assert "-A" in parameters["required"]
        assert "_fcc_arg_type" in parameters["required"]
        assert tool_schema == original_schema

    def test_reported_long_tool_name_uses_generic_alias_after_nim_repairs(self, req):
        """Issue #1307's 52nd tool is portable without losing NIM arg aliases."""
        long_name = "mcp__issue_1307__" + "x" * 85
        assert len(long_name) == 102
        tools = [
            Tool(name=f"tool_{index}", input_schema={"type": "object"})
            for index in range(51)
        ]
        tools.append(
            Tool(
                name=long_name,
                input_schema={
                    "type": "object",
                    "properties": {"type": {"type": "string"}},
                    "required": ["type"],
                },
            )
        )
        req.tools = tools
        snapshot = req.model_dump()

        body = build_request_body(req, reasoning=REASONING_OFF)

        wire_name = body["tools"][51]["function"]["name"]
        assert wire_name != long_name
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", wire_name)
        assert body[NIM_TOOL_ARGUMENT_ALIASES_KEY] == {
            long_name: {"_fcc_arg_type": "type"}
        }
        assert (
            NIM_TOOL_ARGUMENT_ALIASES_KEY
            not in body_without_nim_tool_argument_aliases(body)
        )
        assert req.model_dump() == snapshot

    def test_safe_tool_schema_does_not_add_alias_metadata(self, req):
        tool_schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "output_mode": {"type": "string", "enum": ["content", "count"]},
            },
            "required": ["pattern"],
        }
        req.tools = [
            Tool(
                name="Glob",
                description="Find files",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, reasoning=REASONING_OFF)

        assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in body
        parameters = body["tools"][0]["function"]["parameters"]
        assert parameters["properties"] == tool_schema["properties"]
        assert parameters["required"] == ["pattern"]

    def test_nested_schema_keyword_properties_are_aliased_without_mutating_request(
        self, req
    ):
        tool_schema = {
            "type": "object",
            "properties": {
                "parent": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["page_id"]},
                        "id": {"type": "string"},
                    },
                    "required": ["type", "id"],
                }
            },
            "required": ["parent"],
        }
        original_schema = deepcopy(tool_schema)
        req.tools = [
            Tool(
                name="NotionLike",
                description="Nested type schema",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, reasoning=REASONING_OFF)

        aliases = body[NIM_TOOL_ARGUMENT_ALIASES_KEY]["NotionLike"]
        parent = body["tools"][0]["function"]["parameters"]["properties"]["parent"]
        parent_properties = parent["properties"]
        assert "type" not in parent_properties
        assert parent_properties["_fcc_arg_type"] == {
            "type": "string",
            "enum": ["page_id"],
        }
        assert parent["required"] == ["_fcc_arg_type", "id"]
        assert aliases == {"_fcc_arg_type": "type"}
        assert tool_schema == original_schema

    def test_private_alias_metadata_is_stripped_without_mutating_body(self):
        body = {
            "model": "test",
            NIM_TOOL_ARGUMENT_ALIASES_KEY: {"Grep": {"_fcc_arg_A": "-A"}},
        }

        upstream_body = body_without_nim_tool_argument_aliases(body)

        assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in upstream_body
        assert body[NIM_TOOL_ARGUMENT_ALIASES_KEY] == {"Grep": {"_fcc_arg_A": "-A"}}
        assert nim_tool_argument_aliases_from_body(body) == {
            "Grep": {"_fcc_arg_A": "-A"}
        }

    def test_reasoning_uses_named_effort(self):
        req = make_messages_request(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
            top_k=None,
            thinking=None,
        )

        body = build_request_body(req, reasoning=REASONING_ON)
        extra = body.get("extra_body", {})
        assert body["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in extra
        assert "reasoning_budget" not in extra

    def test_canonicalization_removes_empty_client_reasoning_envelope(self):
        req = make_messages_request(
            model="test",
            extra_body={
                "chat_template_kwargs": {
                    "thinking": True,
                    "enable_thinking": True,
                    "reasoning_budget": 100,
                }
            },
        )

        body = build_request_body(
            req,
            reasoning=ReasoningPolicy.provider_default(),
        )

        assert "chat_template_kwargs" not in body.get("extra_body", {})

    def test_clone_body_without_chat_template(self):
        body = {
            "model": "test",
            "extra_body": {
                "chat_template": "custom_template",
                "chat_template_kwargs": {
                    "thinking": True,
                    "enable_thinking": True,
                    "reasoning_budget": 100,
                },
                "ignore_eos": False,
            },
        }

        cloned = clone_body_without_chat_template(body)

        assert cloned is not None
        assert "chat_template" not in cloned["extra_body"]
        assert "chat_template_kwargs" not in cloned["extra_body"]
        assert cloned["extra_body"]["ignore_eos"] is False
        assert body["extra_body"]["chat_template"] == "custom_template"
        assert body["extra_body"]["chat_template_kwargs"] == {
            "thinking": True,
            "enable_thinking": True,
            "reasoning_budget": 100,
        }

    def test_clone_body_without_chat_template_kwargs_only(self):
        body = {
            "model": "test",
            "extra_body": {
                "chat_template_kwargs": {
                    "thinking": True,
                    "enable_thinking": True,
                    "reasoning_budget": 100,
                },
                "ignore_eos": False,
            },
        }

        cloned = clone_body_without_chat_template(body)

        assert cloned is not None
        assert "chat_template" not in cloned["extra_body"]
        assert "chat_template_kwargs" not in cloned["extra_body"]
        assert cloned["extra_body"]["ignore_eos"] is False

    def test_clone_body_without_chat_template_returns_none_when_unchanged(self):
        body = {"model": "test", "extra_body": {"ignore_eos": False}}

        assert clone_body_without_chat_template(body) is None

    def test_no_chat_template_kwargs_when_thinking_disabled(self):
        req = make_messages_request(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
            top_k=None,
            thinking=None,
        )

        body = build_request_body(req, reasoning=REASONING_OFF)
        extra = body.get("extra_body", {})
        assert body["reasoning_effort"] == "none"
        assert "chat_template_kwargs" not in extra
        assert "reasoning_budget" not in extra

    def test_named_effort_preserves_unrelated_chat_template_kwargs(self):
        req = make_messages_request(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            top_k=None,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "custom": "value",
                }
            },
            thinking=None,
        )

        body = build_request_body(req, reasoning=REASONING_ON)
        assert body["reasoning_effort"] == "high"
        assert body["extra_body"]["chat_template_kwargs"] == {"custom": "value"}

    def test_explicit_chat_template_is_preserved(self):
        req = make_messages_request(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
            top_k=None,
            thinking=None,
        )

        req.extra_body = {"chat_template": "custom_template"}
        body = build_request_body(req, reasoning=REASONING_ON)
        extra = body.get("extra_body", {})
        assert body["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in extra
        assert extra["chat_template"] == "custom_template"

    def test_no_reasoning_params_in_extra_body(self):
        req = make_messages_request(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
            top_k=None,
            thinking=None,
        )

        body = build_request_body(req, reasoning=REASONING_OFF)
        extra = body.get("extra_body", {})
        for param in (
            "thinking",
            "reasoning_split",
            "return_tokens_as_token_ids",
            "include_reasoning",
            "reasoning_effort",
        ):
            assert param not in extra
        assert body["reasoning_effort"] == "none"
        assert "chat_template_kwargs" not in extra

    def test_budget_only_request_uses_enabled_named_effort(self):
        req = make_messages_request(model="test", thinking=None)

        body = build_request_body(
            req,
            reasoning=ReasoningPolicy.on(budget_tokens=321),
        )

        assert body["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in body.get("extra_body", {})
        assert "reasoning_budget" not in body.get("extra_body", {})

    def test_assistant_thinking_blocks_removed_when_disabled(self):
        req = make_messages_request(
            model="test",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "secret"},
                        {"type": "text", "text": "answer"},
                    ],
                }
            ],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
            top_k=None,
            thinking=None,
        )

        body = build_request_body(req, reasoning=REASONING_OFF)
        assert "<think>" not in body["messages"][0]["content"]
        assert "answer" in body["messages"][0]["content"]

    def test_assistant_thinking_replayed_as_reasoning_content_when_enabled(self):
        req = make_messages_request(
            model="test",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "secret"},
                        {"type": "text", "text": "answer"},
                    ],
                }
            ],
            max_tokens=100,
            system=None,
            temperature=None,
            top_p=None,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
            top_k=None,
            thinking=None,
        )

        body = build_request_body(req, reasoning=REASONING_ON)
        assistant = body["messages"][0]
        assert assistant["reasoning_content"] == "secret"
        assert assistant["content"] == "answer"
        assert "<think>" not in assistant["content"]

    def test_clone_body_without_reasoning_content(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "secret",
                },
            ],
        }

        cloned = clone_body_without_reasoning_content(body)

        assert cloned is not None
        assert "reasoning_content" not in cloned["messages"][1]
        assert body["messages"][1]["reasoning_content"] == "secret"

    def test_clone_body_without_reasoning_content_returns_none_when_unchanged(self):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}

        assert clone_body_without_reasoning_content(body) is None
