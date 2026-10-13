from copy import deepcopy

import httpx2
import pytest
from openai import BadRequestError

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.reasoning import ReasoningCapability, ReasoningPolicy
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from tests.providers.request_factory import make_messages_request


@pytest.mark.parametrize(
    ("capability", "can_disable", "control", "limit"),
    [
        ("none", True, "default", 64),
        ("optional", True, "off", 64),
        ("optional", False, "default", 8192),
        ("required", True, "default", 8192),
        ("unknown", True, "off", 8192),
        ("unknown", False, "default", 8192),
    ],
)
def test_prepare_uses_explicit_capability_and_available_encoding(
    capability, can_disable, control, limit
):
    request = make_messages_request(
        "arbitrary",
        max_tokens=64,
        thinking={"type": "disabled"},
        output_config={"effort": "high", "format": {"type": "text"}},
    )
    prepared, policy = prepare_messages_reasoning(
        request,
        ReasoningPolicy.prefer_off(),
        model_info=ProviderModelInfo(
            "arbitrary", reasoning_capability=ReasoningCapability(capability)
        ),
        can_disable=can_disable,
        normal_max_tokens=8192,
    )
    assert policy.control.value == control
    assert prepared.max_tokens == limit
    assert prepared.thinking is None
    assert prepared.output_config == {"format": {"type": "text"}}
    assert request.max_tokens == 64
    assert request.thinking is not None and request.thinking.type == "disabled"


@pytest.mark.parametrize(
    ("requested", "normal", "expected"), [(12000, 8192, 12000), (12000, None, None)]
)
def test_normal_allowance_preserves_larger_limits_or_omission(
    requested, normal, expected
):
    request = make_messages_request("route", max_tokens=requested)
    prepared, _ = prepare_messages_reasoning(
        request,
        ReasoningPolicy.prefer_off(),
        model_info=None,
        can_disable=True,
        normal_max_tokens=normal,
    )
    assert prepared.max_tokens == expected


@pytest.mark.parametrize(
    ("message", "param", "correct"),
    [
        (
            "Reasoning is mandatory for this endpoint and cannot be disabled.",
            None,
            True,
        ),
        ("Thinking cannot be disabled for this endpoint.", None, True),
        ("Unsupported parameter: reasoning_effort", None, True),
        ("Value 'none' is not supported", "reasoning_effort", True),
        ("Value must be one of: low, high", "reasoning_effort", True),
        (
            "Use reasoning_effort for thinking. The response_format parameter is unsupported.",
            "response_format",
            False,
        ),
        (
            "Reasoning is enabled. This response_format cannot be disabled.",
            "response_format",
            False,
        ),
        ("Tool schema required fields missing", "tools", False),
        ("Unknown error", None, False),
    ],
)
def test_correction_requires_rejection_of_the_sent_control(message, param, correct):
    body = {
        "model": "route",
        "messages": [{"role": "user", "content": "classify"}],
        "reasoning_effort": "none",
        "max_tokens": 64,
        "stream_options": {"include_usage": True},
    }
    original = deepcopy(body)
    error = BadRequestError(
        message,
        response=httpx2.Response(
            400, request=httpx2.Request("POST", "https://test.invalid")
        ),
        body={"error": {"message": message, "param": param}},
    )
    result = ReasoningCorrection(
        (("reasoning_effort",),), "max_tokens", 8192
    ).retry_body(error, body)
    assert (result is not None) is correct
    if result is not None:
        assert result == {
            key: value
            for key, value in original.items()
            if key not in {"max_tokens", "reasoning_effort"}
        } | {"max_tokens": 8192}
    assert body == original


@pytest.mark.parametrize("sent", [{}, {"reasoning_effort": "high"}])
def test_correction_requires_the_off_control_to_have_reached_the_wire(sent):
    error = BadRequestError(
        "Reasoning cannot be disabled",
        response=httpx2.Response(
            400, request=httpx2.Request("POST", "https://test.invalid")
        ),
        body={"message": "Reasoning cannot be disabled"},
    )
    correction = ReasoningCorrection((("reasoning_effort",),), "max_tokens", 8192)
    assert (
        correction.retry_body(error, {"reasoning_effort": "none"}, sent_body=sent)
        is None
    )


def test_correction_removes_both_template_controls_preserving_other_options():
    error = BadRequestError(
        "Thinking cannot be disabled",
        response=httpx2.Response(
            400, request=httpx2.Request("POST", "https://test.invalid")
        ),
        body={"message": "Thinking cannot be disabled"},
    )
    body = {
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": False,
                "enable_thinking": False,
                "other": 1,
            }
        },
        "max_tokens": 64,
    }
    correction = ReasoningCorrection(
        (
            ("extra_body", "chat_template_kwargs", "thinking"),
            ("extra_body", "chat_template_kwargs", "enable_thinking"),
        ),
        "max_tokens",
        8192,
        output_cap=4096,
    )
    assert correction.retry_body(error, body) == {
        "extra_body": {"chat_template_kwargs": {"other": 1}},
        "max_tokens": 4096,
    }
