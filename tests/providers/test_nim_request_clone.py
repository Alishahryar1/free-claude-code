"""Tests for NVIDIA NIM request body cloning helpers."""

from copy import deepcopy

from free_claude_code.providers.nvidia_nim.retry import (
    clone_body_without_reasoning_budget,
)


def test_clone_body_without_reasoning_budget_strips_top_level_and_nested():
    body: dict = {
        "model": "x",
        "extra_body": {
            "reasoning_budget": 99,
            "chat_template_kwargs": {"reasoning_budget": 42, "thinking": True},
            "top_k": 1,
        },
    }
    original_extra = deepcopy(body["extra_body"])
    out = clone_body_without_reasoning_budget(body)

    assert out is not None
    assert out["extra_body"]["chat_template_kwargs"] == {"thinking": True}
    assert "reasoning_budget" not in out["extra_body"]
    assert body["extra_body"] == original_extra


def test_clone_body_without_reasoning_budget_returns_none_when_unchanged():
    body = {"model": "x", "extra_body": {"top_k": 3}}
    assert clone_body_without_reasoning_budget(body) is None


def test_clone_body_without_reasoning_budget_returns_none_without_extra_body():
    assert clone_body_without_reasoning_budget({"model": "y"}) is None


def test_clone_body_drops_empty_extra_body_after_strip():
    body = {"model": "z", "extra_body": {"reasoning_budget": 7}}
    out = clone_body_without_reasoning_budget(body)
    assert out is not None
    assert "extra_body" not in out
    assert "extra_body" in body
