import pytest

from free_claude_code.core.gateway_model_ids import (
    decode_gateway_model_id,
    desktop_model_id,
)


@pytest.mark.parametrize(
    "ref",
    [
        "nvidia_nim/nemotron-3.5",
        "open_router/qwen/qwen3",
        "openai/gpt-5",
        "open_router/模型/one",
    ],
)
@pytest.mark.parametrize("no_thinking", [False, True])
def test_desktop_ids_roundtrip_without_vendor_fragments(ref, no_thinking):
    wire = desktop_model_id(ref, no_thinking=no_thinking)
    assert wire.startswith("claude-")
    assert not any(part in wire for part in ("gpt", "qwen", "nemotron", "openai"))
    decoded = decode_gateway_model_id(wire)
    assert decoded is not None
    assert f"{decoded.provider_id}/{decoded.provider_model}" == ref
    assert decoded.force_reasoning_off is no_thinking
    assert desktop_model_id(ref, no_thinking=no_thinking) == wire


@pytest.mark.parametrize(
    "suffix", ["", "ff", "not-hex", "123", "6E", "6162", "2f78", "782f", "61 2f62"]
)
@pytest.mark.parametrize("prefix", ["claude-fcc/", "claude-3-fcc/"])
def test_malformed_reserved_ids_do_not_fall_through(prefix, suffix):
    with pytest.raises(ValueError):
        decode_gateway_model_id(prefix + suffix)
