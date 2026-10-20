import base64
from pathlib import Path

import httpx
import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_lines
from smoke.lib.config import SmokeConfig, auth_headers
from smoke.lib.e2e import ConversationDriver, SmokeServerDriver, assert_product_stream

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("nvidia_nim_vision")]

_FIXTURE = Path(__file__).resolve().parents[1] / "assets" / "vision-fcc.png"
_VISIBLE_TOKEN = "K7P4"


def test_nvidia_nim_vision_tool_result_e2e(smoke_config: SmokeConfig) -> None:
    """Keep a Claude-style image tool result visual through Chat egress."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")
    provider_model = smoke_config.nvidia_nim_vision_model()
    if provider_model is None:
        pytest.skip("missing_env: FCC_SMOKE_MODEL_NVIDIA_NIM_VISION is required")

    image_data = base64.b64encode(_FIXTURE.read_bytes()).decode("ascii")
    payload = {
        "model": provider_model.full_model,
        "max_tokens": 128,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_read_image",
                        "name": "Read",
                        "input": {"file_path": "vision-fcc.png"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_read_image",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Return only the uppercase token visible in the "
                                    "image."
                                ),
                            },
                        ],
                    }
                ],
            },
        ],
    }
    with SmokeServerDriver(
        smoke_config,
        name="product-nvidia-nim-vision-tool-result",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
            "REASONING_POLICY": "off",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)

    assert_product_stream(turn.events)
    assert _VISIBLE_TOKEN in turn.text.upper(), (
        f"{provider_model.full_model} did not read the expected fixture token: "
        f"{turn.text!r}"
    )


def test_nvidia_nim_vision_function_output_e2e(smoke_config: SmokeConfig) -> None:
    """Keep a Responses image function output visual through Chat egress."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")
    provider_model = smoke_config.nvidia_nim_vision_model()
    if provider_model is None:
        pytest.skip("missing_env: FCC_SMOKE_MODEL_NVIDIA_NIM_VISION is required")

    image_data = base64.b64encode(_FIXTURE.read_bytes()).decode("ascii")
    payload = {
        "model": provider_model.full_model,
        "input": [
            {
                "type": "function_call",
                "call_id": "call_read_image",
                "name": "Read",
                "arguments": '{"file_path":"vision-fcc.png"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_read_image",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                    {
                        "type": "input_text",
                        "text": "Return only the uppercase token visible in the image.",
                    },
                ],
            },
        ],
        "max_output_tokens": 128,
        "stream": True,
    }
    with (
        SmokeServerDriver(
            smoke_config,
            name="product-nvidia-nim-vision-function-output",
            env_overrides={
                "MODEL": provider_model.full_model,
                "MESSAGING_PLATFORM": "none",
                "REASONING_POLICY": "off",
            },
        ).run() as server,
        httpx.stream(
            "POST",
            f"{server.base_url}/v1/responses",
            headers=auth_headers(),
            json=payload,
            timeout=smoke_config.timeout_s,
        ) as response,
    ):
        assert response.status_code == 200, response.read()
        events = parse_sse_lines(response.iter_lines())

    event_names = [event.event for event in events]
    assert event_names[0] == "response.created", event_names
    assert event_names[-1] == "response.completed", event_names
    text = "".join(
        str(event.data.get("delta", ""))
        for event in events
        if event.event == "response.output_text.delta"
    )
    assert _VISIBLE_TOKEN in text.upper(), (
        f"{provider_model.full_model} did not read the expected fixture token: {text!r}"
    )
