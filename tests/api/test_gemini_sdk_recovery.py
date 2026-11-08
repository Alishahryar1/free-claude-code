"""The real Gemini SDK consumes and returns recovered native replay correctly."""

import base64
import json
import warnings
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import provider_manager_for_app
from tests.api.test_four_protocol_matrix import matrix_app
from tests.api.test_gemini_continuation_replay import native

# google-genai 3.0 has a docstring escape warning under Python 3.14.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message=r".*invalid escape sequence.*", category=SyntaxWarning
    )
    from google import genai
    from google.genai import types


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "case",
    ["text", "integer", "fractional", "nested", "defaults", "native_id", "null_fields"],
)
def test_sdk_recovers_and_replays_native_generation(streaming, case, monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    args = {
        "integer": {"amount": 9007199254740993},
        "fractional": {"amount": 1.5},
        "nested": {
            "values": [0.1, -0.0, {"small": 1e-100, "large": 1e100, "empty": None}]
        },
        "defaults": {},
        "native_id": {"amount": 1.5},
        "null_fields": {},
    }.get(case)
    call = {"name": "read", "args": args}
    if case == "native_id":
        call["id"] = "native-id"
    elif case == "null_fields":
        call.update(id=None, args=None)
    signed = {
        "role": "model",
        "parts": [
            {
                **({"text": "Done 🧠."} if args is None else {"functionCall": call}),
                "thoughtSignature": base64.b64encode(
                    b"original-native-signature"
                ).decode(),
            }
        ],
    }
    bodies = []

    def upstream(request):
        value = json.loads(request.content)
        bodies.append(deepcopy(value))
        if len(bodies) == 1:
            event = native("Prefix ")
        elif len(bodies) == 2:
            event = {
                "candidates": [{"index": 0, "content": signed, "finishReason": "STOP"}]
            }
        else:
            assert value["contents"][:-2] == bodies[1]["contents"]
            assert value["contents"][-2] == signed
            event = {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [{"text": "FOLLOWUP_OK"}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(event) + "\n\n",
        )

    app = matrix_app("gemini", upstream)
    with TestClient(app) as fcc:

        def bridge(request):
            return fcc.request(
                request.method,
                str(request.url),
                content=request.content,
                headers=request.headers,
            )

        with genai.Client(
            api_key="test",
            http_options=types.HttpOptions(
                base_url="http://testserver",
                api_version="v1beta",
                client_args={"transport": httpx.MockTransport(bridge)},
            ),
        ) as sdk:
            initial = types.Content(role="user", parts=[types.Part(text="hello")])
            config = types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name="read",
                                parameters_json_schema={
                                    "type": "object",
                                    "properties": {
                                        "amount": {"type": "number", "default": 1.5}
                                    },
                                },
                            )
                        ]
                    )
                ]
            )
            if streaming:
                responses = list(
                    sdk.models.generate_content_stream(
                        model="public", contents=[initial], config=config
                    )
                )
            else:
                responses = [
                    sdk.models.generate_content(
                        model="public", contents=[initial], config=config
                    )
                ]
            parts = [
                part
                for response in responses
                for candidate in response.candidates or []
                if candidate.content
                for part in candidate.content.parts or []
            ]
            assert (
                sum(
                    candidate.finish_reason is not None
                    for response in responses
                    for candidate in response.candidates or []
                )
                == 1
            )
            signatures = [
                part.thought_signature for part in parts if part.thought_signature
            ]
            assert len(signatures) == 1 and signatures[0].startswith(b"fcc:history:")
            assistant = types.Content(role="model", parts=parts)
            followup = types.Content(role="user", parts=[types.Part(text="next")])
            if args is not None:
                call = next(part.function_call for part in parts if part.function_call)
                assert call.args == (None if case == "null_fields" else args)
                if case == "defaults":
                    call.args = {"amount": 1.5}
                followup = types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="read", response={"result": "ok"}, id=call.id
                            )
                        )
                    ],
                )
            result = sdk.models.generate_content(
                model="public", contents=[initial, assistant, followup], config=config
            )
            assert result.text == "FOLLOWUP_OK"
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(bodies) == 3
