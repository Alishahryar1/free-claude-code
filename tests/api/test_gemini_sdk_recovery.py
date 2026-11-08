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
@pytest.mark.parametrize("client_api", ["models", "chat_curated", "chat_comprehensive"])
@pytest.mark.parametrize("empty_terminal", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "text",
        "integer",
        "fractional",
        "nested",
        "defaults",
        "native_id",
        "null_fields",
        "unsigned",
        "no_role",
    ],
)
def test_sdk_recovers_and_replays_native_generation(
    streaming, client_api, empty_terminal, case, monkeypatch
):
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
    if case == "unsigned":
        signed["parts"][0].pop("thoughtSignature")
    elif empty_terminal:
        signature = signed["parts"][0].pop("thoughtSignature")
        signed["parts"].append({"thoughtSignature": signature})
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
            if case == "unsigned":
                assert "Prefix " in json.dumps(value["contents"])
                assert "Done" in json.dumps(value["contents"])
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
        events = [deepcopy(event)]
        if len(bodies) == 2:
            if empty_terminal:
                content = events[0]["candidates"][0]["content"]
                events[0]["candidates"][0].pop("finishReason")
                terminal = {"index": 0, "finishReason": "STOP"}
                if case != "unsigned":
                    terminal["content"] = {
                        "role": "model",
                        "parts": [content["parts"].pop()],
                    }
                events.extend(
                    [
                        {"usageMetadata": {"totalTokenCount": 8}},
                        {"candidates": [terminal]},
                        {"usageMetadata": {"totalTokenCount": 9}},
                    ]
                )
            if case == "no_role":
                events[0]["candidates"][0]["content"].pop("role")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(item) + "\n\n" for item in events),
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
            chat = sdk.chats.create(model="public", config=config)
            if client_api != "models":
                responses = (
                    list(chat.send_message_stream("hello"))
                    if streaming
                    else [chat.send_message("hello")]
                )
            elif streaming:
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
            if case == "unsigned":
                assert not signatures
            else:
                assert len(signatures) == 1 and signatures[0].startswith(
                    b"fcc:history:"
                )
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
            if client_api == "chat_curated":
                assert followup.parts is not None
                result = chat.send_message(followup.parts[0])
            else:
                history = (
                    [initial, assistant]
                    if client_api == "models"
                    else chat.get_history()
                )
                result = sdk.models.generate_content(
                    model="public", contents=[*history, followup], config=config
                )
            assert result.text == "FOLLOWUP_OK"
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(bodies) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
async def test_sdk_chat_automatic_tool_result_preserves_recovered_history(
    async_client, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    bodies = []
    executed = []
    signed = {
        "role": "model",
        "parts": [
            {
                "functionCall": {"name": "read", "args": {"amount": 1.5}},
                "thoughtSignature": "c2lnbmF0dXJl",
            }
        ],
    }

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
            assert (
                value["contents"][: len(bodies[1]["contents"])] == bodies[1]["contents"]
            )
            assert value["contents"][len(bodies[1]["contents"])] == signed
            assert "functionResponse" in json.dumps(value["contents"])
            event = {
                "candidates": [
                    {
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

    def read(amount: float):
        """Read the requested amount."""
        executed.append(amount)
        return {"result": "ok"}

    app = matrix_app("gemini", upstream)
    with TestClient(app) as fcc:

        def bridge(request):
            return fcc.request(
                request.method,
                str(request.url),
                content=request.content,
                headers=request.headers,
            )

        async def async_bridge(request):
            response = bridge(request)
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        with genai.Client(
            api_key="test",
            http_options=types.HttpOptions(
                base_url="http://testserver",
                api_version="v1beta",
                client_args={"transport": httpx.MockTransport(bridge)},
                async_client_args={"transport": httpx.MockTransport(async_bridge)},
            ),
        ) as sdk:
            config = types.GenerateContentConfig(tools=[read])
            if async_client:
                async_chat = sdk.aio.chats.create(model="public", config=config)
                try:
                    responses = [
                        item
                        async for item in await async_chat.send_message_stream("hello")
                    ]
                    followup = await async_chat.send_message("next")
                finally:
                    await sdk.aio.aclose()
            else:
                chat = sdk.chats.create(model="public", config=config)
                responses = list(chat.send_message_stream("hello"))
                followup = chat.send_message("next")
            assert any(
                part.text == "FOLLOWUP_OK"
                for response in responses
                for candidate in response.candidates or []
                if candidate.content
                for part in candidate.content.parts or []
            )
            assert followup.text == "FOLLOWUP_OK"
            assert executed == [1.5]
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(bodies) == 4
