"""Source reasoning remains native independently of Gemini presentation."""

from dataclasses import replace

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.history_replay import readable_reasoning
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.deepseek.client import _PROFILE, DeepSeekChatBehavior
from tests.providers.test_gemini_review_regressions import transport
from tests.providers.test_gemini_transport import chat_response, collect, gemini_request


@pytest.mark.asyncio
@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "policy",
    [
        ReasoningPolicy.provider_default(),
        ReasoningPolicy.on(),
        ReasoningPolicy.prefer_off(),
        ReasoningPolicy.off(),
    ],
    ids=["default", "on", "prefer_off", "off"],
)
async def test_native_fallback_field_and_computation_policy(
    visible, structured, policy
):
    profile = replace(
        _PROFILE,
        reasoning_delta_field="reasoning",
        reasoning_delta_fallback_field="reasoning_content",
        structured_reasoning_details=structured,
    )
    details = [{"type": "reasoning.encrypted", "data": "signed", "index": 0}]

    def reply(request):
        return chat_response(
            [
                {
                    "reasoning_content": "native thought",
                    **({"reasoning_details": details} if structured else {}),
                },
                {"content": "answer"},
            ]
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        value, _ = await collect(
            transport(client, DeepSeekChatBehavior(profile)).stream_gemini(
                gemini_request(
                    generationConfig={"thinkingConfig": {"includeThoughts": visible}}
                ),
                reasoning=policy,
            )
        )
    parts = value["candidates"][0]["content"]["parts"]
    bundles = [
        decode_bundle(part["thoughtSignature"])
        for part in parts
        if "thoughtSignature" in part
    ]
    records = [record for bundle in bundles if bundle for record in bundle.records]
    assert bool(records) == (policy.output_enabled or structured)
    if policy.output_enabled:
        assert records[0].native["reasoning_content"] == "native thought"
        assert "reasoning" not in records[0].native
    elif structured:
        assert records[0].native == {"reasoning_details": details}
    assert any(part.get("thought") for part in parts) == (
        visible and policy.output_enabled
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["full thought", "different native"])
async def test_mixed_native_and_structured_reasoning_retains_distinct_text(native):
    profile = replace(_PROFILE, structured_reasoning_details=True)

    def reply(request):
        return chat_response(
            [
                {"reasoning_content": native},
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "index": 0,
                            "text": "full thought",
                            "signature": "signed",
                        }
                    ]
                },
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "index": 1,
                            "summary": "short summary",
                        }
                    ]
                },
                {"content": "answer"},
            ]
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        value, _ = await collect(
            transport(client, DeepSeekChatBehavior(profile)).stream_gemini(
                gemini_request()
            )
        )
    bundle = decode_bundle(
        value["candidates"][0]["content"]["parts"][-1]["thoughtSignature"]
    )
    assert bundle is not None
    assert bundle.records[0].native["reasoning_content"] == native
    readable = readable_reasoning(bundle.records[0].native)
    assert readable.count(("full thought", False)) == 1
    assert ("short summary", True) in readable
    if native != "full thought":
        assert (native, False) in readable


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_equal_independent_groups_are_not_deduplicated(structured):
    profile = replace(_PROFILE, structured_reasoning_details=structured)

    def reply(request):
        return chat_response(
            [
                {"reasoning_content": "same thought"},
                {"content": "one"},
                {"reasoning_content": "same thought"},
                {"content": "two"},
            ]
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        value, _ = await collect(
            transport(client, DeepSeekChatBehavior(profile)).stream_gemini(
                gemini_request()
            )
        )
    bundle = decode_bundle(
        value["candidates"][0]["content"]["parts"][-1]["thoughtSignature"]
    )
    assert bundle is not None
    assert [record.native for record in bundle.records] == [
        {"reasoning_content": "same thought"}
    ] * 2


@pytest.mark.asyncio
async def test_parsed_think_tags_do_not_become_native_reasoning_fields():
    def reply(request):
        return chat_response([{"content": "<think>parsed thought</think>answer"}])

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        value, _ = await collect(
            transport(client, DeepSeekChatBehavior(_PROFILE)).stream_gemini(
                gemini_request(
                    generationConfig={"thinkingConfig": {"includeThoughts": True}}
                )
            )
        )
    parts = value["candidates"][0]["content"]["parts"]
    assert any(part.get("text") == "parsed thought" for part in parts)
    assert not any("thoughtSignature" in part for part in parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("visible", [False, True])
async def test_abandoned_invisible_attempt_does_not_publish_its_reasoning(visible):
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        prefix = "discarded" if calls == 1 else "winning"
        return chat_response(
            [
                {"reasoning_content": prefix + " thought"},
                {"content": prefix + " answer"},
            ],
            finish=None if calls == 1 else "stop",
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        value, chunks = await collect(
            transport(client, DeepSeekChatBehavior(_PROFILE)).stream_gemini(
                gemini_request(
                    generationConfig={"thinkingConfig": {"includeThoughts": visible}}
                )
            )
        )
    assert calls == 2
    assert "discarded" not in "".join(chunks)
    bundle = decode_bundle(
        value["candidates"][0]["content"]["parts"][-1]["thoughtSignature"]
    )
    assert bundle is not None
    assert [record.native for record in bundle.records] == [
        {"reasoning_content": "winning thought"}
    ]
