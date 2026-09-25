"""Executable contracts for Pi's bundled TypeScript extension."""

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Literal

import pytest

from free_claude_code.application.reasoning import client_reasoning_policy
from free_claude_code.cli.launchers.pi import pi_extension_path
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.github_copilot.types import CopilotEgress
from tests.cli.node_diagnostics import run_node_diagnostics
from tests.providers.test_github_copilot_provider import Harness, collect


# Temporary hosted Windows timing probe. Remove after identifying the stall.
def _run_pi_probe(node: str, script: str, *arguments: str):
    prelude = """
import { writeSync } from "node:fs";
function mark(stage) {
    writeSync(2, JSON.stringify({stage, node: process.version,
        elapsedMs: performance.now(), at: Date.now()}) + "\\n");
}
mark("script-entered");
process.once("beforeExit", () => mark("before-exit"));
process.once("exit", () => mark("exit"));
"""
    started = monotonic()
    command = [
        node,
        "--experimental-strip-types",
        "--input-type=module",
        "--eval",
        prelude + script + '\nmark("script-complete");',
        *arguments,
    ]
    if phase := os.environ.get("FCC_PI_DIAGNOSTIC_PHASE"):
        return run_node_diagnostics(command, phase)
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        print(
            f"Pi Node probe timed out after {monotonic() - started:.3f}s; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            flush=True,
        )
        raise


def test_pi_sends_current_conversation_id_only_on_fcc_requests() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    script = """
const { default: extension } = await import(process.argv[1]);
mark("extension-imported");
const handlers = new Map();
process.env.FCC_PI_BASE_URL = "http://fcc.invalid";
process.env.FCC_PI_API_KEY = "test-key";
globalThis.fetch = async () => {
    mark("fake-fetch-entered");
    const response = new Response(JSON.stringify({
        object: "list",
        data: [{ id: "opencode_zen/test", provider_model_ref: "opencode_zen/test" }],
    }));
    mark("fake-response-created");
    return response;
};
await extension({
    registerProvider() {},
    on(event, handler) { handlers.set(event, handler); },
});
mark("extension-registered");
const results = [];
for (const [provider, session] of [
    ["free-claude-code", "conversation-a"],
    ["free-claude-code", "conversation-a"],
    ["free-claude-code", "conversation-b"],
    ["free-claude-code", "conversation-a"],
    ["another-provider", "another-session"],
]) {
    const headers = { "x-existing": "preserve" };
    await handlers.get("before_provider_headers")?.({ headers }, {
        model: { provider },
        sessionManager: { getSessionId() { return session; } },
    });
    results.push(headers);
}
console.log(JSON.stringify(results));
"""
    result = _run_pi_probe(node, script, pi_extension_path().as_uri())
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"x-existing": "preserve", "x-opencode-session": "conversation-a"},
        {"x-existing": "preserve", "x-opencode-session": "conversation-a"},
        {"x-existing": "preserve", "x-opencode-session": "conversation-b"},
        {"x-existing": "preserve", "x-opencode-session": "conversation-a"},
        {"x-existing": "preserve"},
    ]


def _pi_request_payloads(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = """
const { default: extension } = await import(process.argv[1]);
mark("extension-imported");
const handlers = new Map();
let level;
process.env.FCC_PI_BASE_URL = "http://fcc.invalid";
process.env.FCC_PI_API_KEY = "test-key";
globalThis.fetch = async () => {
    mark("fake-fetch-entered");
    const response = new Response(JSON.stringify({
        object: "list",
        data: [{ id: "github_copilot/gpt-5.6-luna", provider_model_ref: "github_copilot/gpt-5.6-luna" }],
    }));
    mark("fake-response-created");
    return response;
};
await extension({
    registerProvider() {},
    on(event, handler) { handlers.set(event, handler); },
    getThinkingLevel() { return level; },
});
mark("extension-registered");
const results = [];
for (const entry of JSON.parse(process.argv[2])) {
    level = entry.level;
    const payload = structuredClone(entry.payload);
    const handler = handlers.get("before_provider_request");
    const replacement = await handler?.({ payload }, { model: entry.model });
    results.push(replacement ?? payload);
}
console.log(JSON.stringify(results));
"""
    result = _run_pi_probe(
        node, script, pi_extension_path().as_uri(), json.dumps(cases)
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_pi_sends_selected_effort_without_an_exact_budget_or_native_thinking_mode() -> (
    None
):
    model = {
        "id": "github_copilot/gpt-5.6-luna",
        "provider": "free-claude-code",
        "api": "anthropic-messages",
        "reasoning": True,
    }
    payload = {
        "model": model["id"],
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16384,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 8192, "display": "summarized"},
        "output_config": {"format": {"type": "text"}},
    }
    levels = ["minimal", "low", "medium", "high", "xhigh", "max"]
    results = _pi_request_payloads(
        [{"model": model, "payload": payload, "level": level} for level in levels]
    )
    for level, result in zip(levels, results, strict=True):
        assert result == payload | {
            "thinking": {"display": "summarized"},
            "output_config": {"format": {"type": "text"}, "effort": level},
        }


def test_pi_preserves_disabled_thinking_and_requests_outside_its_budget_translation() -> (
    None
):
    model = {
        "id": "github_copilot/gpt-5.6-luna",
        "provider": "free-claude-code",
        "api": "anthropic-messages",
        "reasoning": True,
    }
    payload = {
        "model": model["id"],
        "thinking": {"type": "enabled", "budget_tokens": 8192},
    }
    cases: list[dict[str, object]] = [
        {
            "model": model,
            "level": "off",
            "payload": payload | {"thinking": {"type": "disabled"}},
        },
        {"model": model, "level": "high", "payload": {"model": model["id"]}},
        {
            "model": model,
            "level": "high",
            "payload": payload
            | {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        },
        {
            "model": model | {"provider": "anthropic"},
            "level": "high",
            "payload": payload,
        },
        {
            "model": model | {"api": "openai-responses"},
            "level": "high",
            "payload": payload,
        },
        {"model": model | {"reasoning": False}, "level": "high", "payload": payload},
        {
            "model": model,
            "level": "high",
            "payload": payload | {"model": "another-model"},
        },
    ]
    assert _pi_request_payloads(cases) == [case["payload"] for case in cases]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "egress,adaptive",
    [
        (CopilotEgress.CHAT, "optional"),
        (CopilotEgress.RESPONSES, "optional"),
        (CopilotEgress.MESSAGES, "required"),
        (CopilotEgress.MESSAGES, "unsupported"),
    ],
)
async def test_pi_named_effort_reaches_each_copilot_transport(
    tmp_path: Path,
    egress: CopilotEgress,
    adaptive: Literal["optional", "required", "unsupported"],
) -> None:
    model_id = "github_copilot/gpt-5.6-luna"
    payload = _pi_request_payloads(
        [
            {
                "model": {
                    "id": model_id,
                    "provider": "free-claude-code",
                    "api": "anthropic-messages",
                    "reasoning": True,
                },
                "level": "high",
                "payload": {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 16384,
                    "stream": True,
                    "thinking": {"type": "enabled", "budget_tokens": 8192},
                },
            }
        ]
    )[0]
    harness = Harness(tmp_path, egress)
    advertised_model = harness.runtime.available[0]
    harness.runtime.available = (
        replace(
            advertised_model,
            messages=replace(advertised_model.messages, adaptive_thinking=adaptive),
        ),
    )
    try:
        request = MessagesRequest.model_validate(
            payload | {"model": harness.runtime.name}
        )
        await collect(
            harness.provider.stream_messages(
                request, reasoning=client_reasoning_policy(request)
            )
        )
        body = json.loads(harness.seen[0].content)
        if egress is CopilotEgress.CHAT:
            assert body["reasoning_effort"] == "high"
        elif egress is CopilotEgress.RESPONSES:
            assert body["reasoning"]["effort"] == "high"
        elif adaptive == "required":
            assert body["thinking"] == {"type": "adaptive"}
            assert body["output_config"]["effort"] == "high"
        else:
            assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    finally:
        await harness.close()


def test_pi_extension_projects_known_capabilities_and_preserves_unknown_defaults() -> (
    None
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    payload = {
        "object": "list",
        "data": [
            {
                "id": "provider/vision-reasoning",
                "provider_model_ref": "provider/vision-reasoning",
                "supportsReasoning": True,
                "inputModalities": ["text", "image"],
                "contextWindow": 131072,
                "maxCompletionTokens": 8192,
            },
            {
                "id": "claude-3-freecc-no-thinking/provider/text-only",
                "provider_model_ref": "provider/text-only",
                "supportsReasoning": False,
                "inputModalities": ["text"],
                "contextWindow": 65536,
            },
            {
                "id": "provider/unknown ",
                "provider_model_ref": "provider/unknown ",
            },
        ],
    }
    script = """
const { projectFccModels } = await import(process.argv[1]);
mark("extension-imported");
const payload = JSON.parse(process.argv[2]);
console.log(JSON.stringify(projectFccModels(payload)));
"""

    result = _run_pi_probe(
        node, script, pi_extension_path().as_uri(), json.dumps(payload)
    )

    assert result.returncode == 0, result.stderr
    projected = json.loads(result.stdout)
    assert [model["id"] for model in projected] == [
        "provider/vision-reasoning",
        "claude-3-freecc-no-thinking/provider/text-only",
        "provider/unknown ",
    ]
    assert projected[-1]["name"] == "provider/unknown "
    assert [(model["reasoning"], model["input"]) for model in projected] == [
        (True, ["text", "image"]),
        (False, ["text"]),
        (True, ["text"]),
    ]
    assert [(model["contextWindow"], model["maxTokens"]) for model in projected] == [
        (131072, 8192),
        (65536, 16384),
        (128000, 16384),
    ]
