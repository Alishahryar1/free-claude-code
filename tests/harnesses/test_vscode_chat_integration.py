import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.core.json_types import load_json_value
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses import vscode_chat_integration as vscode
from tests.wire_payload import array_payload, object_array, object_payload

URL = "http://127.0.0.1:8082"
MODEL = CatalogModel("nim/model", "nim/model", "NIM Model", None)
native_config_path = vscode.config_path


def test_connect_refresh_disconnect_preserves_other_groups_and_options(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chatLanguageModels.json"
    other = {"name": "Other", "vendor": "customendpoint", "apiKey": "other"}
    path.write_text(json.dumps([other]))
    assert not vscode.status(path)["connected"]
    assert vscode.configure(path, URL, "secret", (MODEL,))
    data = load_json_value(path.read_text())
    assert array_payload(data)[0] == other
    assert (
        object_array(object_array(data)[1]["models"])[0]["url"] == URL + "/v1/messages"
    )
    object_payload(array_payload(data)[1]).update(
        name="My FCC", settings={"keep": True}, url="old", apiKey="old"
    )
    path.write_text(json.dumps(data))
    assert vscode.configure(path, URL, "new", (MODEL,), only_existing=True)
    saved = load_json_value(path.read_text())
    assert object_array(saved)[1]["settings"] == {"keep": True}
    assert object_array(saved)[1]["name"] == "My FCC"
    assert "url" not in object_array(saved)[1]
    assert "apiKey" not in object_array(saved)[1]
    assert object_array(object_payload(array_payload(saved)[1])["models"])[0][
        "requestHeaders"
    ] == {"x-api-key": "new"}
    before = path.stat().st_mtime_ns
    assert not vscode.configure(path, URL, "new", (MODEL,), only_existing=True)
    assert path.stat().st_mtime_ns == before
    assert vscode.status(path)["connected"] is True
    assert "new" not in json.dumps(vscode.status(path))
    vscode.disconnect(path)
    assert load_json_value(path.read_text()) == [other]
    assert not vscode.configure(path, URL, "new", (MODEL,), only_existing=True)


def test_missing_file_read_refresh_disconnect_never_creates_it(tmp_path: Path) -> None:
    path = tmp_path / "missing/chatLanguageModels.json"
    assert vscode.status(path) == {
        "connected": False,
        "paths": {"vscode_models": str(path.resolve())},
    }
    assert not vscode.configure(path, URL, "secret", (MODEL,), only_existing=True)
    vscode.disconnect(path)
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "source", ["{bad", "{}", "[1]", '[{"x":NaN}]', '[{"x":1,"x":2}]']
)
def test_invalid_document_is_never_overwritten(tmp_path: Path, source: str) -> None:
    path = tmp_path / "models.json"
    path.write_text(source)
    for operation in (
        lambda: vscode.status(path),
        lambda: vscode.disconnect(path),
        lambda: vscode.configure(path, URL, "secret", (MODEL,)),
    ):
        with pytest.raises(ValueError):
            operation()
        assert path.read_text() == source


def test_conflict_does_not_claim_an_unmarked_group(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    source = '[{"name":"Free Claude Code","vendor":"customendpoint"}]'
    path.write_text(source)
    with pytest.raises(ValueError):
        vscode.configure(path, URL, "secret", (MODEL,))
    assert path.read_text() == source
    assert not vscode.status(path)["connected"]


def test_duplicate_markers_fail_but_disconnect_recovers_name_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.json"
    vscode.configure(path, URL, "secret", (MODEL,))
    owned = array_payload(load_json_value(path.read_text()))[0]
    path.write_text(json.dumps([owned, owned]))
    with pytest.raises(ValueError):
        vscode.disconnect(path)
    other = {"vendor": "customendpoint", "name": object_payload(owned)["name"]}
    path.write_text(json.dumps([owned, other]))
    with pytest.raises(ValueError):
        vscode.configure(path, URL, "secret", (MODEL,))
    assert vscode.status(path)["connected"]
    vscode.disconnect(path)
    assert load_json_value(path.read_text()) == [other]


@pytest.mark.parametrize(
    "context,output,expected",
    [
        (None, None, (118080, 81920)),
        (8000, None, (4000, 4000)),
        (100000, 20000, (80000, 20000)),
        (None, 1000, (199000, 1000)),
        (8192, 8192, (4096, 4096)),
        (1, 0, (118080, 81920)),
    ],
)
def test_token_allocation_and_capabilities(
    context: int | None, output: int | None, expected: tuple[int, int]
) -> None:
    model = replace(MODEL, context_window_tokens=context, max_output_tokens=output)
    entry = vscode.model_entry(model, URL)
    assert (entry["maxInputTokens"], entry["maxOutputTokens"]) == expected
    assert entry["id"] == MODEL.wire_slug
    assert entry["toolCalling"] is True
    assert entry["vision"] is False
    assert "thinking" not in entry
    assert (
        vscode.model_entry(
            replace(model, input_modalities=frozenset({ModelInputModality.IMAGE})), URL
        )["vision"]
        is True
    )


def test_comment_input_and_order_are_preserved_semantically(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text('[/* comment */ {"vendor":"other", "name":"Other",},]')
    second = replace(MODEL, wire_slug="custom/z", display_name="Custom Z")
    vscode.configure(path, URL, "secret", (MODEL, second))
    assert [
        m["id"]
        for m in object_array(
            object_array(load_json_value(path.read_text()))[1]["models"]
        )
    ] == [
        MODEL.wire_slug,
        second.wire_slug,
    ]


@pytest.mark.parametrize("support", [True, None, False])
def test_native_effort_capability(support: bool | None) -> None:
    entry = vscode.model_entry(replace(MODEL, supports_reasoning=support), URL)
    if support is False:
        assert "supportsReasoningEffort" not in entry
        assert "defaultReasoningEffort" not in entry
    else:
        assert entry["supportsReasoningEffort"] == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        assert entry["defaultReasoningEffort"] == "medium"


def test_refresh_preserves_native_preferences_and_replaces_generated_capabilities(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.json"
    removed = replace(MODEL, wire_slug="nim/removed")
    vscode.configure(path, URL, "old", (MODEL, removed))
    data = load_json_value(path.read_text())
    preferences = {MODEL.wire_slug: {"reasoningEffort": "high"}}
    object_array(data)[0]["settings"] = preferences
    object_array(object_array(data)[0]["models"])[0]["maxOutputTokens"] = 777
    path.write_text(json.dumps(data))
    vscode.configure(path, URL, "new", (replace(MODEL, supports_reasoning=False),))
    saved = array_payload(load_json_value(path.read_text()))[0]
    assert object_payload(saved)["settings"] == preferences
    assert [m["id"] for m in object_array(object_payload(saved)["models"])] == [
        MODEL.wire_slug
    ]
    assert object_array(object_payload(saved)["models"])[0]["maxOutputTokens"] == 81920
    assert (
        "supportsReasoningEffort"
        not in object_array(object_payload(saved)["models"])[0]
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
@pytest.mark.parametrize("already_connected", [False, True])
def test_connect_and_unchanged_refresh_make_existing_file_private(
    tmp_path: Path, already_connected: bool
) -> None:
    path = tmp_path / "models.json"
    path.write_text("[]")
    if already_connected:
        vscode.configure(path, URL, "secret", (MODEL,))
    path.chmod(0o644)
    before = path.stat().st_mtime_ns
    assert vscode.configure(path, URL, "secret", (MODEL,)) is not already_connected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    if already_connected:
        assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "platform,env,tail",
    [
        ("win32", "APPDATA", "Code/User/chatLanguageModels.json"),
        ("linux", "XDG_CONFIG_HOME", "Code/User/chatLanguageModels.json"),
        (
            "darwin",
            None,
            "Library/Application Support/Code/User/chatLanguageModels.json",
        ),
    ],
)
def test_native_config_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    env: str | None,
    tail: str,
) -> None:
    from free_claude_code.harnesses import claude_integration

    monkeypatch.setattr(claude_integration.sys, "platform", platform)
    monkeypatch.setattr(claude_integration.Path, "home", lambda: tmp_path)
    if env:
        monkeypatch.setenv(env, str(tmp_path))
    assert native_config_path() == tmp_path / tail


@pytest.mark.parametrize(
    "effort,preference,no_thinking,expected",
    [
        (effort, "client", False, effort)
        for effort in (None, "none", "low", "medium", "high", "xhigh", "max")
    ]
    + [
        ("high", "off", False, "none"),
        ("none", "high", False, "high"),
        ("high", "high", True, "none"),
    ],
)
def test_native_effort_uses_shared_routing_and_provider_encoders(
    effort: str | None, preference: str, no_thinking: bool, expected: str | None
) -> None:
    from free_claude_code.application.routing import ModelRouter
    from free_claude_code.config.settings import Settings
    from free_claude_code.core.anthropic.models import MessagesRequest
    from free_claude_code.providers.anthropic_messages.request_policy import (
        MessagesModelCapabilities,
        resolve_messages_options,
    )
    from free_claude_code.providers.nvidia_nim.request_options import (
        build_nim_request_body,
    )

    settings = Settings(REASONING_POLICY=preference)
    model_id = "nvidia_nim/test"
    if no_thinking:
        model_id = "claude-3-freecc-no-thinking/" + model_id
    entry = vscode.model_entry(replace(MODEL, wire_slug=model_id), URL)
    request = MessagesRequest.model_validate(
        {
            "model": entry["id"],
            "max_tokens": entry["maxOutputTokens"],
            "messages": [{"role": "user", "content": "hello"}],
            **({"output_config": {"effort": effort}} if effort is not None else {}),
        }
    )
    routed = ModelRouter(settings).resolve_messages_request(request)
    body = build_nim_request_body(routed.request, reasoning=routed.reasoning)
    assert body.get("reasoning_effort") == expected
    assert body["max_tokens"] == entry["maxOutputTokens"]
    native = resolve_messages_options(
        model=routed.request.model,
        max_tokens=routed.request.max_tokens,
        reasoning=routed.reasoning,
        capabilities=MessagesModelCapabilities(
            adaptive_thinking="unsupported", supports_output_effort=False
        ),
    )
    assert native.max_tokens == entry["maxOutputTokens"]
    if routed.reasoning.requests_reasoning:
        assert native.thinking is not None
        budget = routed.reasoning.numeric_budget_tokens
        assert budget is not None
        assert native.thinking["budget_tokens"] == max(1024, budget)
    else:
        assert native.thinking is None or native.thinking["type"] == "disabled"


def test_unchanged_permission_repair_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.json"
    vscode.configure(path, URL, "secret", (MODEL,))
    before = path.read_bytes()

    def fail(_path):
        raise PermissionError("permission repair failed")

    monkeypatch.setattr(vscode, "ensure_private_permissions", fail)
    with pytest.raises(PermissionError, match="permission repair"):
        vscode.configure(path, URL, "secret", (MODEL,))
    assert path.read_bytes() == before
