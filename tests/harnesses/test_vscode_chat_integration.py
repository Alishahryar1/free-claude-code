import json
from dataclasses import replace

import pytest

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses import vscode_chat_integration as vscode

URL = "http://127.0.0.1:8082"
MODEL = CatalogModel("nim/model", "nim/model", "NIM Model", None)
native_config_path = vscode.config_path


def test_connect_refresh_disconnect_preserves_other_groups_and_options(tmp_path):
    path = tmp_path / "chatLanguageModels.json"
    other = {"name": "Other", "vendor": "customendpoint", "apiKey": "other"}
    path.write_text(json.dumps([other]))
    assert not vscode.status(path)["connected"]
    assert vscode.configure(path, URL, "secret", (MODEL,))
    data = json.loads(path.read_text())
    assert data[0] == other
    assert data[1]["models"][0]["url"] == URL + "/v1/messages"
    data[1].update(name="My FCC", settings={"keep": True}, url="old", apiKey="old")
    path.write_text(json.dumps(data))
    assert vscode.configure(path, URL, "new", (MODEL,), only_existing=True)
    saved = json.loads(path.read_text())
    assert saved[1]["settings"] == {"keep": True}
    assert saved[1]["name"] == "My FCC"
    assert "url" not in saved[1]
    assert "apiKey" not in saved[1]
    assert saved[1]["models"][0]["requestHeaders"] == {"x-api-key": "new"}
    before = path.stat().st_mtime_ns
    assert not vscode.configure(path, URL, "new", (MODEL,), only_existing=True)
    assert path.stat().st_mtime_ns == before
    assert vscode.status(path)["connected"] is True
    assert "new" not in json.dumps(vscode.status(path))
    vscode.disconnect(path)
    assert json.loads(path.read_text()) == [other]
    assert not vscode.configure(path, URL, "new", (MODEL,), only_existing=True)


def test_missing_file_read_refresh_disconnect_never_creates_it(tmp_path):
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
def test_invalid_document_is_never_overwritten(tmp_path, source):
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


def test_conflict_does_not_claim_an_unmarked_group(tmp_path):
    path = tmp_path / "models.json"
    source = '[{"name":"Free Claude Code","vendor":"customendpoint"}]'
    path.write_text(source)
    with pytest.raises(ValueError):
        vscode.configure(path, URL, "secret", (MODEL,))
    assert path.read_text() == source
    assert not vscode.status(path)["connected"]


def test_duplicate_markers_fail_but_disconnect_recovers_name_collision(tmp_path):
    path = tmp_path / "models.json"
    vscode.configure(path, URL, "secret", (MODEL,))
    owned = json.loads(path.read_text())[0]
    path.write_text(json.dumps([owned, owned]))
    with pytest.raises(ValueError):
        vscode.disconnect(path)
    other = {"vendor": "customendpoint", "name": owned["name"]}
    path.write_text(json.dumps([owned, other]))
    with pytest.raises(ValueError):
        vscode.configure(path, URL, "secret", (MODEL,))
    assert vscode.status(path)["connected"]
    vscode.disconnect(path)
    assert json.loads(path.read_text()) == [other]


@pytest.mark.parametrize(
    "context,output,expected",
    [
        (None, None, (28672, 4096)),
        (8000, None, (4000, 4000)),
        (100000, 20000, (80000, 20000)),
        (None, 1000, (31768, 1000)),
        (8192, 8192, (4096, 4096)),
        (1, 0, (28672, 4096)),
    ],
)
def test_token_allocation_and_capabilities(context, output, expected):
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


def test_comment_input_and_order_are_preserved_semantically(tmp_path):
    path = tmp_path / "models.json"
    path.write_text('[/* comment */ {"vendor":"other", "name":"Other",},]')
    second = replace(MODEL, wire_slug="custom/z", display_name="Custom Z")
    vscode.configure(path, URL, "secret", (MODEL, second))
    assert [m["id"] for m in json.loads(path.read_text())[1]["models"]] == [
        MODEL.wire_slug,
        second.wire_slug,
    ]


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
def test_native_config_paths(tmp_path, monkeypatch, platform, env, tail):
    from free_claude_code.harnesses import claude_integration

    monkeypatch.setattr(claude_integration.sys, "platform", platform)
    monkeypatch.setattr(claude_integration.Path, "home", lambda: tmp_path)
    if env:
        monkeypatch.setenv(env, str(tmp_path))
    assert native_config_path() == tmp_path / tail
