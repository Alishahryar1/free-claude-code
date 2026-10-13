"""FCC compares and changes only its native DSH Desktop entries."""

import copy
import json

import pytest
from ruamel.yaml import YAML

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from free_claude_code.harnesses import dsh_files

URL = "http://127.0.0.1:8182"
TOKEN = "desktop-test-token"
ROUTE = "free-claude-code"
REF = desktop.DSH_DESKTOP_API_KEY


def catalog(name="fixture/model"):
    return ModelCatalog((CatalogModel(name, name, name, True),), name)


def read(path):
    return YAML(typ="rt").load(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(value, stream)


def config(rows, identity):
    return next(
        row["config"]
        for row in reversed(rows)
        if row.get("id") == identity and "config" in row
    )


@pytest.fixture
def home(tmp_path):
    root = tmp_path / "dsh"
    profile = root / "profiles/desktop"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text('{"dsh":{"profile":{"bundles":[]}}}')
    (profile / "cordis.patch.yml").write_text("# native settings\n[]\n")
    return root


def connect(home, token=TOKEN, model="fixture/model"):
    return desktop.configure(
        home, URL, token, catalog(model), provider_progress_timeout=600
    )


def status(home):
    return desktop.status(home, URL, TOKEN, catalog(), provider_progress_timeout=600)


def test_connect_status_disconnect_and_reconnect(home):
    assert not status(home)["connected"]
    result = connect(home)
    assert result["connected"]
    patch = home / "profiles/desktop/cordis.patch.yml"
    provider = config(read(patch), "llm-pi-ai")["providers"][ROUTE]
    assert provider["api"] == "openai-responses"
    assert provider["baseURL"] == URL + "/v1"
    assert provider["models"][0]["id"] == "fixture/model"
    assert provider["apiKeyEnv"] == REF
    assert read(home / ".credentials.yaml")["refs"][REF] == TOKEN
    assert config(read(patch), "agent-default-model") == {
        "provider": ROUTE,
        "model": "fixture/model",
    }
    assert TOKEN not in patch.read_text() + json.dumps(result)
    assert status(home)["connected"]
    assert not connect(home)["changed"]
    assert not desktop.disconnect(home)["connected"]
    assert ROUTE not in patch.read_text()
    assert REF not in (home / ".credentials.yaml").read_text()
    assert not status(home)["connected"]
    assert not desktop.disconnect(home)["connected"]
    assert connect(home)["connected"]


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "baseURL",
        "models",
        "apiKeyEnv",
        "streamIdleTimeoutMs",
        "displayName",
        "api",
        "defaultInput",
        "retryPolicy",
        "default",
        "token",
        "missing-token",
    ],
)
def test_status_requires_all_fcc_keys_and_connect_replaces_them(home, field):
    connect(home)
    patch = home / "profiles/desktop/cordis.patch.yml"
    credentials = home / ".credentials.yaml"
    rows = read(patch)
    providers = config(rows, "llm-pi-ai")["providers"]
    if field == "provider":
        del providers[ROUTE]
    elif field == "default":
        config(rows, "agent-default-model")["model"] = "other"
    elif field in {"token", "missing-token"}:
        data = read(credentials)
        if field == "token":
            data["refs"][REF] = "edited"
        else:
            del data["refs"][REF]
        write(credentials, data)
    else:
        providers[ROUTE][field] = [] if field == "models" else "changed"
    write(patch, rows)
    assert not status(home)["connected"]
    assert connect(home)["connected"]
    assert status(home)["connected"]


def test_status_compares_current_catalog_url_token_and_timeout(home):
    connect(home)
    for url, token, models, timeout in [
        (URL + "-changed", TOKEN, catalog(), 600),
        (URL, "rotated", catalog(), 600),
        (URL, TOKEN, catalog("fixture/new"), 600),
        (URL, TOKEN, catalog(), 30),
    ]:
        assert not desktop.status(
            home, url, token, models, provider_progress_timeout=timeout
        )["connected"]


def test_other_settings_accounts_comments_tags_and_peer_files_are_preserved(home):
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text("""# keep native settings
- id: unrelated
  config:
    computed: !!js "ctx.get('something')"
- id: llm-pi-ai
  config:
    anotherSetting: unchanged
    providers:
      other:
        apiKeyEnv: OTHER_KEY # keep comment
        models: [unchanged]
""")
    credentials = home / ".credentials.yaml"
    credentials.write_text(
        'version: 1\nrefs:\n  OTHER_KEY: "keep-secret" # keep key\nrecords: {native: unchanged}\n'
    )
    peer = home / "profiles/web/cordis.patch.yml"
    peer.parent.mkdir()
    peer.write_text("unreadable unrelated profile, even if it names " + REF)
    before = read(patch)
    connect(home)
    connect(home, token="rotated", model="fixture/new")
    assert read(credentials)["refs"] == {"OTHER_KEY": "keep-secret", REF: "rotated"}
    desktop.disconnect(home)
    after = read(patch)
    assert after[1] == before[1]
    assert after[0]["config"]["computed"].value == before[0]["config"]["computed"].value
    assert after[0]["config"]["computed"].tag == before[0]["config"]["computed"].tag
    assert "# keep native settings" in patch.read_text()
    assert "# keep comment" in patch.read_text()
    assert read(credentials) == {
        "version": 1,
        "refs": {"OTHER_KEY": "keep-secret"},
        "records": {"native": "unchanged"},
    }
    assert "# keep key" in credentials.read_text()
    assert peer.read_text() == "unreadable unrelated profile, even if it names " + REF


def test_disconnect_removes_all_fcc_copies_and_preserves_other_default(home):
    connect(home)
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = read(patch)
    rows.append(copy.deepcopy(rows[0]))
    rows.append(
        {
            "id": "agent-default-model",
            "config": {"provider": "other", "model": "other-model"},
        }
    )
    write(patch, rows)
    desktop.disconnect(home)
    assert ROUTE not in patch.read_text()
    assert config(read(patch), "agent-default-model") == {
        "provider": "other",
        "model": "other-model",
    }


@pytest.mark.parametrize("operation", ["connect", "disconnect"])
def test_partial_write_is_disconnected_and_connect_can_repair(
    home, monkeypatch, operation
):
    if operation == "disconnect":
        connect(home)
    original = dsh_files.write_yaml

    def fail(path, *args, **kwargs):
        if path == home / ".credentials.yaml":
            raise PermissionError("test failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(dsh_files, "write_yaml", fail)
        with pytest.raises(PermissionError):
            connect(home) if operation == "connect" else desktop.disconnect(home)
    assert not status(home)["connected"]
    assert connect(home)["connected"]
    assert not desktop.disconnect(home)["connected"]


def test_absent_home_status_disconnect_and_refresh_create_nothing(tmp_path):
    home = tmp_path / "absent"
    assert not status(home)["connected"]
    assert not desktop.disconnect(home)["connected"]
    assert not desktop.refresh_connected(
        home, URL, TOKEN, catalog(), provider_progress_timeout=600
    )
    assert not home.exists()


def test_invalid_yaml_does_not_modify_files_or_expose_secrets(home):
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text("secret: [private-token")
    before = patch.read_bytes()
    with pytest.raises(desktop.DshConfigError, match="Invalid YAML") as error:
        connect(home)
    assert "private-token" not in str(error.value)
    assert patch.read_bytes() == before
    assert not (home / ".credentials.yaml").exists()


def test_connect_honors_native_profile_lock(home, monkeypatch):
    lock = home / "profiles/desktop/package.json.lock"
    lock.write_text("native writer")
    original = dsh_files.file_lock
    monkeypatch.setattr(
        dsh_files, "file_lock", lambda path, **kwargs: original(path, wait=0)
    )
    with pytest.raises(TimeoutError):
        connect(home)
    assert lock.read_text() == "native writer"
    assert not (home / ".credentials.yaml").exists()
