"""FCC compares and changes only its native DSH Desktop entries."""

import copy
import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq, TaggedScalar

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.core.json_types import JsonTree
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from free_claude_code.harnesses import dsh_files
from tests.wire_payload import object_array, object_payload

URL = "http://127.0.0.1:8182"
TOKEN = "desktop-test-token"
ROUTE = "free-claude-code"
REF = desktop.DSH_DESKTOP_API_KEY


def catalog(name: str = "fixture/model"):
    return ModelCatalog((CatalogModel(name, name, name, True),), name)


type YamlValue = JsonTree[TaggedScalar]
type YamlMapping = dict[str, YamlValue]


def read(path: Path) -> YamlValue:
    return YAML(typ="rt").load(path.read_text(encoding="utf-8"))


def write(path: Path, value: YamlValue):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(value, stream)


def config(rows: YamlValue, identity: str) -> YamlMapping:
    return next(
        object_payload(row["config"])
        for row in reversed(object_array(rows))
        if row.get("id") == identity and "config" in row
    )


@pytest.fixture
def home(tmp_path: Path):
    root = tmp_path / "dsh"
    profile = root / "profiles/desktop"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text('{"dsh":{"profile":{"bundles":[]}}}')
    (profile / "cordis.patch.yml").write_text("# native settings\n[]\n")
    return root


def connect(home: Path, token: str = TOKEN, model: str = "fixture/model"):
    return desktop.configure(home, URL, token, catalog(model))


def status(home: Path):
    return desktop.status(home, URL, TOKEN, catalog())


def test_connect_status_disconnect_and_reconnect(home: Path):
    assert not status(home)["connected"]
    result = connect(home)
    assert result["connected"]
    patch = home / "profiles/desktop/cordis.patch.yml"
    provider = object_payload(
        object_payload(config(read(patch), "llm-pi-ai")["providers"])[ROUTE]
    )
    assert provider["api"] == "openai-responses"
    assert provider["baseURL"] == URL + "/v1"
    assert object_array(provider["models"])[0]["id"] == "fixture/model"
    assert provider["apiKeyEnv"] == REF
    assert (
        object_payload(object_payload(read(home / ".credentials.yaml"))["refs"])[REF]
        == TOKEN
    )
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
def test_status_requires_all_fcc_keys_and_connect_replaces_them(home: Path, field: str):
    connect(home)
    patch = home / "profiles/desktop/cordis.patch.yml"
    credentials = home / ".credentials.yaml"
    rows = read(patch)
    providers = object_payload(config(rows, "llm-pi-ai")["providers"])
    if field == "provider":
        del providers[ROUTE]
    elif field == "default":
        config(rows, "agent-default-model")["model"] = "other"
    elif field in {"token", "missing-token"}:
        data = object_payload(read(credentials))
        refs = object_payload(data["refs"])
        if field == "token":
            refs[REF] = "edited"
        else:
            del refs[REF]
        write(credentials, data)
    else:
        object_payload(providers[ROUTE])[field] = [] if field == "models" else "changed"
    write(patch, rows)
    assert not status(home)["connected"]
    assert connect(home)["connected"]
    assert status(home)["connected"]


def test_status_compares_current_catalog_url_and_token(home: Path):
    connect(home)
    for url, token, models in [
        (URL + "-changed", TOKEN, catalog()),
        (URL, "rotated", catalog()),
        (URL, TOKEN, catalog("fixture/new")),
    ]:
        assert not desktop.status(home, url, token, models)["connected"]


def test_other_settings_accounts_comments_tags_and_peer_files_are_preserved(home: Path):
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
    before = object_array(read(patch))
    connect(home)
    connect(home, token="rotated", model="fixture/new")
    assert object_payload(read(credentials))["refs"] == {
        "OTHER_KEY": "keep-secret",
        REF: "rotated",
    }
    desktop.disconnect(home)
    after = object_array(read(patch))
    assert after[1] == before[1]
    previous = object_payload(before[0]["config"])["computed"]
    current = object_payload(after[0]["config"])["computed"]
    assert isinstance(previous, TaggedScalar) and isinstance(current, TaggedScalar)
    assert current.value == previous.value
    assert current.tag == previous.tag
    assert "# keep native settings" in patch.read_text()
    assert "# keep comment" in patch.read_text()
    assert read(credentials) == {
        "version": 1,
        "refs": {"OTHER_KEY": "keep-secret"},
        "records": {"native": "unchanged"},
    }
    assert "# keep key" in credentials.read_text()
    assert peer.read_text() == "unreadable unrelated profile, even if it names " + REF


def test_disconnect_removes_all_fcc_copies_and_preserves_other_default(home: Path):
    connect(home)
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = object_array(read(patch))
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
    home: Path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    if operation == "disconnect":
        connect(home)
    original = dsh_files.write_yaml

    def fail(
        path: Path, document: CommentedMap | CommentedSeq, *, private: bool = False
    ):
        if path == home / ".credentials.yaml":
            raise PermissionError("test failure")
        return original(path, document, private=private)

    with monkeypatch.context() as scoped:
        scoped.setattr(dsh_files, "write_yaml", fail)
        with pytest.raises(PermissionError):
            connect(home) if operation == "connect" else desktop.disconnect(home)
    assert not status(home)["connected"]
    assert connect(home)["connected"]
    assert not desktop.disconnect(home)["connected"]


def test_absent_home_status_disconnect_and_refresh_create_nothing(tmp_path: Path):
    home = tmp_path / "absent"
    assert not status(home)["connected"]
    assert not desktop.disconnect(home)["connected"]
    assert not desktop.refresh_connected(home, URL, TOKEN, catalog())
    assert not home.exists()


def test_invalid_yaml_does_not_modify_files_or_expose_secrets(home: Path):
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text("secret: [private-token")
    before = patch.read_bytes()
    with pytest.raises(desktop.DshConfigError, match="Invalid YAML") as error:
        connect(home)
    assert "private-token" not in str(error.value)
    assert patch.read_bytes() == before
    assert not (home / ".credentials.yaml").exists()


def test_connect_honors_native_profile_lock(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    lock = home / "profiles/desktop/package.json.lock"
    lock.write_text("native writer")
    original = dsh_files.file_lock

    def replace_file_lock(path: Path, **kwargs: object):
        return original(path, wait=0)

    monkeypatch.setattr(dsh_files, "file_lock", replace_file_lock)
    with pytest.raises(TimeoutError):
        connect(home)
    assert lock.read_text() == "native writer"
    assert not (home / ".credentials.yaml").exists()


def test_refresh_replaces_old_derived_timeout_and_preserves_other_settings(home: Path):
    connect(home)
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = read(patch)
    llm = config(rows, "llm-pi-ai")
    providers = object_payload(llm["providers"])
    object_payload(providers[ROUTE])["streamIdleTimeoutMs"] = 660_000
    providers["other"] = {"models": ["unchanged"]}
    llm["anotherSetting"] = "unchanged"
    write(patch, rows)
    assert not status(home)["connected"]
    assert desktop.refresh_connected(home, URL, TOKEN, catalog())
    assert status(home)["connected"]
    refreshed = config(read(patch), "llm-pi-ai")
    refreshed_providers = object_payload(refreshed["providers"])
    assert (
        object_payload(refreshed_providers[ROUTE])["streamIdleTimeoutMs"]
        == 2_147_483_647
    )
    assert refreshed_providers["other"] == {"models": ["unchanged"]}
    assert refreshed["anotherSetting"] == "unchanged"
