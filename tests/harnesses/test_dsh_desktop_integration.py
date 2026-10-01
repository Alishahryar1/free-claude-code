"""Persistent DSH integration must preserve native configuration and ownership."""

import importlib
import json

import pytest
from ruamel.yaml import YAML

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog

URL = "http://127.0.0.1:8182"
TOKEN = "desktop-test-token"
ROUTE = "free-claude-code"
REF = "FCC_DSH_DESKTOP_API_KEY"


def catalog(name="fixture/model"):
    return ModelCatalog((CatalogModel(name, name, name, True),), name)


def read(path):
    return YAML(typ="rt").load(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(value, stream)


def config(rows, identity):
    values = [
        row["config"] for row in rows if row.get("id") == identity and "config" in row
    ]
    return values[-1] if values else {}


@pytest.fixture
def desktop():
    return importlib.import_module("free_claude_code.harnesses.dsh_desktop_integration")


@pytest.fixture
def home(tmp_path):
    root = tmp_path / "dsh"
    profile = root / "profiles/desktop"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text(
        json.dumps(
            {
                "name": "dsh-profile-desktop",
                "private": True,
                "dsh": {
                    "profile": {
                        "bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"]
                    }
                },
            }
        )
    )
    (profile / "cordis.patch.yml").write_text("# native settings\n[]\n")
    return root


@pytest.fixture
def state(tmp_path):
    return tmp_path / "fcc/dsh-desktop-integration.json"


def connect(desktop, home, state, token=TOKEN, model="fixture/model"):
    return desktop.configure(
        home,
        URL,
        token,
        catalog(model),
        state_path=state,
        provider_progress_timeout=600,
    )


def test_connect_native_route_credentials_default_and_disconnect(desktop, home, state):
    result = connect(desktop, home, state)
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
    assert TOKEN not in patch.read_text() + state.read_text() + json.dumps(result)
    assert desktop.status(home, URL, TOKEN, state_path=state)["connected"]
    assert not desktop.disconnect(home, state_path=state)["connected"]
    assert ROUTE not in patch.read_text()
    assert REF not in (home / ".credentials.yaml").read_text()
    assert not state.exists()


def test_preserves_other_providers_accounts_tags_and_default(desktop, home, state):
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text("""# keep native settings
- id: unrelated
  config:
    computed: !!js "ctx.get('something')"
- id: llm-pi-ai
  config:
    providers:
      other:
        apiKeyEnv: OTHER_KEY # keep comment
        models: [unchanged]
- id: agent-default-model
  config:
    provider: other
    model: prior
    reasoningEffort: high
""")
    credentials = home / ".credentials.yaml"
    credentials.write_text(
        'version: 1\nrefs:\n  OTHER_KEY: "keep-secret" # keep key\nrecords: {}\n'
    )
    credentials.chmod(0o600)
    before = read(patch)
    connect(desktop, home, state)
    connect(desktop, home, state, token="rotated", model="fixture/new")
    assert read(credentials)["refs"] == {"OTHER_KEY": "keep-secret", REF: "rotated"}
    assert config(read(patch), "agent-default-model")["model"] == "fixture/new"
    desktop.disconnect(home, state_path=state)
    after = read(patch)
    assert after[1:] == before[1:]
    original_tag = before[0]["config"]["computed"]
    actual_tag = after[0]["config"]["computed"]
    assert (actual_tag.tag.value, actual_tag.value) == (
        original_tag.tag.value,
        original_tag.value,
    )
    assert "keep native settings" in patch.read_text()
    assert "!!js" in patch.read_text() and "keep comment" in patch.read_text()
    assert read(credentials)["refs"] == {"OTHER_KEY": "keep-secret"}
    assert "keep key" in credentials.read_text()


def test_user_default_change_survives_refresh_and_disconnect(desktop, home, state):
    connect(desktop, home, state)
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = read(patch)
    config(rows, "agent-default-model").update(provider="other", model="chosen")
    write(patch, rows)
    assert desktop.refresh_connected(
        home,
        URL,
        TOKEN,
        catalog("fixture/new"),
        state_path=state,
        provider_progress_timeout=600,
    )
    assert config(read(patch), "agent-default-model")["model"] == "chosen"
    desktop.disconnect(home, state_path=state)
    assert config(read(patch), "agent-default-model") == {
        "provider": "other",
        "model": "chosen",
    }


@pytest.mark.parametrize("refresh", [False, True])
def test_reset_to_native_default_is_not_taken_back(desktop, home, state, refresh):
    patch = home / "profiles/desktop/cordis.patch.yml"
    write(
        patch,
        [
            {
                "id": "agent-default-model",
                "config": {"provider": "other", "model": "prior"},
            }
        ],
    )
    connect(desktop, home, state)
    rows = read(patch)
    rows[:] = [row for row in rows if row.get("id") != "agent-default-model"]
    write(patch, rows)
    if refresh:
        desktop.refresh_connected(
            home,
            URL,
            TOKEN,
            catalog("fixture/new"),
            state_path=state,
            provider_progress_timeout=600,
        )
        assert not config(read(patch), "agent-default-model")
    desktop.disconnect(home, state_path=state)
    assert not config(read(patch), "agent-default-model")


def test_refresh_never_creates_a_connection(desktop, home, state):
    before = (home / "profiles/desktop/cordis.patch.yml").read_bytes()
    assert not desktop.refresh_connected(
        home, URL, TOKEN, catalog(), state_path=state, provider_progress_timeout=600
    )
    assert (home / "profiles/desktop/cordis.patch.yml").read_bytes() == before
    assert not state.exists()


@pytest.mark.parametrize(
    "target",
    [
        "route",
        "credential",
        "home_patch",
        "dynamic",
        "duplicate",
        "disabled",
        "replacement",
    ],
)
def test_conflicts_make_no_writes(desktop, home, state, target):
    patch = home / "profiles/desktop/cordis.patch.yml"
    if target == "route":
        write(
            patch,
            [{"id": "llm-pi-ai", "config": {"providers": {ROUTE: {"name": "mine"}}}}],
        )
    elif target == "credential":
        write(home / ".credentials.yaml", {"version": 1, "refs": {REF: "mine"}})
    elif target == "home_patch":
        write(
            home / "cordis.patch.yml",
            [{"id": "llm-pi-ai", "config": {"providers": {}}}],
        )
    elif target == "dynamic":
        patch.write_text('- id: llm-pi-ai\n  config: !!js "makeConfig()"\n')
    elif target == "duplicate":
        write(
            patch,
            [{"insert": [{"id": "llm-pi-ai", "name": "@deepseek-ai/dsh-llm-pi-ai"}]}],
        )
    elif target == "replacement":
        write(patch, [{"id": "llm-pi-ai", "name": "my-custom-provider"}])
    else:
        write(patch, [{"id": "llm-pi-ai", "disabled": True}])
    before = {str(p): p.read_bytes() for p in home.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        connect(desktop, home, state)
    assert not state.exists()
    assert {str(p): p.read_bytes() for p in home.rglob("*") if p.is_file()} == before


def test_modified_owned_route_is_not_overwritten(desktop, home, state):
    connect(desktop, home, state)
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = read(patch)
    config(rows, "llm-pi-ai")["providers"][ROUTE]["baseURL"] = "http://user-change/v1"
    write(patch, rows)
    before = patch.read_bytes()
    with pytest.raises(ValueError):
        connect(desktop, home, state)
    assert patch.read_bytes() == before


def test_status_missing_app_does_not_create_files(desktop, tmp_path, state):
    root = tmp_path / "missing"
    result = desktop.status(root, URL, TOKEN, state_path=state)
    assert not result["connected"]
    assert not result["installed"]
    assert not root.exists()
    with pytest.raises(ValueError, match=r"open|Open"):
        connect(desktop, root, state)
    assert not root.exists()


@pytest.mark.parametrize("failure_index", [1, 2, 3, 4])
def test_interrupted_connect_recovers_without_losing_original_default(
    desktop, home, state, monkeypatch, failure_index
):
    import errno

    from free_claude_code.harnesses import dsh_files

    patch = home / "profiles/desktop/cordis.patch.yml"
    write(
        patch,
        [
            {
                "id": "agent-default-model",
                "config": {"provider": "other", "model": "prior"},
            }
        ],
    )
    original = dsh_files.atomic_write_text
    count = 0

    def fail_write(path, content, **kwargs):
        nonlocal count
        count += 1
        if count == failure_index:
            raise OSError(errno.EIO, "injected interruption")
        return original(path, content, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(dsh_files, "atomic_write_text", fail_write)
        with pytest.raises(OSError, match="interruption"):
            connect(desktop, home, state)
    assert not list(home.rglob("*.lock"))
    assert not desktop.status(home, URL, TOKEN, state_path=state)["connected"]
    assert connect(desktop, home, state)["connected"]
    desktop.disconnect(home, state_path=state)
    assert config(read(patch), "agent-default-model") == {
        "provider": "other",
        "model": "prior",
    }


@pytest.mark.parametrize("failure_index", [1, 2, 3])
def test_interrupted_disconnect_is_not_reconnected_by_refresh(
    desktop, home, state, monkeypatch, failure_index
):
    import errno

    from free_claude_code.harnesses import dsh_files

    connect(desktop, home, state)
    original = dsh_files.atomic_write_text
    count = 0

    def fail_write(path, content, **kwargs):
        nonlocal count
        count += 1
        if count == failure_index:
            raise OSError(errno.EIO, "injected interruption")
        return original(path, content, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(dsh_files, "atomic_write_text", fail_write)
        with pytest.raises(OSError, match="interruption"):
            desktop.disconnect(home, state_path=state)
    if failure_index > 1:
        assert desktop.status(home, URL, TOKEN, state_path=state)["disconnect_pending"]
        with pytest.raises(ValueError, match="disconnecting"):
            connect(desktop, home, state)
        desktop.refresh_connected(
            home, URL, TOKEN, catalog(), state_path=state, provider_progress_timeout=600
        )
    else:
        desktop.disconnect(home, state_path=state)
    assert not state.exists()
    assert not desktop.refresh_connected(
        home, URL, TOKEN, catalog(), state_path=state, provider_progress_timeout=600
    )
    assert ROUTE not in (home / "profiles/desktop/cordis.patch.yml").read_text()


def test_disconnect_restores_inheritance_across_duplicate_normal_patches(
    desktop, home, state
):
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = [
        {
            "id": "llm-pi-ai",
            "config": {"providers": {"other": {"models": ["original"]}}},
        },
        {"id": "llm-pi-ai", "disabled": False},
        {
            "id": "agent-default-model",
            "config": {"provider": "other", "model": "original"},
        },
        {"id": "agent-default-model", "disabled": False},
    ]
    write(patch, rows)
    connect(desktop, home, state)
    desktop.disconnect(home, state_path=state)
    assert read(patch) == rows


def test_native_account_records_are_preserved(desktop, home, state):
    credentials = home / ".credentials.yaml"
    records = {
        "deepseek/account": {
            "kind": "grant",
            "payload": {"refreshToken": "keep-account"},
        }
    }
    write(credentials, {"version": 1, "records": records})
    credentials.chmod(0o600)
    connect(desktop, home, state)
    desktop.disconnect(home, state_path=state)
    assert read(credentials)["records"] == records


def test_unrelated_edits_during_connection_survive_disconnect(desktop, home, state):
    connect(desktop, home, state)
    patch = home / "profiles/desktop/cordis.patch.yml"
    rows = read(patch)
    config(rows, "llm-pi-ai")["providers"]["new-provider"] = {"models": ["keep"]}
    rows.append({"id": "user-plugin", "config": {"theme": "blue"}})
    write(patch, rows)
    desktop.disconnect(home, state_path=state)
    assert config(read(patch), "llm-pi-ai")["providers"] == {
        "new-provider": {"models": ["keep"]}
    }
    assert config(read(patch), "user-plugin") == {"theme": "blue"}


def test_original_default_comments_and_quotes_are_restored(desktop, home, state):
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text(
        """- id: agent-default-model
  config:
    provider: "other" # original provider
    model: 'prior' # original model
    reasoningEffort: high # original effort
""",
        encoding="utf-8",
    )
    connect(desktop, home, state)
    desktop.disconnect(home, state_path=state)
    text = patch.read_text(encoding="utf-8")
    assert '"other" # original provider' in text
    assert "'prior' # original model" in text
    assert "high # original effort" in text


@pytest.mark.parametrize(
    "change",
    [
        {"shape": {}},
        {"intent": []},
        {"manage_default": "yes"},
        {"baseline_default": 23},
        {"baseline_default": "bad: [yaml"},
        {"version": True},
        {"next_route": {}},
    ],
)
def test_malformed_ownership_record_is_rejected_without_writes(
    desktop, home, state, change
):
    connect(desktop, home, state)
    record = json.loads(state.read_text())
    record.update(change)
    state.write_text(json.dumps(record))
    before = {str(p): p.read_bytes() for p in home.rglob("*") if p.is_file()}
    saved = state.read_bytes()
    with pytest.raises(ValueError, match="ownership record"):
        desktop.disconnect(home, state_path=state)
    assert state.read_bytes() == saved
    assert {str(p): p.read_bytes() for p in home.rglob("*") if p.is_file()} == before


def test_native_credential_version_rejects_boolean(desktop, home, state):
    credentials = home / ".credentials.yaml"
    write(credentials, {"version": True, "refs": {"OTHER_KEY": "keep"}})
    credentials.chmod(0o600)
    before = credentials.read_bytes()
    with pytest.raises(ValueError, match="version-1"):
        connect(desktop, home, state)
    assert credentials.read_bytes() == before
    assert not state.exists()


def test_environment_shadow_rejected_without_writes(desktop, home, state, monkeypatch):
    monkeypatch.setenv(REF, "different-token")
    with pytest.raises(ValueError, match="environment"):
        connect(desktop, home, state)
    assert not state.exists()
    assert not (home / ".credentials.yaml").exists()
