"""Write, compare, and remove FCC's entries in DSH Desktop's native files."""

import copy
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import dsh_files
from free_claude_code.harnesses.dsh_config import DSH_PROVIDER_ID, build_dsh_provider
from free_claude_code.harnesses.dsh_files import DshConfigError, mapping

DSH_DESKTOP_API_KEY = "FCC_DSH_DESKTOP_API_KEY"


def config_home() -> Path:
    value = os.environ.get("DSH_HOME", "").strip()
    if value == "~":
        return Path.home()
    if value.startswith(("~/", "~\\")):
        return (Path.home() / value[2:]).resolve()
    return Path(value).resolve() if value else Path.home() / ".dsh"


def _paths(home: Path) -> JsonObject:
    return {
        "desktop_profile": str(home / "profiles/desktop/cordis.patch.yml"),
        "credentials": str(home / ".credentials.yaml"),
    }


def _read(home: Path) -> tuple[CommentedSeq, CommentedMap]:
    for name in ("profiles/desktop/cordis.patch.yml", ".credentials.yaml"):
        dsh_files.regular_path(home / name, root=home)
    rows = dsh_files.read_yaml(
        home / "profiles/desktop/cordis.patch.yml", sequence=True
    )
    credentials = dsh_files.read_yaml(home / ".credentials.yaml")
    assert isinstance(rows, CommentedSeq) and isinstance(credentials, CommentedMap)
    return rows, credentials


def _rows(rows: CommentedSeq, identity: str) -> list[CommentedMap]:
    return [
        row
        for row in rows
        if isinstance(row, CommentedMap) and row.get("id") == identity
    ]


def _configs(
    rows: CommentedSeq, identity: str, *, create: bool = False
) -> list[CommentedMap]:
    matches = _rows(rows, identity)
    configs = [mapping(row["config"]) for row in matches if "config" in row]
    if not configs and create:
        row = matches[-1] if matches else CommentedMap(id=identity)
        if not matches:
            rows.append(row)
        row["config"] = CommentedMap()
        configs.append(row["config"])
    return configs


def _providers(config: CommentedMap) -> CommentedMap:
    return mapping(config["providers"]) if "providers" in config else CommentedMap()


def has_provider(home: Path) -> bool:
    rows, _ = _read(home)
    return any(
        DSH_PROVIDER_ID in _providers(config) for config in _configs(rows, "llm-pi-ai")
    )


@contextmanager
def _locked(home: Path) -> Iterator[None]:
    # Use DSH's locks so a native edit cannot lose unrelated settings during our write.
    _read(home)
    with ExitStack() as locks:
        if (home / "profiles/desktop").is_dir():
            locks.enter_context(
                dsh_files.file_lock(home / "profiles/desktop/package.json.lock", wait=2)
            )
        if home.is_dir():
            locks.enter_context(
                dsh_files.file_lock(home / ".credentials.yaml.lock", wait=30)
            )
        yield


def _provider(catalog: ModelCatalog, url: str, timeout: float) -> JsonObject:
    return build_dsh_provider(
        catalog.models,
        proxy_root_url=url,
        credential_ref=DSH_DESKTOP_API_KEY,
        provider_progress_timeout=timeout,
    )


def status(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    provider_progress_timeout: float,
) -> JsonObject:
    home = home.resolve()
    rows, credentials = _read(home)
    providers = _configs(rows, "llm-pi-ai")
    defaults = _configs(rows, "agent-default-model")
    refs = mapping(credentials["refs"]) if "refs" in credentials else CommentedMap()
    connected = bool(
        auth_token
        and providers
        and defaults
        and _providers(providers[-1]).get(DSH_PROVIDER_ID)
        == _provider(catalog, proxy_root_url, provider_progress_timeout)
        and defaults[-1]
        == {"provider": DSH_PROVIDER_ID, "model": catalog.default_model_id}
        and refs.get(DSH_DESKTOP_API_KEY) == auth_token
        and credentials.get("version") == 1
    )
    return {"connected": connected, "paths": _paths(home)}


def configure(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    provider_progress_timeout: float,
    only_existing: bool = False,
) -> JsonObject:
    home = home.resolve()
    if not auth_token.strip():
        raise DshConfigError(
            "Set an FCC proxy authentication token before connecting DSH Desktop."
        )
    with _locked(home):
        if only_existing and not has_provider(home):
            return {"connected": False, "changed": False, "paths": _paths(home)}
        if not (home / "profiles/desktop/package.json").is_file():
            raise DshConfigError(
                "Install and open DeepSeek Harness Desktop once, then retry Connect."
            )
        rows, credentials = _read(home)
        provider = _provider(catalog, proxy_root_url, provider_progress_timeout)
        for config in _configs(rows, "llm-pi-ai", create=True):
            providers = _providers(config)
            providers[DSH_PROVIDER_ID] = CommentedMap(copy.deepcopy(provider))
            config["providers"] = providers
        _configs(rows, "agent-default-model", create=True)
        for row in _rows(rows, "agent-default-model"):
            if "config" in row:
                row["config"] = CommentedMap(
                    provider=DSH_PROVIDER_ID, model=catalog.default_model_id
                )
        refs = mapping(credentials["refs"]) if "refs" in credentials else CommentedMap()
        refs[DSH_DESKTOP_API_KEY] = auth_token
        credentials["version"], credentials["refs"] = 1, refs
        changed = dsh_files.write_yaml(home / "profiles/desktop/cordis.patch.yml", rows)
        changed |= dsh_files.write_yaml(
            home / ".credentials.yaml", credentials, private=True
        )
    return status(
        home,
        proxy_root_url,
        auth_token,
        catalog,
        provider_progress_timeout=provider_progress_timeout,
    ) | {"changed": changed}


def disconnect(home: Path) -> JsonObject:
    home = home.resolve()
    with _locked(home):
        rows, credentials = _read(home)
        profile_changed = False
        for config in _configs(rows, "llm-pi-ai"):
            providers = _providers(config)
            if DSH_PROVIDER_ID in providers:
                del providers[DSH_PROVIDER_ID]
                profile_changed = True
        for row in _rows(rows, "agent-default-model"):
            if (
                "config" in row
                and mapping(row["config"]).get("provider") == DSH_PROVIDER_ID
            ):
                # Removing our selection override lets the native bundle supply its default.
                del row["config"]
                profile_changed = True
        if profile_changed:
            dsh_files.write_yaml(home / "profiles/desktop/cordis.patch.yml", rows)
        refs = mapping(credentials["refs"]) if "refs" in credentials else CommentedMap()
        if DSH_DESKTOP_API_KEY in refs:
            del refs[DSH_DESKTOP_API_KEY]
            dsh_files.write_yaml(home / ".credentials.yaml", credentials, private=True)
    return {"connected": False, "paths": _paths(home)}


def refresh_connected(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    provider_progress_timeout: float,
) -> bool:
    result = configure(
        home,
        proxy_root_url,
        auth_token,
        catalog,
        provider_progress_timeout=provider_progress_timeout,
        only_existing=True,
    )
    return result["changed"] is True
