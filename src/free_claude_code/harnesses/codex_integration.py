"""Connect native Codex clients through their shared TOML configuration."""

import os
from pathlib import Path

import tomlkit
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.exceptions import ParseError
from tomlkit.items import InlineTable, Table
from tomlkit.toml_document import TOMLDocument

from free_claude_code.config.server_urls import proxy_v1_url, same_proxy_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.codex import (
    PRINT_PROXY_AUTH_TOKEN_FLAG,
    codex_config_values,
)
from free_claude_code.harnesses.config_file import atomic_write_text


def config_path() -> Path:
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    return root / "config.toml"


def _child_table(
    parent: TOMLDocument | Table | InlineTable | OutOfOrderTableProxy,
    key: str,
    *,
    create: bool = False,
) -> Table | InlineTable | OutOfOrderTableProxy:
    if key not in parent:
        child = (
            tomlkit.inline_table()
            if isinstance(parent, InlineTable)
            else tomlkit.table()
        )
        if create:
            parent[key] = child
        return child
    child = parent[key]
    if not isinstance(child, Table | InlineTable | OutOfOrderTableProxy):
        raise ValueError("Codex provider configuration must use TOML tables")
    return child


def _read(path: Path) -> TOMLDocument:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return tomlkit.document()
    try:
        document = tomlkit.parse(content)
    except ParseError:
        raise ValueError("Invalid Codex TOML") from None
    providers = _child_table(document, "model_providers")
    provider = _child_table(providers, "fcc")
    _child_table(provider, "auth")
    return document


def _same_catalog(value: object, catalog_path: Path) -> bool:
    return (
        isinstance(value, str)
        and Path(value).is_absolute()
        and Path(value).resolve() == catalog_path
    )


def _recognizes_connection(document: TOMLDocument, proxy_root_url: str) -> bool:
    provider = _child_table(_child_table(document, "model_providers"), "fcc")
    return document.get("model_provider") == "fcc" and same_proxy_url(
        provider.get("base_url"), proxy_v1_url(proxy_root_url)
    )


def recognizes_connection(path: Path, proxy_root_url: str) -> bool:
    """Identify FCC independently of newly introduced managed settings."""
    return _recognizes_connection(_read(path.resolve()), proxy_root_url)


def _connected(document: TOMLDocument, catalog_path: Path, proxy_root_url: str) -> bool:
    provider = _child_table(_child_table(document, "model_providers"), "fcc")
    auth = _child_table(provider, "auth")
    return (
        _recognizes_connection(document, proxy_root_url)
        and _same_catalog(document.get("model_catalog_json"), catalog_path)
        and provider.get("wire_api") == "responses"
        and auth.get("command") == "fcc-codex"
        and auth.get("args") == [PRINT_PROXY_AUTH_TOKEN_FLAG]
    )


def _connect(
    path: Path, document: TOMLDocument, catalog_path: Path, proxy_root_url: str
) -> bool:
    before = tomlkit.dumps(document)
    values = codex_config_values(api_url=proxy_root_url)
    values["model_catalog_json"] = str(catalog_path)
    for dotted_key, value in values.items():
        *parents, key = dotted_key.split(".")
        parent: TOMLDocument | Table | InlineTable | OutOfOrderTableProxy = document
        for name in parents:
            parent = _child_table(parent, name, create=True)
        if parent.get(key) != value:
            parent[key] = value
    content = tomlkit.dumps(document)
    if content == before:
        return False
    atomic_write_text(path, content)
    return True


def refresh_connected(path: Path, catalog_path: Path, proxy_root_url: str) -> bool:
    """Refresh an existing FCC connection without selecting a provider or model."""
    path = path.resolve()
    document = _read(path)
    if not _recognizes_connection(document, proxy_root_url):
        return False
    return _connect(path, document, catalog_path.resolve(), proxy_root_url)


def configure(
    path: Path, catalog_path: Path, proxy_root_url: str, connected: bool | None = None
) -> JsonObject:
    """Inspect or edit FCC's entries without changing the user's selected model."""
    path = path.resolve()
    catalog_path = catalog_path.resolve()
    document = _read(path)
    if connected is True:
        _connect(path, document, catalog_path, proxy_root_url)
    elif connected is False:
        before = tomlkit.dumps(document)
        _child_table(document, "model_providers").pop("fcc", None)
        if document.get("model_provider") == "fcc":
            document.pop("model_provider")
        if _same_catalog(document.get("model_catalog_json"), catalog_path):
            document.pop("model_catalog_json")
        content = tomlkit.dumps(document)
        if content != before:
            atomic_write_text(path, content)
    if connected is not None:
        document = _read(path)
    return {
        "connected": _connected(document, catalog_path, proxy_root_url),
        "paths": {"codex_config": str(path)},
    }
