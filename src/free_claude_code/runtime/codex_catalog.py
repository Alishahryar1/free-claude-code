"""Publish the application model inventory for Codex clients."""

import json
import uuid
from collections.abc import Mapping
from pathlib import Path

from free_claude_code.application.model_catalog import read_model_catalog
from free_claude_code.application.ports import ModelCatalogPort
from free_claude_code.config.paths import codex_model_catalog_path
from free_claude_code.core.json_types import JsonValue
from free_claude_code.harnesses.codex_model_catalog import (
    build_codex_model_catalog,
)


class CodexModelCatalogPublisher:
    """Own synchronization of the stable Codex model catalog file."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._catalog_path = catalog_path

    def publish(self, runtime: ModelCatalogPort) -> None:
        """Publish the complete current application model inventory."""

        self._publish(runtime, self._resolved_catalog_path())

    def _publish(
        self,
        runtime: ModelCatalogPort,
        catalog_path: Path,
    ) -> None:
        catalog = build_codex_model_catalog(read_model_catalog(runtime).models)
        models = catalog.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError("Codex model catalog contains no routable models.")
        write_codex_model_catalog(catalog_path, catalog)

    def _resolved_catalog_path(self) -> Path:
        return self._catalog_path or codex_model_catalog_path()


def write_codex_model_catalog(
    catalog_path: Path, catalog: Mapping[str, JsonValue]
) -> bool:
    """Atomically write changed Codex model catalog JSON."""

    content = (json.dumps(catalog, ensure_ascii=True, indent=2) + "\n").encode()
    try:
        if catalog_path.read_bytes() == content:
            return False
    except FileNotFoundError:
        pass

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = catalog_path.with_name(f".{catalog_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(content)
        temp_path.replace(catalog_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return True
