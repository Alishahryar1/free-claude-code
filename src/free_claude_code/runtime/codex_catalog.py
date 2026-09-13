"""Publish the application model inventory for Codex clients."""

import json
import uuid
from collections.abc import Mapping
from pathlib import Path

from free_claude_code.api.model_catalog import (
    ModelCatalogView,
    build_models_list_response,
)
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.config.paths import codex_model_catalog_path
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonValue
from free_claude_code.harnesses.codex_model_catalog import (
    build_codex_model_catalog,
)
from free_claude_code.harnesses.model_catalog import (
    ClientModel,
    client_models_from_response,
)


def current_codex_models(
    runtime: RequestRuntimePort, settings: Settings | None = None
) -> tuple[ClientModel, ...]:
    """Read the current FCC inventory without a request to the server itself."""
    response = build_models_list_response(
        settings or runtime.current_settings(),
        runtime,
        view=ModelCatalogView.RESPONSES,
    )
    return client_models_from_response(
        response.model_dump(by_alias=True, exclude_none=True)
    )


class CodexModelCatalogPublisher:
    """Own synchronization of the stable Codex model catalog file."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._catalog_path = catalog_path

    def ensure_exists(self, runtime: RequestRuntimePort) -> None:
        """Publish a startup catalog only when no prior catalog exists."""

        catalog_path = self._resolved_catalog_path()
        if catalog_path.exists():
            return
        self._publish(runtime, catalog_path)

    def publish(self, runtime: RequestRuntimePort) -> None:
        """Publish the complete current application model inventory."""

        self._publish(runtime, self._resolved_catalog_path())

    def _publish(
        self,
        runtime: RequestRuntimePort,
        catalog_path: Path,
    ) -> None:
        catalog = build_codex_model_catalog(current_codex_models(runtime))
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
