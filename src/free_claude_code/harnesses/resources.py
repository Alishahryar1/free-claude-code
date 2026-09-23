"""Private native configuration owned by one launcher process lifetime."""

import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

from free_claude_code.config.paths import launcher_temp_dir_path
from free_claude_code.core.json_types import JsonValue


class LaunchResources:
    """Lazily allocate files and register cleanup with the shared launch scope."""

    def __init__(self, stack: ExitStack) -> None:
        self._stack = stack
        self._directory: Path | None = None

    @property
    def directory(self) -> Path:
        if self._directory is None:
            base = launcher_temp_dir_path()
            base.mkdir(parents=True, mode=0o700, exist_ok=True)
            if os.name != "nt":
                base.chmod(0o700)
            directory = self._stack.enter_context(
                tempfile.TemporaryDirectory(prefix="fcc-", dir=base)
            )
            self._directory = Path(directory)
        return self._directory

    def write_json(self, filename: str, payload: JsonValue) -> Path:
        """Write JSON, also accepted by native YAML readers, without overwrites."""

        path = self.directory / filename
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=True, indent=2)
            output.write("\n")
        return path
