"""Register Headroom in DSH's native home patch without changing profiles."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import dsh_files
from free_claude_code.harnesses.headroom_mcp import McpSetupError, Registration

_PLUGIN = "@deepseek-ai/dsh-mcp-client"
_ID = "fcc-headroom-mcp"


def _entries(value: object) -> list[JsonObject]:
    """Inspect composed plugin rows, including nested insert rows in patches."""
    found: list[JsonObject] = []
    if isinstance(value, list):
        for child in value:
            found.extend(_entries(child))
    elif isinstance(value, dict):
        config = value.get("config")
        if value.get("id") == _ID and (
            value.get("name") != _PLUGIN
            or not isinstance(config, dict)
            or config.get("serverName") != "headroom"
        ):
            raise McpSetupError(f"DSH already uses {_ID} for a different server.")
        if (
            value.get("name") == _PLUGIN
            and isinstance(config, dict)
            and config.get("serverName") == "headroom"
        ):
            entry = cast(JsonObject, dict(config))
            if value.get("disabled") is True:
                entry["disabled"] = True
            found.append(entry)
        for key in ("insert", "plugins"):
            if key in value:
                found.extend(_entries(value[key]))
    return found


def dsh_registration(
    headroom: Path, home: Path, dump_config: Callable[[], str], *, profile: str = "web"
) -> Registration | None:
    if not (home / "profiles" / profile / "package.json").is_file():
        return None
    path = home / "cordis.patch.yml"
    expected: JsonObject = {
        "serverName": "headroom",
        "transport": "stdio",
        "command": str(headroom),
        "args": ["mcp", "serve"],
    }

    def read() -> JsonObject | None:
        dsh_files.regular_path(path, root=home)
        rows = dsh_files.read_yaml(path, sequence=True)
        local = _entries(rows)
        source = dump_config()
        try:
            composed = dsh_files.yaml_parser().load(source)
        except ValueError, YAMLError:
            raise McpSetupError(
                "DSH returned invalid composed configuration."
            ) from None
        if not isinstance(composed, (list, dict)):
            raise McpSetupError("DSH did not return composed plugin configuration.")
        effective = _entries(composed)
        if len(local) > 1 or len(effective) > 1:
            raise McpSetupError(
                "DSH has duplicate headroom servers. Resolve them before retrying."
            )
        if local and effective and local[0] != effective[0]:
            raise McpSetupError(
                "DSH has conflicting headroom configuration across profile and home scopes."
            )
        return next(iter(effective or local), None)

    def write() -> None:
        home.mkdir(parents=True, exist_ok=True)
        with dsh_files.file_lock(home / "cordis.patch.yml.lock", wait=10):
            if read() is not None:
                raise McpSetupError(
                    "DSH headroom configuration changed during setup. Retry the installer."
                )
            rows = dsh_files.read_yaml(path, sequence=True)
            assert isinstance(rows, CommentedSeq)
            rows.append(
                CommentedMap(
                    insert=CommentedSeq(
                        [
                            CommentedMap(
                                id=_ID, name=_PLUGIN, config=CommentedMap(expected)
                            )
                        ]
                    )
                )
            )
            dsh_files.write_yaml(path, rows, private=True)

    return Registration(f"dsh ({profile})", path, expected, read, write)
