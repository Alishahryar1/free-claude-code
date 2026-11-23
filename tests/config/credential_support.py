"""Build masked credential edits for existing configuration test fixtures."""

from free_claude_code.core.json_types import JsonObject, JsonValue
from tests.wire_payload import object_array, object_payload


def credential_edit(projection: JsonObject, value: JsonValue):
    if isinstance(value, dict):
        return value
    if value is None:
        entries: list[JsonObject] = []
    elif isinstance(value, str) and (not value.strip() or value == "********"):
        entries = [
            {"retain": object_payload(row)["reference"]}
            for row in object_array(projection["entries"])
        ]
    else:
        entries = [
            {"value": entry}
            for entry in (value if isinstance(value, list) else [value])
        ]
    return {"revision": projection["revision"], "entries": entries}
