"""Build masked credential edits for existing configuration test fixtures."""


def credential_edit(projection, value):
    if isinstance(value, dict):
        return value
    if value is None:
        entries = []
    elif isinstance(value, str) and (not value.strip() or value == "********"):
        entries = [{"retain": row["reference"]} for row in projection["entries"]]
    else:
        entries = [
            {"value": entry}
            for entry in (value if isinstance(value, list) else [value])
        ]
    return {"revision": projection["revision"], "entries": entries}
