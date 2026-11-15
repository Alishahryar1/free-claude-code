"""Historical on-disk connection fixture, independent of current launch setup."""

import json
from pathlib import Path


def write_previous_connection(path: Path, state: Path, model: str = "provider/model"):
    original = json.loads(path.read_text()) if path.exists() else {}
    previous = {key: original[key] for key in ("modelProvider",) if key in original}
    document = json.loads(json.dumps(original))
    document["modelProvider"] = "gemini"
    document.setdefault("customModelsConfig", {}).setdefault("customModels", {})[
        model
    ] = {
        "modelName": "gemini-api://models/" + model,
        "label": "FCC: " + model,
        "toolFormatterType": "TOOL_FORMATTER_TYPE_NONE",
        "modelFeatures": {"supportsToolCalls": True},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    state.write_text(
        json.dumps(
            {
                "settings_path": str(path.resolve()),
                "previous_provider": previous,
                "model_keys": [model],
            }
        ),
        encoding="utf-8",
    )
    return original
