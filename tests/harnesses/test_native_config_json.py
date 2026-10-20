import json

import pytest

from free_claude_code.harnesses import (
    claude_desktop_integration as desktop,
)
from free_claude_code.harnesses import (
    claude_integration as claude,
)
from free_claude_code.harnesses import (
    jetbrains_acp_integration as jetbrains,
)
from free_claude_code.harnesses import (
    vscode_chat_integration as vscode,
)


@pytest.fixture(params=["claude", "vscode", "desktop", "jetbrains"])
def native_reader(request):
    readers = {
        "claude": claude._read_object,
        "vscode": lambda path: vscode._read(path)[0],
        "desktop": desktop._read,
        "jetbrains": jetbrains._read,
    }
    return readers[request.param], request.param == "vscode"


def source_for(source, array):
    return f"[{source}]" if array else source


def test_standard_json_does_not_use_json5(native_reader, tmp_path, monkeypatch):
    read, array = native_reader
    document = {"nested": {"value": 1}, "models": [{"id": str(i)} for i in range(999)]}
    expected = [document] if array else document
    path = tmp_path / "native.json"
    path.write_text("\ufeff" + json.dumps(expected), encoding="utf-8")

    def slow_decoder(*args, **kwargs):
        raise AssertionError("Standard JSON must not enter the JSON5 parser")

    monkeypatch.setattr("json5.loads", slow_decoder)
    assert read(path) == expected


def test_json5_native_syntax_still_works(native_reader, tmp_path):
    read, array = native_reader
    path = tmp_path / "native.json"
    path.write_text(source_for("{/* comment */ nested: {'value': 1,},}", array))
    expected = {"nested": {"value": 1}}
    assert read(path) == ([expected] if array else expected)


@pytest.mark.parametrize(
    "source",
    [
        '{"x": 1, "x": 2}',
        '{"x": 1, "\\u0078": 2}',
        '{"nested": {"x": 1, "x": 2}}',
        "{nested: {x: 1, x: 2,},}",
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{"x": 1e999}',
        "{x: +Infinity,}",
    ],
)
def test_invalid_native_values_remain_errors(native_reader, tmp_path, source):
    read, array = native_reader
    path = tmp_path / "native.json"
    path.write_text(source_for(source, array))
    with pytest.raises(ValueError):
        read(path)
