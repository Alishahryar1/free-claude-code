"""Edits to client configuration preserve text outside the requested settings."""

import pytest

from free_claude_code.application.integrations import IntegrationError
from free_claude_code.runtime.integrations.documents import (
    MISSING,
    JsonDocument,
    TomlDocument,
)


def test_jsonc_value_change_preserves_bom_unicode_crlf_and_comments():
    source = '\ufeff{\r\n  // keep\r\n  "unrelated": "é",\r\n  "owned": "old", /* keep */\r\n}\r\n'.encode()
    document = JsonDocument(source, jsonc=True)
    document.set(("owned",), "new")
    assert document.render() == source.replace(b'"old"', b'"new"')
    assert document.get(("unrelated",)) == "é"


@pytest.mark.parametrize("trailing", ["", ","])
def test_insertion_keeps_line_comment_attached_to_existing_value(trailing):
    source = f'{{\n  "keep": 1{trailing} // keep this comment\n}}'.encode()
    document = JsonDocument(source, jsonc=True)
    document.set(("owned",), True)
    result = document.render()
    assert b'"keep": 1, // keep this comment' in result
    assert document.get(("owned",)) is True
    assert result.endswith(b"}")


@pytest.mark.parametrize("source", [b"{}", b"{ /* keep */ }", b"{\n// keep\n}"])
def test_add_nested_property_to_empty_or_comment_only_object(source):
    document = JsonDocument(source, jsonc=True)
    document.set(("agent_servers", "Claude Code (FCC)", "env", "TOKEN"), "test")
    assert (
        document.get(("agent_servers", "Claude Code (FCC)", "env", "TOKEN")) == "test"
    )
    assert source.strip(b"{} \n") in document.render()


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("trailing", ["", ","])
def test_array_removal_keeps_other_entries_and_comments(index, trailing):
    source = (
        '{"env": [ {"name":"A","value":"a"}, /* keep */ '
        '{"name":"B","value":"b"}, {"name":"C","value":"c"}' + trailing + " ]}"
    ).encode()
    document = JsonDocument(source, jsonc=True)
    document.delete(("env", index))
    expected = [{"name": name, "value": name.lower()} for name in "ABC"]
    expected.pop(index)
    assert document.get(("env",)) == expected
    assert b"/* keep */" in document.render()
    for entry in expected:
        assert (
            f'{{"name":"{entry["name"]}","value":"{entry["value"]}"}}'.encode()
            in document.render()
        )


@pytest.mark.parametrize(
    "source", [b'{"owned":true}', b'{"owned":true,}', b'{"owned":true, // keep\n}']
)
def test_delete_only_property_leaves_valid_object(source):
    document = JsonDocument(source, jsonc=True)
    document.delete(("owned",))
    assert document.data == {}
    if b"// keep" in source:
        assert b"// keep" in document.render()


def test_array_append_and_nested_value_edit_preserve_other_values():
    document = JsonDocument(
        '{"env":[{"name":"KEEP", "value":"é"},]}'.encode(), jsonc=True
    )
    document.set(("env", 1), {"name": "FCC", "value": "old"})
    document.set(("env", 1, "value"), "new")
    assert document.get(("env", 1, "value")) == "new"
    assert b'{"name":"KEEP", "value":"\xc3\xa9"}' in document.render()


@pytest.mark.parametrize(
    "source",
    [
        b"{,}",
        b"[1,,]",
        b'{"a":01}',
        b"{} {}",
        b'{"a":NaN}',
        b'{"a":}',
        b'{"a": /* unclosed',
        b"{a:1}",
    ],
)
def test_jsonc_rejects_invalid_syntax(source):
    with pytest.raises(IntegrationError):
        JsonDocument(source, jsonc=True)


@pytest.mark.parametrize("source", [b'{"a":1,}', b'{/* comment */"a":1}'])
def test_strict_json_does_not_accept_jsonc_extensions(source):
    with pytest.raises(IntegrationError):
        JsonDocument(source, jsonc=False)


def test_duplicate_owned_key_is_rejected_without_exposing_values():
    document = JsonDocument(
        b'{"owned":"secret-a","owned":"secret-b","keep":1}', jsonc=True
    )
    with pytest.raises(IntegrationError) as error:
        document.set(("owned",), "new")
    assert "secret" not in str(error.value)
    assert document.render() == b'{"owned":"secret-a","owned":"secret-b","keep":1}'


def test_unrelated_duplicate_keys_are_preserved_but_owned_parent_is_ambiguous():
    document = JsonDocument(b'{"keep":1,"keep":2,"owned":false}', jsonc=True)
    document.set(("owned",), True)
    assert document.render() == b'{"keep":1,"keep":2,"owned":true}'
    document = JsonDocument(b'{"env":{},"env":{}}', jsonc=True)
    with pytest.raises(IntegrationError):
        document.set(("env", "key"), "value")


def test_noop_json_edit_is_byte_identical_and_missing_delete_is_safe():
    source = b'{ "owned": true, "keep": "slash\\/value" }'
    document = JsonDocument(source, jsonc=True)
    document.set(("owned",), True)
    document.delete(("absent",))
    assert document.get(("absent",)) is MISSING
    assert document.render() == source


def test_append_at_immediate_closing_delimiter_orders_separator_before_entry():
    document = JsonDocument(b'{"one":true}', jsonc=True)
    document.set(("two",), [])
    document.set(("two", 0), "first")
    document.set(("two", 1), "second")
    assert document.data == {"one": True, "two": ["first", "second"]}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_toml_changes_one_value_without_reformatting_other_tables(newline):
    source = newline.join(
        [
            "# keep",
            'model = "old" # model',
            "[model_providers.other]",
            'name = "Other"',
            "",
        ]
    ).encode()
    document = TomlDocument(source)
    document.set(("model",), "new")
    assert document.render() == source.replace(b'"old"', b'"new"')
    document.set(
        ("model_providers", "fcc", "auth", "args"), ["--print-proxy-auth-token"]
    )
    assert document.get(("model_providers", "other", "name")) == "Other"


def test_toml_inline_tables_and_deletion_preserve_other_provider():
    document = TomlDocument(
        b'[model_providers]\nfcc = { name = "Old", auth = {command="old"} }\nother = {name="Keep"}\n'
    )
    document.set(("model_providers", "fcc", "auth", "command"), "fcc-codex")
    document.delete(("model_providers", "fcc", "name"))
    assert document.get(("model_providers", "fcc", "auth", "command")) == "fcc-codex"
    assert b'other = {name="Keep"}' in document.render()


@pytest.mark.parametrize(
    "document",
    [
        JsonDocument(b'{"env": "wrong"}', jsonc=True),
        TomlDocument(b'model_providers = "wrong"\n'),
    ],
)
def test_incompatible_container_cannot_be_replaced_as_a_side_effect(document):
    path = (
        ("env", "key")
        if isinstance(document, JsonDocument)
        else ("model_providers", "fcc", "name")
    )
    before = document.render()
    with pytest.raises(IntegrationError):
        document.set(path, "new")
    assert document.render() == before
