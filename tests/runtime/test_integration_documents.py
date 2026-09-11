"""Edits to client configuration preserve text outside the requested settings."""

import pytest
import tomlkit
from tomlkit.exceptions import TOMLKitError

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


def test_noop_json_edit_is_byte_identical():
    source = b'{ "owned": true, "keep": "slash\\/value" }'
    document = JsonDocument(source, jsonc=True)
    document.set(("owned",), True)
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
    "source",
    [
        b'model_providers.fcc.name="Old"\nmodel_providers.other.name="Keep"\n',
        b'[model_providers]\nfcc={name="Old"}\nother={name="Keep"}\n',
        b'model_providers={fcc.name="Old", other.name="Keep"}\n',
    ],
)
def test_missing_auth_preserves_dotted_and_inline_provider_scope(source):
    document = TomlDocument(source)
    document.set(("model_providers", "fcc", "auth", "command"), "fcc-codex")
    document.set(("model_provider",), "fcc")
    assert document.data == {
        "model_providers": {
            "fcc": {"name": "Old", "auth": {"command": "fcc-codex"}},
            "other": {"name": "Keep"},
        },
        "model_provider": "fcc",
    }


@pytest.mark.parametrize(
    "source",
    [
        b'model_providers.fcc.name="Old"\nmodel_providers.other.name="Keep"\n',
        b'model_providers={fcc.name="Old", other.name="Keep"}\n',
    ],
)
def test_deleting_final_dotted_leaf_preserves_its_empty_parent(source):
    document = TomlDocument(source)
    document.delete(("model_providers", "fcc", "name"))
    assert document.data == {"model_providers": {"fcc": {}, "other": {"name": "Keep"}}}


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


@pytest.mark.parametrize(
    "original,corrupted",
    [
        ("keep=true", "keep=1"),
        ("keep=1.00000000000000001", "keep=1.00000000000000002"),
        ("keep=1979-05-27T07:32:00Z", "keep=1979-05-27T00:32:00-07:00"),
        ('keep={nested=[true, "same"]}', 'keep={nested=[1, "same"]}'),
    ],
)
def test_toml_rejects_parseable_unrelated_semantic_changes(
    monkeypatch, original, corrupted
):
    source = f'owned="old"\n{original}\n'.encode()
    document = TomlDocument(source)
    actual_dumps = tomlkit.dumps
    monkeypatch.setattr(
        tomlkit,
        "dumps",
        lambda candidate: actual_dumps(candidate).replace(original, corrupted),
    )
    with pytest.raises(IntegrationError):
        document.set(("owned",), "new")
    assert document.render() == source
    assert document.get(("owned",)) == "old"


def test_json_rejects_serializer_changes_to_unrelated_values(monkeypatch):
    source = b'{"keep":1,"owned":"old"}'
    document = JsonDocument(source, jsonc=False)
    monkeypatch.setattr(
        "free_claude_code.runtime.integrations.documents.json.dumps",
        lambda *_args, **_kwargs: '"new", "keep": 2',
    )
    with pytest.raises(IntegrationError):
        document.set(("owned",), "new")
    assert document.render() == source
    assert document.data == {"keep": 1, "owned": "old"}


@pytest.mark.parametrize("kind", ["json", "toml"])
def test_nested_boolean_to_integer_is_not_treated_as_a_noop(kind):
    document = (
        JsonDocument(b'{"owned":{"inner":true}}', jsonc=False)
        if kind == "json"
        else TomlDocument(b"owned={inner=true}\n")
    )
    document.set(("owned",), {"inner": 1})
    assert type(document.get(("owned", "inner"))) is int


@pytest.mark.parametrize("error_type", [ValueError, TOMLKitError])
def test_toml_dump_errors_are_sanitized_without_adopting_changes(
    monkeypatch, error_type
):
    source = b'owned="old"\n'
    document = TomlDocument(source)

    def fail_dump(_candidate):
        raise error_type("private contents")

    monkeypatch.setattr(tomlkit, "dumps", fail_dump)
    with pytest.raises(IntegrationError) as caught:
        document.set(("owned",), "new")
    assert "private contents" not in str(caught.value)
    assert document.render() == source


def test_toml_preserves_unrelated_nonfinite_exact_numbers_dates_and_arrays():
    source = (
        'owned="old"\r\n'
        "numbers=[nan,-nan,+inf,-inf,1.00000000000000001]\r\n"
        "day=1979-05-27\r\nclock=07:32:00.123\r\n"
        "stamp=1979-05-27T07:32:00-07:00\r\n"
        'mixed=[true,1,"é",{name="keep"}] # untouched\r\n'
    ).encode()
    document = TomlDocument(source)
    document.set(("owned",), "new")
    assert document.render() == source.replace(b'"old"', b'"new"')
