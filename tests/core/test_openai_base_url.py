"""OpenAI-compatible base URL normalization policy."""

from free_claude_code.core.openai_base_url import openai_v1_base_url


def test_server_root_gains_v1_suffix() -> None:
    assert openai_v1_base_url("http://localhost:9000") == "http://localhost:9000/v1"


def test_v1_base_is_unchanged() -> None:
    assert openai_v1_base_url("http://localhost:9000/v1") == "http://localhost:9000/v1"


def test_trailing_slash_is_normalized() -> None:
    assert openai_v1_base_url("http://localhost:9000/v1/") == "http://localhost:9000/v1"


def test_nested_path_gains_v1_suffix() -> None:
    assert (
        openai_v1_base_url("https://host.example/api") == "https://host.example/api/v1"
    )
