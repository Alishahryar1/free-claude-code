import json

import pytest
from pydantic import ValidationError

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.custom_providers import (
    CustomProviderDefinition,
    encode_custom_providers,
)
from free_claude_code.config.settings import Settings

ID = "custom_" + "a" * 32


def definition(**changes):
    return CustomProviderDefinition.model_validate(
        {
            "provider_id": ID,
            "display_name": "My Gateway",
            "base_url": "https://gateway.example/api/v2/",
            "api_key": "private-key",
            **changes,
        }
    )


def test_private_storage_round_trip_and_routing():
    original = definition(model_ids=[" org/model ", "org/model"])
    encoded = encode_custom_providers((original,))
    settings = Settings.model_validate(
        {"FCC_CUSTOM_PROVIDERS": encoded, "MODEL": f"{ID}/org/model"}
    )
    assert settings.custom_providers == (original,)
    assert original.model_ids == ("org/model",)
    assert original.base_url == "https://gateway.example/api/v2"
    assert "private-key" not in repr(original)
    assert "private-key" not in original.model_dump_json()
    assert json.loads(encoded)[0]["api_key"] == "private-key"
    assert ModelRouter(settings).resolve(settings.model).primary.provider_id == ID


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org",
        "https://u:p@example.org",
        "https://example.org?key=secret",
        "https://example.org/#fragment",
    ],
)
def test_invalid_api_bases_are_rejected(url):
    with pytest.raises(ValidationError):
        definition(base_url=url)


@pytest.mark.parametrize("prefix", ["", "anthropic/", "claude-3-freecc-no-thinking/"])
def test_unknown_custom_reference_never_falls_back(prefix):
    with pytest.raises(UnknownProviderError):
        ModelRouter(Settings()).resolve(f"{prefix}{ID}/missing")


def test_duplicate_names_and_missing_route_targets_are_rejected():
    with pytest.raises(ValidationError):
        Settings(
            custom_providers=[
                definition(),
                definition(provider_id="custom_" + "b" * 32, display_name="MY GATEWAY"),
            ]
        )
    with pytest.raises(ValidationError):
        Settings(model=f"{ID}/missing")


def test_format_and_reasoning_must_match():
    with pytest.raises(ValidationError):
        definition(api_format="anthropic_messages", reasoning_format="openai_effort")
