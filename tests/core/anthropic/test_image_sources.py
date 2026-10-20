import pytest

from free_claude_code.core.anthropic.image_sources import (
    AnthropicImageSourceError,
    portable_anthropic_image_url,
)


@pytest.mark.parametrize(
    "media_type",
    ("image/jpeg", "image/png", "image/gif", "image/webp"),
)
def test_portable_image_url_wraps_supported_base64(media_type: str) -> None:
    assert (
        portable_anthropic_image_url(
            {"type": "base64", "media_type": media_type, "data": "aGVsbG8="}
        )
        == f"data:{media_type};base64,aGVsbG8="
    )


def test_portable_image_url_canonicalizes_equivalent_data_url() -> None:
    assert (
        portable_anthropic_image_url(
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "data:IMAGE/PNG;BASE64,aGVs\r\nbG8=",
            }
        )
        == "data:image/png;base64,aGVsbG8="
    )


def test_portable_image_url_removes_base64_line_wrapping() -> None:
    assert (
        portable_anthropic_image_url(
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "  aG\tVs\r\nbG8=  ",
            }
        )
        == "data:image/png;base64,aGVsbG8="
    )


def test_portable_image_url_preserves_remote_url_without_fetching() -> None:
    url = "https://images.example.test/vision.png"

    assert portable_anthropic_image_url({"type": "url", "url": url}) == url


@pytest.mark.parametrize(
    ("source", "message"),
    (
        ({"type": "base64", "media_type": "image/png", "data": ""}, "data"),
        (
            {
                "type": "base64",
                "media_type": "image/svg+xml",
                "data": "aGVsbG8=",
            },
            "media type",
        ),
        (
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "data:image/jpeg;base64,aGVsbG8=",
            },
            "does not match",
        ),
        (
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "data:image/png,aGVsbG8=",
            },
            "data URL",
        ),
        (
            {"type": "base64", "media_type": "image/png", "data": "@@@="},
            "base64",
        ),
        (
            {"type": "base64", "media_type": "image/png", "data": "abcde"},
            "base64",
        ),
        ({"type": "file", "file_id": "file_123"}, "file"),
        ({"type": "url", "url": ""}, "URL"),
    ),
)
def test_portable_image_url_rejects_unrepresentable_source(
    source: dict[str, str], message: str
) -> None:
    with pytest.raises(AnthropicImageSourceError, match=message):
        portable_anthropic_image_url(source)


def test_image_source_error_does_not_expose_payload() -> None:
    secret_payload = "SECRET_IMAGE_PAYLOAD"

    with pytest.raises(AnthropicImageSourceError) as exc_info:
        portable_anthropic_image_url(
            {
                "type": "base64",
                "media_type": "image/png",
                "data": secret_payload,
            }
        )

    assert secret_payload not in str(exc_info.value)
