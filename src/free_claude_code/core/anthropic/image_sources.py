"""Portable Anthropic image-source normalization."""

import base64
import binascii
import re
from collections.abc import Mapping

_SUPPORTED_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
_DATA_URL_PATTERN = re.compile(
    r"data:([^;,]+);base64,(.*)",
    flags=re.IGNORECASE | re.DOTALL,
)
_ASCII_WHITESPACE = str.maketrans("", "", " \t\r\n\f\v")


class AnthropicImageSourceError(ValueError):
    """Raised when an Anthropic image source cannot cross protocol boundaries."""


def _source_field(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def portable_anthropic_image_url(source: object) -> str:
    """Return a portable URL for one Anthropic image source."""
    source_type = _source_field(source, "type")
    if source_type == "url":
        url = _source_field(source, "url")
        if not isinstance(url, str) or not url.strip():
            raise AnthropicImageSourceError("Image URL must be a non-empty string")
        return url

    if source_type != "base64":
        label = source_type if isinstance(source_type, str) else "unknown"
        raise AnthropicImageSourceError(
            f"Image source type {label!r} cannot cross this protocol boundary"
        )

    media_type_value = _source_field(source, "media_type")
    if not isinstance(media_type_value, str):
        raise AnthropicImageSourceError("Image media type must be a string")
    media_type = media_type_value.lower()
    if media_type not in _SUPPORTED_MEDIA_TYPES:
        raise AnthropicImageSourceError(
            f"Unsupported image media type {media_type_value!r}"
        )

    data = _source_field(source, "data")
    if not isinstance(data, str) or not data.strip():
        raise AnthropicImageSourceError("Base64 image data must be non-empty")

    raw_data = data.strip()
    if raw_data[:5].lower() == "data:":
        match = _DATA_URL_PATTERN.fullmatch(raw_data)
        if match is None:
            raise AnthropicImageSourceError("Malformed base64 image data URL")
        embedded_media_type, payload = match.groups()
        if embedded_media_type.lower() != media_type:
            raise AnthropicImageSourceError(
                "Image data URL media type does not match the declared media type"
            )
    else:
        payload = raw_data

    canonical_payload = payload.translate(_ASCII_WHITESPACE)
    if not canonical_payload:
        raise AnthropicImageSourceError("Base64 image data must be non-empty")
    try:
        base64.b64decode(canonical_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AnthropicImageSourceError("Image data is not valid base64") from exc

    return f"data:{media_type};base64,{canonical_payload}"
