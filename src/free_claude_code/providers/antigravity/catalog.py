"""Catalog metadata for the native Google Antigravity connected account."""

from free_claude_code.config.provider_catalog import (
    ProviderAuthKind,
    ProviderDescriptor,
)

ANTIGRAVITY_DEFAULT_BASE = "https://daily-cloudcode-pa.googleapis.com"


def register_antigravity_descriptor(
    catalog: dict[str, ProviderDescriptor],
) -> None:
    """Register Antigravity once without coupling the neutral catalog to its adapter."""

    catalog.setdefault(
        "antigravity",
        ProviderDescriptor(
            provider_id="antigravity",
            display_name="Google Antigravity",
            website_url="https://antigravity.google/",
            logo_filename="gemini-color.svg",
            auth_kind=ProviderAuthKind.CONNECTED_ACCOUNT,
            default_base_url=ANTIGRAVITY_DEFAULT_BASE,
        ),
    )
