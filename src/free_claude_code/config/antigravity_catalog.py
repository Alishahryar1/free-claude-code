"""Registration for connected providers whose adapters own account credentials."""

from types import ModuleType

from . import provider_catalog as _provider_catalog
from .provider_catalog import ProviderAuthKind, ProviderDescriptor

ANTIGRAVITY_DEFAULT_BASE = "https://daily-cloudcode-pa.googleapis.com"


def register_antigravity_catalog(module: ModuleType) -> None:
    """Insert Antigravity beside other connected accounts and refresh ID order."""

    catalog = module.PROVIDER_CATALOG
    if "antigravity" in catalog:
        return

    descriptor = ProviderDescriptor(
        provider_id="antigravity",
        display_name="Google Antigravity",
        website_url="https://antigravity.google/",
        logo_filename="gemini-color.svg",
        auth_kind=ProviderAuthKind.CONNECTED_ACCOUNT,
        default_base_url=ANTIGRAVITY_DEFAULT_BASE,
    )
    ordered: dict[str, ProviderDescriptor] = {}
    for provider_id, existing in catalog.items():
        ordered[provider_id] = existing
        if provider_id == "github_copilot":
            ordered["antigravity"] = descriptor
    if "antigravity" not in ordered:
        ordered["antigravity"] = descriptor

    catalog.clear()
    catalog.update(ordered)
    module.__dict__["SUPPORTED_PROVIDER_IDS"] = tuple(catalog.keys())


# Run before config.settings imports SUPPORTED_PROVIDER_IDS by value.
register_antigravity_catalog(_provider_catalog)
