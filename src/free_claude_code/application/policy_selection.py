"""Settings-dependent candidate eligibility at the application boundary."""

import ipaddress
from urllib.parse import urlsplit

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG, ProviderAuthKind
from free_claude_code.config.routing_policy import LOCAL_PROVIDERS, RoutingCandidate
from free_claude_code.config.settings import Settings


def candidate_destination_allowed(
    candidate: RoutingCandidate, settings: Settings
) -> bool:
    provider_id = candidate.model.partition("/")[0]
    descriptor = PROVIDER_CATALOG[provider_id]
    if descriptor.auth_kind is not ProviderAuthKind.CONFIGURATION:
        return False
    if any(
        not getattr(settings, attr, None) for attr in descriptor.configuration_attrs()
    ):
        return False
    if settings.routing_profile != "local-only":
        return True
    if provider_id not in LOCAL_PROVIDERS:
        return False
    if descriptor.proxy_attr and getattr(settings, descriptor.proxy_attr, None):
        return False
    try:
        url = urlsplit(getattr(settings, descriptor.base_url_attr or "", ""))
        if url.scheme not in {"http", "https"} or url.username or url.password:
            return False
        return (
            url.hostname == "localhost"
            or ipaddress.ip_address(url.hostname or "").is_loopback
        )
    except ValueError:
        return False
