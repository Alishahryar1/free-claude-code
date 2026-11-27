"""Classification for provider proxies owned by the provider catalog."""

from collections.abc import Mapping

import httpx

from .provider_catalog import PROVIDER_CATALOG
from .settings import ENV_KEY_BY_SETTING

PROVIDER_PROXY_ENV_KEYS = tuple(
    ENV_KEY_BY_SETTING[descriptor.proxy_attr]
    for descriptor in PROVIDER_CATALOG.values()
    if descriptor.proxy_attr is not None
)


def invalid_provider_proxy_keys(values: Mapping[str, str]) -> tuple[str, ...]:
    """Return catalog provider-proxy keys whose nonblank values are unusable."""

    invalid: list[str] = []
    for key in PROVIDER_PROXY_ENV_KEYS:
        value = values.get(key, "").strip()
        if not value:
            continue
        try:
            proxy = httpx.Proxy(value)
        except httpx.InvalidURL, ValueError:
            invalid.append(key)
            continue
        if not proxy.url.host:
            invalid.append(key)
    return tuple(invalid)
