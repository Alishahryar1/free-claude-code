"""Classification for provider proxies owned by the provider catalog."""

from collections.abc import Mapping

import httpx

from .provider_catalog import PROVIDER_CATALOG
from .settings import Settings


def _provider_proxy_env_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.proxy_attr is None:
            continue
        field = Settings.model_fields[descriptor.proxy_attr]
        alias = field.validation_alias
        if not isinstance(alias, str):
            raise AssertionError(
                f"Settings field {descriptor.proxy_attr!r} needs one string alias"
            )
        keys.append(alias)
    return tuple(keys)


PROVIDER_PROXY_ENV_KEYS = _provider_proxy_env_keys()


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
