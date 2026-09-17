"""Literal domain constraints shared by web protocols and fetch egress."""

import re
from dataclasses import dataclass

_DOMAIN_PATTERN = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


@dataclass(frozen=True, slots=True)
class WebSearchDomainFilter:
    """Literal hostname constraints supplied by Claude Code."""

    allowed: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()


def parse_domains(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of literal hostnames.")

    domains: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry or entry != entry.strip():
            raise ValueError(f"{field} entries must be non-empty hostnames.")
        if any(character in entry for character in ":/\\?#@*"):
            raise ValueError(
                f"{field} entry {entry!r} must be a literal hostname without a "
                "scheme, port, path, query, fragment, user info, or wildcard."
            )
        try:
            domain = entry.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError(
                f"{field} entry {entry!r} is not a valid hostname."
            ) from exc
        if _DOMAIN_PATTERN.fullmatch(domain) is None:
            raise ValueError(f"{field} entry {entry!r} is not a valid hostname.")
        domains.append(domain)

    if len(domains) != len(set(domains)):
        raise ValueError(f"{field} must not contain duplicate hostnames.")
    return tuple(domains)


def domain_matches(host: str, domain: str) -> bool:
    host = host.rstrip(".").encode("idna").decode("ascii").lower()
    return host == domain or host.endswith("." + domain)
