"""Completed web results shared by application operations and wire formatters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class WebFetchResult:
    url: str
    title: str
    media_type: str
    data: str
