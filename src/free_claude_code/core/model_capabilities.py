"""Neutral capability values shared by model discovery and client catalogs."""

from enum import StrEnum


class ModelInputModality(StrEnum):
    """Input media that FCC can preserve across supported protocol paths."""

    TEXT = "text"
    IMAGE = "image"
