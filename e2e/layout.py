"""Geometry for browser checks that require elements to be visible."""

from playwright.sync_api import FloatRect, Locator


def visible_bounds(locator: Locator) -> FloatRect:
    bounds = locator.bounding_box()
    assert bounds is not None, f"Expected visible element: {locator}"
    return bounds
