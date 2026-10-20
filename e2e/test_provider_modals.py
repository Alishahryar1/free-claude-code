"""Provider organization and isolated modal editing through the real UI."""

import pytest
from playwright.sync_api import Page, Route, expect

from e2e.provider_support import api_key_input, close_provider, open_provider


def test_provider_titles_link_to_websites_and_every_logo_loads(
    page: Page, admin_base_url: str
) -> None:
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    cards = page.locator(".provider-card")
    expect(cards.first).to_be_visible()
    page.wait_for_function("""() => Array.from(document.querySelectorAll(
        '.provider-card .provider-title a img'
    )).every(image => image.complete)""")
    # Read the rendered DOM together instead of tracing a browser round trip
    # for each attribute of every provider. Keep card identity in failures.
    snapshots = cards.evaluate_all("""cards => cards.map(card => ({
        provider: card.dataset.provider,
        links: Array.from(card.querySelectorAll('.provider-title a'), link => ({
            href: link.getAttribute('href'),
            target: link.getAttribute('target'),
            rel: link.getAttribute('rel'),
            logos: Array.from(link.querySelectorAll('img'), image => ({
                alt: image.getAttribute('alt'),
                complete: image.complete,
                naturalWidth: image.naturalWidth,
            })),
        })),
    }))""")
    assert snapshots
    for card in snapshots:
        assert len(card["links"]) == 1, card
        link = card["links"][0]
        assert link["href"] and link["href"].startswith("https://"), card
        assert link["target"] == "_blank", card
        assert link["rel"] == "noopener noreferrer", card
        assert len(link["logos"]) == 1, card
        logo = link["logos"][0]
        assert logo["alt"] == "", card
        assert logo["complete"] is True, card
        assert logo["naturalWidth"] > 0, card
    expect(page.locator('[data-provider="llamacpp"] .provider-title')).to_have_text(
        "LLaMA.cpp"
    )


def test_provider_website_opens_in_new_tab_and_preserves_pending_settings(
    page: Page, admin_base_url: str
) -> None:
    page.context.route(
        "https://build.nvidia.com/**",
        lambda route: route.fulfill(content_type="text/html", body="Provider website"),
    )
    page.goto(f"{admin_base_url}/admin")
    timeout = page.locator("#field-HTTP_READ_TIMEOUT")
    timeout.fill("181")
    with page.expect_popup() as opened:
        page.locator('[data-provider="nvidia_nim"] .provider-title a').click()
    popup = opened.value
    try:
        expect(popup).to_have_url("https://build.nvidia.com/")
        assert popup.evaluate("window.opener === null") is True
    finally:
        popup.close()
    expect(timeout).to_have_value("181")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")


def test_provider_card_buttons_align_at_the_bottom_of_each_desktop_row(
    page: Page, admin_base_url: str
) -> None:
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    positions = page.locator(
        '[data-provider-group="cloud"] .provider-card'
    ).evaluate_all(
        """cards => cards.map(card => ({
            row: Math.round(card.getBoundingClientRect().top),
            bottom: card.querySelector('[data-provider-settings]').getBoundingClientRect().bottom,
        }))"""
    )
    rows: dict[int, list[float]] = {}
    for position in positions:
        rows.setdefault(position["row"], []).append(position["bottom"])
    assert any(len(bottoms) > 1 for bottoms in rows.values())
    for bottoms in rows.values():
        assert max(bottoms) - min(bottoms) < 1


def test_provider_groups_sort_each_subgroup_and_keep_setup_separate_from_health(
    page: Page, admin_base_url: str
) -> None:
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    groups = page.locator("[data-provider-group]")
    assert groups.locator("h3").all_text_contents() == [
        "OAuth providers",
        "Cloud providers",
        "Local providers",
        "Custom providers",
    ]
    for group in ("cloud", "local"):
        for subgroup in ("configured", "unconfigured"):
            section = page.locator(
                f'[data-provider-group="{group}"] [data-provider-subgroup="{subgroup}"]'
            )
            names = section.locator(".provider-title strong").all_text_contents()
            assert names == sorted(names, key=str.casefold)
    expect(
        page.locator(
            '[data-provider-group="cloud"] [data-provider-subgroup="configured"] [data-provider="open_router"]'
        )
    ).to_be_visible()
    expect(
        page.locator(
            '[data-provider-group="cloud"] [data-provider-subgroup="unconfigured"] [data-provider="cloudflare"]'
        )
    ).to_be_visible()
    expect(
        page.locator(
            '[data-provider-group="local"] [data-provider-subgroup="configured"] [data-provider="lmstudio"]'
        )
    ).to_be_visible()
    expect(page.locator("#providerGroups .status-pill")).to_have_count(0)
    expect(page.locator("#section-providers")).to_have_count(0)
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_have_count(0)


@pytest.mark.parametrize("dismiss", ["close", "escape", "outside"])
def test_modal_groups_all_provider_fields_and_discards_cancelled_edits(
    page: Page, admin_base_url: str, dismiss: str
) -> None:
    page.goto(f"{admin_base_url}/admin")
    dialog = open_provider(page, "azure_openai")
    for key in ("AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_PROXY"):
        expect(dialog.locator(f"#field-{key}")).to_be_visible()
    expect(dialog.locator("#field-NVIDIA_NIM_API_KEY")).to_have_count(0)
    api_key_input(page, "AZURE_OPENAI_API_KEY").fill("cancelled-key")
    expect(page.locator("#saveProvider")).to_be_enabled()
    if dismiss == "close":
        close_provider(page)
    elif dismiss == "escape":
        page.keyboard.press("Escape")
    else:
        page.mouse.click(2, 2)
    expect(dialog).not_to_be_visible()
    expect(
        page.locator('[data-provider="azure_openai"] [data-provider-settings]')
    ).to_be_focused()
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    dialog = open_provider(page, "azure_openai")
    expect(dialog.locator(".api-key-row")).to_have_count(0)


@pytest.mark.parametrize("restart", [False, True])
def test_modal_save_applies_only_its_changes_and_preserves_other_page_edits(
    page: Page, admin_base_url: str, restart: bool
) -> None:
    submissions: list[dict] = []

    def save(route: Route) -> None:
        payload = route.request.post_data_json
        assert isinstance(payload, dict)
        submissions.append(payload)
        route.fulfill(
            json={
                "applied": True,
                "credential_checks": [],
                "restart": {
                    "required": restart,
                    "automatic": restart,
                    "admin_url": "/admin",
                    "instance_id": "before",
                },
            }
        )

    page.route("**/admin/api/config/apply", save)
    if restart:
        page.route(
            "**/admin/api/status",
            lambda route: route.fulfill(
                json={"status": "running", "instance_id": "after"}
            ),
        )
    page.goto(f"{admin_base_url}/admin/model_config")
    model = page.locator("#field-MODEL_SONNET")
    model.fill("open_router/pending-model")
    page.get_by_role("button", name="Providers", exact=True).click()
    dialog = open_provider(page, "nvidia_nim")
    api_key_input(page, "NVIDIA_NIM_API_KEY").fill("new-key")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(page.locator("#messageArea")).to_have_text("Applied")
    assert len(submissions) == 1
    assert set(submissions[0]["values"]) == {"NVIDIA_NIM_API_KEY"}
    assert submissions[0]["values"]["NVIDIA_NIM_API_KEY"]["entries"] == [
        {"value": "new-key"}
    ]
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    page.get_by_role("button", name="Model Config", exact=True).click()
    expect(model).to_have_value("open_router/pending-model")
