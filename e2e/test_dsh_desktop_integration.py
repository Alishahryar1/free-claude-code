"""The DSH Desktop card uses real isolated native configuration files."""

import asyncio
import json
import threading

from playwright.sync_api import expect
from ruamel.yaml import YAML


def install(tmp_path):
    home = tmp_path / ".dsh"
    profile = home / "profiles/desktop"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "package.json").write_text(
        json.dumps(
            {
                "dsh": {
                    "profile": {
                        "bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"]
                    }
                },
            }
        )
    )
    return home


def test_connect_disconnect_preserves_native_settings(page, admin_base_url, tmp_path):
    home = install(tmp_path)
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text("- id: unrelated\n  config: {keep: unchanged}\n")
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Connect")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    expect(dialog).to_be_visible()
    expect(dialog.locator("#dshDesktopIntegrationFiles")).to_contain_text(
        str(patch.resolve())
    )
    dialog.get_by_role("button", name="Connect", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")
    expect(opener).to_have_css("color", "rgb(239, 68, 68)")
    page.screenshot(path=str(tmp_path / "dsh-integrations.png"), full_page=True)
    assert "free-claude-code" in patch.read_text()
    page.reload()
    expect(opener).to_have_text("Disconnect")
    opener.click()
    expect(dialog.locator("#dshDesktopIntegrationDescription")).to_contain_text(
        "Existing"
    )
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(opener).to_have_text("Connect")
    assert "free-claude-code" not in patch.read_text()
    rows = YAML().load(patch.read_text())
    assert rows[0] == {"id": "unrelated", "config": {"keep": "unchanged"}}
    expect(page.locator("#dirtyState")).to_have_text("No changes")


def test_setup_error_and_retry_remain_in_dialog(page, admin_base_url, tmp_path):
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Connect")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    dialog.get_by_role("button", name="Connect", exact=True).click()
    expect(dialog.get_by_role("alert")).to_contain_text("open")
    expect(dialog.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    install(tmp_path)
    dialog.get_by_role("button", name="Connect", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")


def test_superseded_connect_notice_matches_real_disconnect(
    page, admin_base_url, tmp_path, monkeypatch
):
    from free_claude_code.runtime.provider_manager import ProviderRuntimeManager

    home = install(tmp_path)
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Connect")
    entered, release = threading.Event(), threading.Event()
    original = ProviderRuntimeManager.wait_for_catalog

    async def held(manager):
        entered.set()
        assert await asyncio.to_thread(release.wait, 10)
        return await original(manager)

    monkeypatch.setattr(ProviderRuntimeManager, "wait_for_catalog", held)
    endpoint = admin_base_url + "/admin/api/integrations/dsh-desktop"
    try:
        opener.click()
        page.locator("#confirmDshDesktopIntegration").click()
        assert entered.wait(5)
        response = page.request.post(endpoint + "/disconnect")
        assert response.ok
        assert not response.json()["connected"]
    finally:
        release.set()
    expect(page.locator("#dshDesktopIntegrationDialog")).not_to_be_visible()
    expect(opener).to_have_text("Connect")
    notice = page.locator("#dshDesktopIntegrationMessage")
    expect(notice).to_contain_text("disconnected")
    expect(notice).not_to_contain_text("Connected.")
    assert not page.request.get(endpoint).json()["connected"]
    assert not (home / "profiles/desktop/cordis.patch.yml").exists()


def test_changed_fcc_setting_shows_connect_and_is_replaced(
    page, admin_base_url, tmp_path
):
    home = install(tmp_path)
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Connect")
    opener.click()
    page.locator("#confirmDshDesktopIntegration").click()
    expect(opener).to_have_text("Disconnect")
    credentials = home / ".credentials.yaml"
    credentials.write_text("version: 1\nrefs: {FCC_DSH_DESKTOP_API_KEY: edited}\n")
    page.reload()
    expect(opener).to_have_text("Connect")
    opener.click()
    page.locator("#confirmDshDesktopIntegration").click()
    expect(opener).to_have_text("Disconnect")
    assert "edited" not in credentials.read_text()
