"""The DSH Desktop card uses real isolated native configuration files."""

import json

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


def test_configure_disconnect_preserves_native_settings(page, admin_base_url, tmp_path):
    home = install(tmp_path)
    patch = home / "profiles/desktop/cordis.patch.yml"
    patch.write_text(
        "- id: agent-default-model\n  config:\n    provider: other\n    model: prior\n"
    )
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Configure")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    expect(dialog).to_be_visible()
    expect(dialog.locator("#dshDesktopIntegrationFiles")).to_contain_text(
        str(patch.resolve())
    )
    dialog.get_by_role("button", name="Configure", exact=True).click()
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
    expect(opener).to_have_text("Configure")
    assert "free-claude-code" not in patch.read_text()
    rows = YAML().load(patch.read_text())
    assert rows == [
        {"id": "agent-default-model", "config": {"provider": "other", "model": "prior"}}
    ]
    assert not (tmp_path / ".fcc/dsh-desktop-integration.json").exists()
    expect(page.locator("#dirtyState")).to_have_text("No changes")


def test_setup_error_and_retry_remain_in_dialog(page, admin_base_url, tmp_path):
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Configure")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    dialog.get_by_role("button", name="Configure", exact=True).click()
    expect(dialog.get_by_role("alert")).to_contain_text("open")
    expect(dialog.get_by_role("button", name="Configure", exact=True)).to_be_enabled()
    install(tmp_path)
    dialog.get_by_role("button", name="Configure", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")


def test_failed_disconnect_can_be_retried_after_reload(
    page, admin_base_url, tmp_path, monkeypatch
):
    from free_claude_code.harnesses import dsh_files

    home = install(tmp_path)
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    dialog.get_by_role("button", name="Configure", exact=True).click()
    expect(opener).to_have_text("Disconnect")
    original = dsh_files.atomic_write_text

    def fail(path, content, **kwargs):
        if path == home / ".credentials.yaml":
            raise PermissionError("private file error")
        return original(path, content, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(dsh_files, "atomic_write_text", fail)
        opener.click()
        dialog.get_by_role("button", name="Disconnect", exact=True).click()
        expect(dialog.get_by_role("alert")).to_be_visible()
        expect(
            dialog.get_by_role("button", name="Retry disconnect", exact=True)
        ).to_be_enabled()
        page.reload()
        expect(opener).to_have_text("Retry disconnect")
    opener.click()
    dialog.get_by_role("button", name="Retry disconnect", exact=True).click()
    expect(opener).to_have_text("Configure")
    assert not (tmp_path / ".fcc/dsh-desktop-integration.json").exists()


def test_partial_configure_keeps_disconnect_after_unreadable_reload(
    page, admin_base_url, tmp_path, monkeypatch
):
    from free_claude_code.harnesses import dsh_files

    home = install(tmp_path)
    patch = home / "profiles/desktop/cordis.patch.yml"
    original = dsh_files.atomic_write_text

    def fail(path, content, **kwargs):
        if path == patch:
            raise OSError("profile interruption")
        return original(path, content, **kwargs)

    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Configure")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    with monkeypatch.context() as scoped:
        scoped.setattr(dsh_files, "atomic_write_text", fail)
        dialog.get_by_role("button", name="Configure", exact=True).click()
        expect(dialog.get_by_role("alert")).to_be_visible()
        expect(
            dialog.get_by_role("button", name="Disconnect", exact=True)
        ).to_be_enabled()
    patch.write_text("bad: [yaml")
    page.reload()
    expect(opener).to_have_text("Disconnect")
    expect(page.locator("#dshDesktopIntegrationMessage")).to_contain_text("YAML")
    patch.write_text("[]\n")
    opener.click()
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(opener).to_have_text("Configure")


def test_shared_credential_disconnect_is_successful_and_reconnectable(
    page, admin_base_url, tmp_path
):
    home = install(tmp_path)
    peer = home / "profiles/web"
    peer.mkdir()
    (peer / "package.json").write_bytes(
        (home / "profiles/desktop/package.json").read_bytes()
    )
    (peer / "cordis.patch.yml").write_text(
        "- id: llm-deepseek\n  config:\n    apiKeyEnv: FCC_DSH_DESKTOP_API_KEY\n"
    )
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openDshDesktopIntegration")
    expect(opener).to_have_text("Configure")
    opener.click()
    dialog = page.locator("#dshDesktopIntegrationDialog")
    dialog.get_by_role("button", name="Configure", exact=True).click()
    expect(opener).to_have_text("Disconnect")
    opener.click()
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(opener).to_have_text("Configure")
    message = page.locator("#dshDesktopIntegrationMessage")
    expect(message).to_contain_text("retained")
    expect(message).not_to_have_class("message-area error")
    page.reload()
    expect(opener).to_have_text("Configure")
    expect(message).to_contain_text("retained")
    opener.click()
    dialog.get_by_role("button", name="Configure", exact=True).click()
    expect(opener).to_have_text("Disconnect")
