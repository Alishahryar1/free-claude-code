import json
import subprocess

from playwright.sync_api import expect

from free_claude_code.harnesses import jetbrains_acp_integration as jb
from tests.harnesses.test_jetbrains_acp_integration import install


def test_connect_retry_disconnect_and_modal_dismissal(
    page, admin_base_url, monkeypatch
):
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openJetBrainsIntegration")
    dialog = page.locator("#jetBrainsIntegrationDialog")
    action = page.locator("#confirmJetBrainsIntegration")
    expect(opener).to_have_text("Connect")
    for dismiss in ("close", "escape", "outside"):
        opener.click()
        expect(dialog).to_be_visible()
        expect(page.locator("#jetBrainsIntegrationFiles li")).to_have_text(
            [str(jb.config_path().resolve())]
        )
        page.locator("#jetBrainsIntegrationDescription").click()
        expect(dialog).to_be_visible()
        if dismiss == "close":
            page.locator("#closeJetBrainsIntegration").click()
        elif dismiss == "escape":
            page.keyboard.press("Escape")
        else:
            page.mouse.click(1, 1)
        expect(dialog).not_to_be_visible()
    opener.click()
    action.click()
    expect(page.locator("#jetBrainsIntegrationDialogMessage")).to_be_visible()
    expect(action).to_be_enabled()
    assert not jb.config_path().exists()
    install(jb.registry_path(), jb.system_root())
    monkeypatch.setattr(
        jb.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 0, "v24.19.0", ""),
    )
    action.click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")
    assert (
        "Claude Code (FCC)" in json.loads(jb.config_path().read_text())["agent_servers"]
    )
    page.reload()
    expect(opener).to_have_text("Disconnect")
    expect(opener).to_have_css("color", "rgb(239, 68, 68)")
    jb.registry_path().unlink()
    assert page.request.post(
        f"{admin_base_url}/admin/api/integrations/jetbrains-acp/refresh"
    ).ok
    page.reload()
    expect(page.locator("#jetBrainsIntegrationMessage")).to_be_visible()
    expect(opener).to_have_text("Disconnect")
    expect(opener).to_be_enabled()
    opener.click()
    action.click()
    expect(opener).to_have_text("Connect")
    assert json.loads(jb.config_path().read_text()) == {"agent_servers": {}}


def test_unreadable_config_can_be_retried(page, admin_base_url):
    path = jb.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{")
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openJetBrainsIntegration")
    expect(opener).to_have_text("Retry")
    expect(page.locator("#jetBrainsIntegrationMessage")).to_be_visible()
    path.write_text("{}")
    opener.click()
    expect(opener).to_have_text("Connect")
