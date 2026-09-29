import re

import pytest
from playwright.sync_api import expect


def test_claude_creation_prepares_without_sending_and_stays_fixed(
    page, admin_base_url, tmp_path, code_control
):
    page.goto(f"{admin_base_url}/admin/code")
    page.get_by_role("button", name="New code session", exact=True).click()
    page.get_by_role("combobox", name="Harness", exact=True).select_option("claude")
    page.get_by_role("textbox", name="Folder", exact=True).fill(str(tmp_path))
    page.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
    expect(page.locator("#codeMode")).to_be_enabled()
    expect(page.locator("#codeHarness")).to_have_value("claude")
    expect(page.locator("#codeHarness")).to_be_disabled()
    expect(page.locator("#codeComposer")).to_have_attribute(
        "placeholder", "Ask Claude Code to work on this folder…"
    )
    assert [
        option.text_content() for option in page.locator("#codeMode option").all()
    ] == ["Use config", "Manual", "Plan"]
    assert not code_control.harness.connections
    assert len(code_control.claude.connections) == 1
    assert not code_control.claude.connections[0].inputs
    page.locator("#codeComposer").fill("Keep this draft")
    page.reload()
    expect(page.locator("#codeSend")).to_be_enabled()
    expect(page.locator("#codeComposer")).to_have_value("Keep this draft")
    assert len(code_control.claude.connections) == 1
    page.locator("#codeSend").click()
    code_control.run(code_control.claude.wait_inputs(1))
    assert code_control.claude.connections[0].inputs[0][1] == "Keep this draft"
    assert not code_control.harness.connections


def test_claude_setup_failure_preserves_history_and_allows_retry(
    page, admin_base_url, tmp_path, code_control, monkeypatch
):
    from free_claude_code.application.code_sessions.models import CodeUnavailableError

    original = code_control.claude.prepare

    async def broken(*args):
        raise CodeUnavailableError("Test initialization failure")

    monkeypatch.setattr(code_control.claude, "prepare", broken)
    page.goto(f"{admin_base_url}/admin/code")
    page.get_by_role("button", name="New code session", exact=True).click()
    page.get_by_role("combobox", name="Harness", exact=True).select_option("claude")
    page.get_by_role("textbox", name="Folder", exact=True).fill(str(tmp_path))
    page.get_by_role("button", name="Create session", exact=True).click()
    expect(page.locator("#codePrepareRetry")).to_be_visible()
    expect(page.locator("#codeSelectionError")).to_contain_text(
        "Test initialization failure"
    )
    page.locator("#codeComposer").fill("Preserve me")
    monkeypatch.setattr(code_control.claude, "prepare", original)
    page.locator("#codePrepareRetry").click()
    expect(page.locator("#codeSend")).to_be_enabled()
    expect(page.locator("#codeComposer")).to_have_value("Preserve me")


@pytest.mark.parametrize(
    "setting,value,control",
    [("mode", "plan", "codeMode"), ("reasoning_effort", "max", "codeReasoning")],
)
def test_unavailable_saved_claude_setting_remains_visible(
    page, admin_base_url, tmp_path, code_control, setting, value, control
):
    import uuid

    session = code_control.run(
        code_control.service.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
    )

    code_control.run(
        code_control.service.update_settings(
            session.id, session.revision, {setting: value}
        )
    )
    code_control.claude.modes = code_control.claude.modes[:1]
    code_control.claude.efforts = ("low",)
    page.route(
        f"**/admin/api/code/sessions/{session.id}/prepare",
        lambda route: route.fulfill(
            status=400, json={"detail": "Saved setting unavailable"}
        ),
    )
    page.goto(f"{admin_base_url}/admin/code/{session.id}")
    expect(page.locator("#codePrepareRetry")).to_be_visible()
    expect(page.locator(f"#{control}")).to_have_value(value)
    expect(page.locator(f"#{control} option:checked")).to_have_js_property(
        "disabled", True
    )
    expect(page.locator("#codeComposer")).to_have_attribute(
        "placeholder", "Ask Claude Code to work on this folder…"
    )
