"""Rendered contracts for Admin asset generation consistency."""

from urllib.parse import urlsplit

from playwright.sync_api import Error, Page, Request, expect

from free_claude_code.core.version import package_version


def test_selected_admin_tab_survives_refresh_and_browser_navigation(
    page, admin_base_url
):
    page.goto(f"{admin_base_url}/admin")
    for title, path in (
        ("Model Config", "/admin/model_config"),
        ("Messaging", "/admin/messaging"),
        ("Integrations", "/admin/integrations"),
        ("Providers", "/admin"),
    ):
        page.get_by_role("button", name=title, exact=True).click()
        expect(page).to_have_url(f"{admin_base_url}{path}")
        page.reload()
        expect(page.locator("#pageTitle")).to_have_text(title)
        expect(page.get_by_role("button", name=title, exact=True)).to_have_attribute(
            "aria-current", "page"
        )
    page.go_back()
    expect(page.locator("#pageTitle")).to_have_text("Integrations")
    page.go_forward()
    expect(page.locator("#pageTitle")).to_have_text("Providers")


def test_admin_removes_chat_drafts_and_preserves_other_storage(page, admin_base_url):
    page.add_init_script("""(() => {
      for (let index = 0; index < 16; index++) {
        sessionStorage.setItem(`fcc.chat.draft.${index}`, `old draft ${index}`);
      }
      for (let index = 0; index < 16; index++) {
        sessionStorage.setItem(`fcc.code.${index}`, `keep code draft ${index}`);
      }
      sessionStorage.setItem('fcc.chat.draftWithoutDot', 'keep unrelated key');
      sessionStorage.setItem('other-admin-state', 'keep admin state');
    })();""")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator('[data-provider="nvidia_nim"]')).to_be_visible()
    assert page.evaluate("Object.fromEntries(Object.entries(sessionStorage))") == {
        **{f"fcc.code.{index}": f"keep code draft {index}" for index in range(16)},
        "fcc.chat.draftWithoutDot": "keep unrelated key",
        "other-admin-state": "keep admin state",
    }
    expect(page.locator('.nav-link[data-view="providers"]')).to_have_attribute(
        "aria-current", "page"
    )
    expect(page.get_by_role("button", name="Chat Sessions", exact=True)).to_have_count(
        0
    )
    page.get_by_role("button", name="Code sessions", exact=True).click()
    expect(page).to_have_url(f"{admin_base_url}/admin/code")
    expect(
        page.get_by_role("button", name="New code session", exact=True)
    ).to_be_enabled()


def test_admin_storage_denial_keeps_admin_usable(page, admin_base_url):
    errors = []
    warnings = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: (
            warnings.append(message.text) if message.type == "warning" else None
        ),
    )
    page.add_init_script("""Object.defineProperty(window, 'sessionStorage', {
      get() { throw new DOMException('Storage is denied', 'SecurityError'); }
    });""")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator('[data-provider="nvidia_nim"]')).to_be_visible()
    page.get_by_role("button", name="Code sessions", exact=True).click()
    expect(
        page.get_by_role("button", name="New code session", exact=True)
    ).to_be_enabled()
    assert errors == []
    assert any("Chat draft cleanup deferred" in message for message in warnings)


def test_admin_loads_current_release_assets_before_rendering_dynamic_content(
    page: Page,
    admin_base_url: str,
) -> None:
    requested_paths: list[str] = []
    page_errors: list[str] = []

    def record_request(request: Request) -> None:
        requested_paths.append(urlsplit(request.url).path)

    def record_page_error(error: Error) -> None:
        page_errors.append(str(error))

    page.on("request", record_request)
    page.on("pageerror", record_page_error)
    page.goto(f"{admin_base_url}/admin")

    expect(page.locator('[data-provider="nvidia_nim"]')).to_be_visible()
    expect(page.locator(".brand p")).to_have_text(
        f"Server Control · v{package_version()}"
    )
    logo_link = page.get_by_role("link", name="Open Free Claude Code on GitHub")
    expect(logo_link).to_be_visible()
    expect(logo_link).to_have_attribute(
        "href", "https://github.com/Alishahryar1/free-claude-code"
    )
    expect(logo_link).to_have_attribute("target", "_blank")

    versioned_root = f"/admin/assets/{package_version()}"
    assert f"{versioned_root}/app-icon.svg" in requested_paths
    assert f"{versioned_root}/admin.css" in requested_paths
    assert f"{versioned_root}/code_sessions.css" in requested_paths
    assert f"{versioned_root}/model_combobox.js" in requested_paths
    assert f"{versioned_root}/code_sessions.js" in requested_paths
    assert f"{versioned_root}/admin.js" in requested_paths
    assert "/admin/assets/admin.css" not in requested_paths
    assert "/admin/assets/code_sessions.css" not in requested_paths
    assert "/admin/assets/model_combobox.js" not in requested_paths
    assert "/admin/assets/code_sessions.js" not in requested_paths
    assert "/admin/assets/admin.js" not in requested_paths
    assert not any("/chat_sessions." in path for path in requested_paths)
    assert page_errors == []
