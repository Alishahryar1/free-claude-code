"""Rendered one-step Admin configuration workflow regressions."""

import pytest
from playwright.sync_api import Page, Route, expect


def test_apply_is_the_only_config_action_and_retains_invalid_edits(
    page: Page,
    admin_base_url: str,
) -> None:
    config_mutations: list[tuple[str, str]] = []

    def record_config_mutation(method: str, url: str) -> None:
        if "/admin/api/config/" in url:
            config_mutations.append((method, url.rsplit("/", maxsplit=1)[-1]))

    page.on(
        "request",
        lambda request: record_config_mutation(request.method, request.url),
    )
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")

    expect(page.get_by_role("button", name="Validate", exact=True)).to_have_count(0)
    apply_button = page.get_by_role("button", name="Apply", exact=True)
    expect(apply_button).to_be_disabled()

    runtime_section = page.locator("#section-runtime")
    runtime_section.get_by_role("button", name="Show advanced", exact=True).click()
    timeout_input = runtime_section.locator("#field-PROVIDER_PROGRESS_TIMEOUT")
    timeout_input.fill("0")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    expect(apply_button).to_be_enabled()

    apply_button.click()

    expect(page.locator("#messageArea")).to_contain_text("PROVIDER_PROGRESS_TIMEOUT")
    expect(timeout_input).to_have_value("0")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    expect(apply_button).to_be_enabled()
    assert config_mutations == [("POST", "apply")]


def test_key_rejection_retains_edits_and_restores_focus(
    page: Page, admin_base_url: str
):
    pending: list[Route] = []
    page.route("**/admin/api/config/apply", lambda route: pending.append(route))
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    key = page.locator("#field-MISTRAL_API_KEY")
    other = page.locator("#field-NVIDIA_NIM_API_KEY")
    key.fill("bad-key")
    other.fill("other-edit")
    page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#messageArea")).to_have_text("Checking API keys…")
    expect(page.locator("#view-providers")).to_have_attribute("inert", "")
    expect(page.locator("#applyButton")).to_be_disabled()
    assert len(pending) == 1
    submitted = pending[0].request.post_data_json
    assert isinstance(submitted, dict)
    assert submitted["values"] == {
        "MISTRAL_API_KEY": "bad-key",
        "NVIDIA_NIM_API_KEY": "other-edit",
    }
    pending.pop().fulfill(
        json={
            "applied": False,
            "errors": ["Rejected key"],
            "credential_checks": [
                {
                    "key": "MISTRAL_API_KEY",
                    "status": "rejected",
                    "message": "Check this API key.",
                }
            ],
        }
    )
    expect(key).to_be_focused()
    expect(key).to_have_attribute("aria-invalid", "true")
    expect(page.locator("#field-MISTRAL_API_KEY-error")).to_have_text(
        "Check this API key."
    )
    expect(key).to_have_value("bad-key")
    expect(other).to_have_value("other-edit")
    expect(page.locator("#dirtyState")).to_have_text("2 unsaved changes")
    expect(page.locator("#field-OPENROUTER_API_KEY")).to_be_disabled()
    key.fill("corrected")
    expect(page.locator("#field-MISTRAL_API_KEY-error")).to_have_count(0)
    expect(page.locator("#applyButton")).to_be_enabled()


@pytest.mark.parametrize("restart", [False, True])
def test_unverified_warning_survives_apply(
    page: Page, admin_base_url: str, restart: bool
):
    availability: list[Route] = []
    page.route(
        "**/admin/api/providers/local-status", lambda route: availability.append(route)
    )
    page.route(
        "**/admin/api/config/apply",
        lambda route: route.fulfill(
            json={
                "applied": True,
                "restart": {
                    "required": restart,
                    "automatic": restart,
                    "admin_url": "/admin",
                    "instance_id": "old-server",
                },
                "credential_checks": [
                    {
                        "key": "NVIDIA_NIM_API_KEY",
                        "status": "unverified",
                        "message": "Verification unavailable.",
                    }
                ],
            }
        ),
    )
    page.route(
        "**/admin/api/status",
        lambda route: route.fulfill(
            json={"status": "running", "instance_id": "new-server"}
        ),
    )
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.locator("#field-NVIDIA_NIM_API_KEY").fill("new-key")
    page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_editable()
    expect(page.locator("#applyButton")).to_have_text("Apply")
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")

    current = page.locator('[data-provider-check-result="lmstudio"]')
    with page.expect_response("**/admin/api/providers/local-status") as response:
        old = availability.pop(0)
        if restart:
            old.fulfill(status=503, json={"detail": "Old check failed"})
        else:
            payload = old.fetch().json()
            for provider in payload["providers"]:
                provider.update(
                    status="offline", label="Offline", message="Old availability result"
                )
            old.fulfill(json=payload)
    response.value.finished()
    page.evaluate("() => new Promise(requestAnimationFrame)")
    expect(current).to_be_hidden()

    if restart:
        availability.pop().fulfill(status=503, json={"detail": "New check failed"})
        expect(current).to_have_text("Availability check failed. Use Test to retry.")
    else:
        availability.pop().continue_()
        expect(current).to_have_text("Reachable: http://localhost:1234/v1")
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_editable()


def test_apply_network_error_unlocks_form_and_keeps_edits(
    page: Page, admin_base_url: str
):
    page.route("**/admin/api/config/apply", lambda route: route.abort())
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    key = page.locator("#field-NVIDIA_NIM_API_KEY")
    key.fill("unsaved-key")
    page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#messageArea")).to_contain_text("Could not apply settings")
    expect(key).to_have_value("unsaved-key")
    expect(key).to_be_editable()
    expect(page.locator("#applyButton")).to_be_enabled()


def test_restart_waits_for_a_new_running_server(page: Page, admin_base_url: str):
    pending: list[Route] = []
    page.route("**/admin/api/status", lambda route: pending.append(route))
    page.route(
        "**/admin/api/config/apply",
        lambda route: route.fulfill(
            json={
                "applied": True,
                "restart": {
                    "required": True,
                    "automatic": True,
                    "admin_url": "/admin",
                    "fields": ["NVIDIA_NIM_API_KEY"],
                    "instance_id": "old-server",
                },
                "credential_checks": [],
            }
        ),
    )
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.locator("#field-NVIDIA_NIM_API_KEY").fill("new-key")
    with page.expect_request("**/admin/api/status"):
        page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#applyButton")).to_have_text("Reconnecting…")
    for status in (
        {"status": "running", "instance_id": "old-server"},
        {"status": "stopping", "instance_id": "new-server"},
    ):
        with page.expect_request("**/admin/api/status"):
            pending.pop(0).fulfill(json=status)
        expect(page.locator("#view-providers")).to_have_attribute("inert", "")
        expect(page.locator("#dirtyState")).to_have_text("Changes saved")
    pending.pop(0).fulfill(json={"status": "running", "instance_id": "new-server"})
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_editable()
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    expect(page.locator("#messageArea")).to_have_text("Applied")


def test_restart_timeout_can_reconnect_without_resubmitting_keys(
    page: Page, admin_base_url: str
):
    ready = False
    submissions: list[object] = []

    def apply_result(route: Route):
        submissions.append(route.request.post_data_json)
        route.fulfill(
            json={
                "applied": True,
                "restart": {
                    "required": True,
                    "automatic": True,
                    "admin_url": "/admin",
                    "fields": ["NVIDIA_NIM_API_KEY"],
                    "instance_id": "old-server",
                },
                "credential_checks": [
                    {
                        "key": "NVIDIA_NIM_API_KEY",
                        "status": "unverified",
                        "message": "Verification unavailable.",
                    }
                ],
            }
        )

    def status_result(route: Route):
        if ready:
            route.fulfill(json={"status": "running", "instance_id": "new-server"})
        else:
            route.abort()

    page.route("**/admin/api/config/apply", apply_result)
    page.route("**/admin/api/status", status_result)
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.clock.install()
    page.locator("#field-NVIDIA_NIM_API_KEY").fill("new-key")
    with page.expect_request("**/admin/api/status"):
        page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#applyButton")).to_have_text("Reconnecting…")
    page.clock.fast_forward(31_000)
    page.clock.run_for(1_000)
    reconnect = page.get_by_role("button", name="Reconnect", exact=True)
    expect(reconnect).to_be_enabled()
    expect(page.locator("#messageArea")).to_contain_text("Settings were saved")
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
    expect(page.locator("#view-providers")).to_have_attribute("inert", "")
    expect(page.get_by_role("link", name="Open Admin")).to_have_attribute(
        "href", f"{admin_base_url}/admin"
    )
    ready = True
    reconnect.click()
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_editable()
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
    assert submissions == [{"values": {"NVIDIA_NIM_API_KEY": "new-key"}}]


def test_restart_to_another_local_origin_keeps_the_warning(
    page: Page, admin_base_url: str
):
    target = admin_base_url.replace("127.0.0.1", "localhost") + "/admin"
    page.route(
        "**/admin/api/config/apply",
        lambda route: route.fulfill(
            json={
                "applied": True,
                "restart": {
                    "required": True,
                    "automatic": True,
                    "admin_url": target,
                    "fields": ["HOST"],
                    "instance_id": "old-server",
                },
                "credential_checks": [
                    {
                        "key": "NVIDIA_NIM_API_KEY",
                        "status": "unverified",
                        "message": "Verification unavailable.",
                    }
                ],
            }
        ),
    )
    page.route(
        "**/admin/api/status",
        lambda route: route.fulfill(
            headers={"Access-Control-Allow-Origin": admin_base_url},
            json={"status": "running", "instance_id": "new-server"},
        ),
    )
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.locator("#field-NVIDIA_NIM_API_KEY").fill("new-key")
    page.get_by_role("button", name="Apply", exact=True).click()
    page.wait_for_url(target)
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_editable()
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
    assert "new-key" not in page.url
    page.reload()
    expect(page.locator("#messageArea")).to_have_text("")
