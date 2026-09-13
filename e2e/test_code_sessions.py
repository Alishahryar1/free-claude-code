import re
import sys
import uuid

import pytest
from playwright.sync_api import expect

from e2e.form_support import assert_autofill_opt_out
from free_claude_code.application.code_sessions.models import (
    HarnessEvent,
    PromptRequest,
)
from free_claude_code.runtime.codex_protocol import CodexProtocol
from tests.code_sessions_support import CodexPackets


@pytest.mark.parametrize("width", [1440, 900, 390])
def test_five_header_controls_fit_without_hiding_composer(
    page, admin_base_url, tmp_path, code_control, width
):
    page.set_viewport_size({"width": width, "height": 900})
    create_session(page, admin_base_url, tmp_path)
    expect(page.locator("#codeMode")).to_be_enabled()
    for control in (
        "codeProvider",
        "codeModel",
        "codeReasoning",
        "codeHarness",
        "codeMode",
    ):
        expect(page.locator(f"#{control}")).to_be_visible()
        box = page.locator(f"#{control}").bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= width
    controls = page.locator(".session-controls").bounding_box()
    composer = page.locator("#codeComposer").bounding_box()
    assert composer["y"] >= controls["y"] + controls["height"]
    assert composer["y"] + composer["height"] <= 900
    page.screenshot(path=str(tmp_path / f"code-modes-{width}.png"))


@pytest.mark.parametrize("width", [1440, 390])
def test_context_usage_is_live_persistent_and_uses_only_raw_provider_capacity(
    page, admin_base_url, tmp_path, code_control, width
):
    code_control.harness.context_windows["provider/model"] = 100_000
    code_control.harness.configurations["provider/unknown"] = "unknown"
    page.set_viewport_size({"width": width, "height": 900})
    create_session(page, admin_base_url, tmp_path)
    usage = page.locator("#codeContextUsage")
    expect(usage).to_be_hidden()

    send(page, "Measure the active context")
    connection = code_control.connection()
    code_control.run(connection.context_usage("turn-1", 90_000))
    expect(usage).to_have_text("90K / 100K (90%)")
    code_control.run(connection.context_usage("turn-1", 12_438))
    expect(usage).to_have_text("12.4K / 100K (12%)")
    expect(usage).to_have_attribute(
        "title", "Context used: 12,438 of 100,000 tokens (12%)"
    )
    expect(usage).to_have_accessible_name(
        "Context used: 12,438 of 100,000 tokens (12%)"
    )
    usage_box = usage.bounding_box()
    stop_box = page.locator("#codeStop").bounding_box()
    assert usage_box["x"] + usage_box["width"] <= stop_box["x"]
    assert stop_box["x"] + stop_box["width"] <= width

    page.reload()
    expect(usage).to_have_text("12.4K / 100K (12%)")
    code_control.run(connection.context_usage("turn-1", 104_000))
    expect(usage).to_have_text("104K / 100K (104%)")
    code_control.run(connection.finish("turn-1"))

    page.locator("#codeModel").fill("unknown")
    page.get_by_role("option", name="unknown", exact=True).click()
    expect(usage).to_have_text("104K")
    expect(usage).to_have_accessible_name("Context used: 104,000 tokens")
    send(page, "Use the model without capacity metadata")
    code_control.run(code_control.harness.wait_inputs(2))
    code_control.run(connection.context_usage("turn-2", 12_438))
    expect(usage).to_have_text("12.4K")
    expect(usage).to_have_attribute("title", "Context used: 12,438 tokens")
    expect(usage).to_have_accessible_name("Context used: 12,438 tokens")
    code_control.run(connection.finish("turn-2"))


def test_delayed_settings_response_cannot_replace_newer_context_usage(
    page, admin_base_url, tmp_path, code_control
):
    code_control.harness.context_windows["provider/model"] = 100_000
    create_session(page, admin_base_url, tmp_path)
    send(page, "Start")
    connection = code_control.connection()
    code_control.run(connection.context_usage("turn-1", 10_000))
    code_control.run(connection.finish("turn-1"))
    usage = page.locator("#codeContextUsage")
    expect(usage).to_have_text("10K / 100K (10%)")

    page.evaluate(
        """
        () => {
          const originalFetch = window.fetch;
          window.fetch = async (...args) => {
            const response = await originalFetch(...args);
            const options = args[1] || {};
            if (options.method === "PATCH" && String(args[0]).includes("/sessions/")) {
              await new Promise((resolve) => { window.releaseCodeSettings = resolve; });
            }
            return response;
          };
        }
        """
    )
    title = page.get_by_role("textbox", name="Code title", exact=True)
    title.fill("Renamed while usage changes")
    title.press("Tab")
    page.wait_for_function("() => typeof window.releaseCodeSettings === 'function'")

    code_control.run(connection.context_usage("turn-1", 20_000))
    expect(usage).to_have_text("20K / 100K (20%)")
    page.evaluate("window.releaseCodeSettings()")
    expect(title).to_be_enabled()
    expect(usage).to_have_text("20K / 100K (20%)")


def test_header_provider_draft_and_mode_sync(
    page, context, admin_base_url, tmp_path, code_control
):
    code_control.harness.configurations.update(
        {"other/vendor/model": "other", "other/second": "second"}
    )
    url = create_session(page, admin_base_url, tmp_path)
    expect(
        page.locator(
            ".session-controls .session-control > span, .session-controls .session-control > label"
        )
    ).to_have_text(["Provider", "Model", "Effort", "Harness", "Mode"])
    expect(page.locator("#codeHarness")).to_have_value("codex")
    expect(page.locator("#codeMode")).to_have_value("config")
    second = context.new_page()
    try:
        second.goto(url)
        page.locator("#codeComposer").fill("Keep this draft")
        page.locator("#codeProvider").select_option("other")
        expect(page.locator("#codeModel")).to_have_value("")
        expect(page.locator("#codeSend")).to_be_disabled()
        expect(second.locator("#codeProvider")).to_have_value("provider")
        page.locator("#codeModel").fill("vendor/model")
        page.get_by_role("option", name="vendor/model", exact=True).click()
        expect(second.locator("#codeProvider")).to_have_value("other")
        expect(second.locator("#codeModel")).to_have_value("vendor/model")
        page.locator("#codeMode").select_option("auto_review")
        expect(second.locator("#codeMode")).to_have_value("auto_review")
        page.reload()
        expect(page.locator("#codeMode")).to_have_value("auto_review")
        expect(page.locator("#codeComposer")).to_have_value("Keep this draft")
        send(page, "Use the selected mode")
        connection = code_control.connection()
        assert connection.modes == ["auto_review"]
        assert connection.inputs[0][2] == "other/vendor/model"
        for control in ("codeProvider", "codeModel", "codeReasoning", "codeMode"):
            expect(page.locator(f"#{control}")).to_be_disabled()
            expect(second.locator(f"#{control}")).to_be_disabled()
    finally:
        second.close()


@pytest.mark.parametrize("reset", ["refresh", "conflict"])
def test_unfinished_provider_selection_is_discarded_without_losing_message(
    page, context, admin_base_url, tmp_path, code_control, reset
):
    code_control.harness.configurations["other/model"] = "other"
    url = create_session(page, admin_base_url, tmp_path)
    second = context.new_page()
    try:
        second.goto(url)
        page.locator("#codeComposer").fill("Keep me")
        page.locator("#codeProvider").select_option("other")
        expect(page.locator("#codeSend")).to_be_disabled()
        if reset == "refresh":
            page.reload()
        else:
            second.locator("#codeMode").select_option("ask")
        expect(page.locator("#codeProvider")).to_have_value("provider")
        expect(page.locator("#codeModel")).to_have_value("model")
        expect(page.locator("#codeSend")).to_be_enabled()
        expect(page.locator("#codeComposer")).to_have_value("Keep me")
    finally:
        second.close()


def test_provider_change_in_another_tab_closes_old_model_options(
    page, context, admin_base_url, tmp_path, code_control
):
    code_control.harness.configurations["other/model"] = "other"
    url = create_session(page, admin_base_url, tmp_path)
    second = context.new_page()
    try:
        second.goto(url)
        page.locator("#codeModel").fill("mod")
        expect(page.get_by_role("option", name="model", exact=True)).to_be_visible()
        second.locator("#codeProvider").select_option("other")
        second.locator("#codeModel").fill("model")
        second.get_by_role("option", name="model", exact=True).click()
        expect(page.locator("#codeProvider")).to_have_value("other")
        expect(page.get_by_role("listbox")).to_be_hidden()
        expect(page.locator("#codeModel")).to_have_value("model")
    finally:
        second.close()


def test_failed_mode_patch_restores_saved_controls_and_keeps_draft(
    page, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    path = f"**/admin/api/code/sessions/{url.rsplit('/', 1)[1]}"
    page.route(
        path,
        lambda route: (
            route.fulfill(status=409, json={"detail": "Settings changed"})
            if route.request.method == "PATCH"
            else route.continue_()
        ),
    )
    page.locator("#codeComposer").fill("Keep this message")
    page.locator("#codeMode").select_option("full_access")
    expect(page.locator("#codeNotice")).to_contain_text("Settings changed")
    expect(page.locator("#codeMode")).to_have_value("config")
    expect(page.locator("#codeMode")).to_be_enabled()
    expect(page.locator("#codeComposer")).to_have_value("Keep this message")


def test_review_updates_inline_and_unfinished_review_settles_after_restart_view(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Review a command")
    connection = code_control.connection()
    protocol = CodexProtocol(connection.generation)
    payload = {
        "threadId": connection.thread_id,
        "turnId": "turn-1",
        "reviewId": "one",
        "targetItemId": "tool",
        "action": {
            "type": "command",
            "command": "echo harmless",
            "cwd": str(tmp_path),
            "source": "shell",
        },
        "review": {"status": "inProgress"},
    }
    second = context.new_page()
    try:
        second.goto(url)
        code_control.run(
            connection.sink(
                protocol.notification("item/autoApprovalReview/started", payload)
            )
        )
        review = page.locator('[data-kind="auto_review"]')
        expect(review.locator("summary")).to_have_text("Auto-review: Reviewing")
        review.locator("summary").click()
        code_control.run(
            connection.sink(
                protocol.notification(
                    "item/autoApprovalReview/completed",
                    {
                        **payload,
                        "review": {
                            "status": "approved",
                            "rationale": "Harmless local action",
                        },
                    },
                )
            )
        )
        expect(review.locator("summary")).to_have_text("Auto-review: Approved")
        expect(review.locator("details")).to_have_attribute("open", "")
        expect(review).to_contain_text("Harmless local action")
        expect(second.locator('[data-kind="auto_review"] summary')).to_have_text(
            "Auto-review: Approved"
        )
        code_control.run(
            connection.sink(
                protocol.notification(
                    "item/autoApprovalReview/started", {**payload, "reviewId": "two"}
                )
            )
        )
        code_control.run(connection.finish("turn-1", "interrupted"))
        expect(page.locator('[data-kind="auto_review"] summary')).to_have_text(
            ["Auto-review: Approved", "Auto-review: Result unavailable"]
        )
        page.reload()
        expect(page.locator('[data-kind="auto_review"] summary')).to_have_text(
            ["Auto-review: Approved", "Auto-review: Result unavailable"]
        )
    finally:
        second.close()


def test_child_review_outlives_parent_and_updates_in_place_in_both_tabs(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Delegate work")
    connection = code_control.connection()
    packets = CodexPackets(connection)
    code_control.run(packets.spawn())
    second = context.new_page()
    try:
        second.goto(url)
        code_control.run(packets.review())
        review = page.locator('[data-kind="subagent_auto_review"]')
        expect(review.locator("summary")).to_have_text(
            "Sub-agent Auto-review: Reviewing"
        )
        identity = review.get_attribute("data-id")
        review.locator("summary").click()
        code_control.run(connection.finish("turn-1"))
        second.reload()
        expect(
            second.locator('[data-kind="subagent_auto_review"] summary')
        ).to_have_text("Sub-agent Auto-review: Reviewing")
        page.locator("#codeComposer").fill("Continue while reviewing")
        expect(page.locator("#codeSend")).to_be_enabled()
        page.locator("#codeSend").click()
        code_control.run(code_control.harness.wait_inputs(2))
        code_control.run(packets.review(status="approved"))
        expect(review.locator("summary")).to_have_text(
            "Sub-agent Auto-review: Approved"
        )
        expect(review).to_have_attribute("data-id", identity)
        expect(review.locator("details")).to_have_attribute("open", "")
        expect(
            second.locator('[data-kind="subagent_auto_review"] summary')
        ).to_have_text("Sub-agent Auto-review: Approved")
        messages = page.locator(".code-item")
        expect(messages).to_have_count(3)
        expect(messages.nth(1)).to_have_attribute("data-id", identity)
        expect(messages.nth(2)).to_contain_text("Continue while reviewing")
        code_control.run(connection.finish("turn-2"))
        code_control.run(packets.review(review_id="pending"))
        code_control.run(packets.warning())
        expect(page.locator('[data-kind="notice"]')).to_contain_text(
            "Sub-agent: Native warning"
        )
        code_control.run(connection.close())
        expected = [
            "Sub-agent Auto-review: Approved",
            "Sub-agent Auto-review: Result unavailable",
        ]
        expect(page.locator('[data-kind="subagent_auto_review"] summary')).to_have_text(
            expected
        )
        expect(
            second.locator('[data-kind="subagent_auto_review"] summary')
        ).to_have_text(expected)
        page.reload()
        expect(page.locator('[data-kind="subagent_auto_review"] summary')).to_have_text(
            expected
        )
    finally:
        second.close()


@pytest.mark.parametrize(
    ("delivery", "status", "title"),
    [
        ("live", "approved", "Approved"),
        ("reconnect", "denied", "Denied"),
        ("stale_snapshot", "approved", "Approved"),
    ],
)
def test_pending_child_review_outside_page_survives_refresh_in_both_tabs(
    page, context, admin_base_url, tmp_path, code_control, delivery, status, title
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Delegate")
    connection = code_control.connection()
    packets = CodexPackets(connection)
    code_control.run(packets.spawn())
    code_control.run(packets.review())
    review = page.locator('[data-kind="subagent_auto_review"]')
    expect(review).to_have_count(1)
    identity = review.get_attribute("data-id")
    code_control.run(connection.finish("turn-1"))
    send(page, "Continue")
    code_control.run(code_control.harness.wait_inputs(2))

    async def more_output():
        for index in range(55):
            await connection.text(
                "turn-2", str(index), f"Output {index}", complete=True
            )
        await connection.finish("turn-2")

    code_control.run(more_output())
    page.reload()
    second = context.new_page()
    try:
        second.goto(url)
        for tab in (page, second):
            expect(
                tab.locator('[data-kind="subagent_auto_review"] summary')
            ).to_have_text("Sub-agent Auto-review: Reviewing")
            expect(tab.locator(".code-item").first).to_have_attribute(
                "data-id", identity
            )
        review.locator("summary").click()
        page.locator("#codeComposer").fill("Keep draft")
        if delivery == "reconnect":
            page.evaluate("window.dropCodeEvents = ['item.updated']")
        elif delivery == "stale_snapshot":
            endpoint = url.replace(admin_base_url, "").replace(
                "/admin/code/", "/admin/api/code/sessions/"
            )
            hold_code_reads(page, endpoint)
            page.evaluate("void window.codeFeed.onerror(new Event('error'))")
            page.wait_for_function(
                "path => window.codeReadHolds.some(held => held.path === path)",
                arg=endpoint,
            )
        code_control.run(packets.review(status=status))
        if delivery == "reconnect":
            page.evaluate(
                "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
            )
        elif delivery == "stale_snapshot":
            expect(review.locator("summary")).to_have_text(
                f"Sub-agent Auto-review: {title}"
            )
            page.evaluate("window.releaseCodeReads()")
        expect(page.locator("#codeSend")).to_be_enabled()
        for tab in (page, second):
            expect(
                tab.locator('[data-kind="subagent_auto_review"] summary')
            ).to_have_text(f"Sub-agent Auto-review: {title}")
            expect(tab.locator(".code-item").first).to_have_attribute(
                "data-id", identity
            )
            expect(tab.locator('[data-kind="subagent_auto_review"]')).to_have_count(1)
        expect(review.locator("details")).to_have_attribute("open", "")
        expect(page.locator("#codeComposer")).to_have_value("Keep draft")
        page.get_by_role("button", name="Load older messages", exact=True).click()
        expect(page.locator(".code-item")).to_have_count(58)
        expect(page.locator('[data-kind="subagent_auto_review"]')).to_have_count(1)
    finally:
        second.close()


def test_reobserved_child_review_outside_page_appears_without_refresh(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Delegate")
    first = code_control.connection()
    packets = CodexPackets(first)
    code_control.run(packets.spawn())
    code_control.run(packets.review())
    review = page.locator('[data-kind="subagent_auto_review"]')
    expect(review).to_have_count(1)
    identity = review.get_attribute("data-id")
    code_control.run(first.finish("turn-1"))
    code_control.run(first.close())
    send(page, "Continue")
    code_control.run(code_control.harness.wait_inputs(2))
    replacement = code_control.harness.connections[-1]

    async def more_output():
        for index in range(55):
            await replacement.text(
                "turn-2", str(index), f"Output {index}", complete=True
            )
        await replacement.finish("turn-2")

    code_control.run(more_output())
    page.reload()
    second = context.new_page()
    try:
        second.goto(url)
        for tab in (page, second):
            expect(tab.locator("#codeOlder")).to_be_visible()
            expect(tab.locator('[data-kind="subagent_auto_review"]')).to_have_count(0)
        packets = CodexPackets(replacement)
        code_control.run(packets.spawn())
        code_control.run(packets.review())
        for tab in (page, second):
            expect(
                tab.locator('[data-kind="subagent_auto_review"] summary')
            ).to_have_text("Sub-agent Auto-review: Reviewing")
            expect(tab.locator(".code-item").first).to_have_attribute(
                "data-id", identity
            )
        code_control.run(packets.review(status="approved"))
        for tab in (page, second):
            expect(
                tab.locator('[data-kind="subagent_auto_review"] summary')
            ).to_have_text("Sub-agent Auto-review: Approved")
            expect(tab.locator('[data-kind="subagent_auto_review"]')).to_have_count(1)
    finally:
        second.close()


@pytest.mark.parametrize("change", ["start", "close"])
def test_child_review_liveness_ignores_an_older_detail_snapshot(
    page, admin_base_url, tmp_path, code_control, change
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Delegate")
    connection = code_control.connection()
    packets = CodexPackets(connection)
    code_control.run(packets.spawn())
    code_control.run(connection.finish("turn-1"))
    if change == "close":
        code_control.run(packets.review())
    path = f"{admin_base_url}/admin/api/code/sessions/{url.rsplit('/', 1)[1]}"
    previous = page.request.get(path).json()
    code_control.run(packets.review() if change == "start" else connection.close())
    title = "Sub-agent Auto-review: " + (
        "Reviewing" if change == "start" else "Result unavailable"
    )
    summary = page.locator('[data-kind="subagent_auto_review"] summary')
    expect(summary).to_have_text(title)
    page.route(path, lambda route: route.fulfill(json=previous))
    page.evaluate("window.CodeSessions.refresh()")
    expect(summary).to_have_text(title)


def create_session(page, base_url, directory):
    page.goto(f"{base_url}/admin/code")
    page.get_by_role("button", name="New code session", exact=True).click()
    page.get_by_role("textbox", name="Folder", exact=True).fill(str(directory))
    expect(page.get_by_role("combobox", name="Harness", exact=True)).to_have_value(
        "codex"
    )
    page.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_enabled()
    return page.url


def send(page, text):
    page.get_by_role("textbox", name="Message", exact=True).fill(text)
    page.get_by_role("button", name="Send", exact=True).click()
    expect(page.get_by_role("button", name="Stop", exact=True)).to_be_visible()


def open_creation(page, base_url):
    page.goto(f"{base_url}/admin/code")
    page.get_by_role("button", name="New code session", exact=True).click()
    return page.get_by_role("dialog", name="New code session", exact=True)


def test_settings_apply_preserves_open_creation_and_picker(
    page, admin_base_url, tmp_path, code_control
):
    pending = []
    page.route("**/admin/api/config/apply", lambda route: pending.append(route))
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.locator("#field-NVIDIA_NIM_API_KEY").fill("new-key")
    page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#messageArea")).to_have_text("Checking API keys…")
    assert len(pending) == 1
    page.get_by_role("button", name="Code sessions", exact=True).click()
    page.get_by_role("button", name="New code session", exact=True).click()
    form = page.get_by_role("dialog", name="New code session", exact=True)
    form.get_by_role("button", name="Browse…", exact=True).click()
    call = code_control.run(code_control.folder_picker.calls.get())

    pending.pop().fulfill(json={"applied": True, "credential_checks": []})
    expect(page.locator("#messageArea")).to_have_text("Applied")
    expect(form).to_be_visible()
    code_control.run(code_control.folder_picker.finish(call, str(tmp_path)))
    expect(form.get_by_role("textbox", name="Folder", exact=True)).to_have_value(
        str(tmp_path)
    )
    form.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
    sessions = page.request.get(f"{admin_base_url}/admin/api/code/sessions").json()[
        "sessions"
    ]
    assert [session["cwd"] for session in sessions] == [str(tmp_path.resolve())]


@pytest.mark.parametrize(
    ("suffix", "display"), [("\nA", "␊A"), ("\rA", "␍A"), ("\r\nA", "␍␊A")]
)
def test_picker_preserves_line_breaks_in_hint_and_submission(
    page, admin_base_url, tmp_path, code_control, suffix, display
):
    selected = str(tmp_path / f"project{suffix}")
    form = open_creation(page, admin_base_url)
    folder = form.get_by_role("textbox", name="Folder", exact=True)
    browse = form.get_by_role("button", name="Browse…", exact=True)
    browse.click()
    call = code_control.run(code_control.folder_picker.calls.get())
    code_control.run(code_control.folder_picker.finish(call, selected))
    expect(browse).to_be_enabled()
    # Returning to Browse must use the real path, even after a native Cancel.
    browse.click()
    call = code_control.run(code_control.folder_picker.calls.get())
    assert call[0] == selected
    code_control.run(code_control.folder_picker.finish(call, None))
    with page.expect_response("**/admin/api/code/sessions") as submission:
        form.get_by_role("button", name="Create session", exact=True).click()
    assert submission.value.request.post_data_json["cwd"] == selected
    # The directory deliberately does not exist; the form stays available.
    expect(folder).to_have_value(str(tmp_path / f"project{display}"))
    expect(folder).not_to_be_editable()


@pytest.mark.parametrize("replacement", ["manual", "browse"])
def test_replacing_line_break_selection_uses_the_new_path(
    page, admin_base_url, tmp_path, code_control, replacement
):
    form = open_creation(page, admin_base_url)
    folder = form.get_by_role("textbox", name="Folder", exact=True)
    browse = form.get_by_role("button", name="Browse…", exact=True)
    browse.click()
    call = code_control.run(code_control.folder_picker.calls.get())
    code_control.run(
        code_control.folder_picker.finish(call, str(tmp_path / "project\nA"))
    )
    expect(browse).to_be_enabled()
    expect(folder).not_to_be_editable()
    if replacement == "manual":
        form.get_by_role("button", name="Enter path manually", exact=True).click()
        expect(folder).to_have_value("")
    else:
        browse.click()
        call = code_control.run(code_control.folder_picker.calls.get())
        code_control.run(code_control.folder_picker.finish(call, str(tmp_path)))
        expect(folder).to_have_value(str(tmp_path))
    expect(folder).to_be_editable()
    expect(
        form.get_by_role("button", name="Enter path manually", exact=True)
    ).to_be_hidden()
    replacement_folder = tmp_path / "manual replacement"
    replacement_folder.mkdir()
    folder.fill(str(replacement_folder))
    form.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
    sessions = page.request.get(f"{admin_base_url}/admin/api/code/sessions").json()[
        "sessions"
    ]
    assert [session["cwd"] for session in sessions] == [
        str(replacement_folder.resolve())
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows forbids CR/LF in names")
def test_picker_creates_in_selected_folder_not_its_line_break_free_neighbor(
    page, admin_base_url, tmp_path, code_control
):
    selected = tmp_path / "project\r\nA"
    selected.mkdir()
    (tmp_path / "projectA").mkdir()
    form = open_creation(page, admin_base_url)
    form.get_by_role("button", name="Browse…", exact=True).click()
    call = code_control.run(code_control.folder_picker.calls.get())
    code_control.run(code_control.folder_picker.finish(call, str(selected)))
    form.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
    sessions = page.request.get(f"{admin_base_url}/admin/api/code/sessions").json()[
        "sessions"
    ]
    assert [session["cwd"] for session in sessions] == [str(selected.resolve())]


def test_browse_fills_only_its_tab_and_waits_for_create(
    page, context, admin_base_url, tmp_path, code_control
):
    second = context.new_page()
    try:
        other = open_creation(second, admin_base_url)
        other.get_by_role("textbox", name="Folder", exact=True).fill("other tab")
        form = open_creation(page, admin_base_url)
        folder = form.get_by_role("textbox", name="Folder", exact=True)
        folder.fill(str(tmp_path))
        form.get_by_role("button", name="Browse…", exact=True).click()
        call = code_control.run(code_control.folder_picker.calls.get())
        assert call[0] == str(tmp_path)
        expect(folder).to_be_disabled()
        expect(
            form.get_by_role("button", name="Create session", exact=True)
        ).to_be_disabled()
        expect(form.get_by_role("status")).to_be_hidden()
        other.get_by_role("button", name="Browse…", exact=True).click()
        expect(
            other.get_by_text("A folder picker is already open", exact=True)
        ).to_be_visible()
        selected = tmp_path / "project café with spaces"
        selected.mkdir()
        code_control.run(code_control.folder_picker.finish(call, str(selected)))
        expect(folder).to_have_value(str(selected))
        expect(folder).to_be_enabled()
        expect(other.get_by_role("textbox", name="Folder", exact=True)).to_have_value(
            "other tab"
        )
        expect(page).to_have_url(f"{admin_base_url}/admin/code")
        assert (
            page.request.get(f"{admin_base_url}/admin/api/code/sessions").json()[
                "sessions"
            ]
            == []
        )
        other.get_by_role("button", name="Cancel", exact=True).click()
        form.get_by_role("button", name="Create session", exact=True).click()
        expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
        expect(
            second.get_by_role("button", name=re.compile("project café"))
        ).to_be_visible()
        sessions = page.request.get(f"{admin_base_url}/admin/api/code/sessions").json()[
            "sessions"
        ]
        assert [session["cwd"] for session in sessions] == [str(selected.resolve())]
    finally:
        second.close()


@pytest.mark.parametrize("failure", [False, True])
def test_picker_cancel_or_error_preserves_manual_entry(
    page, admin_base_url, tmp_path, code_control, failure
):
    form = open_creation(page, admin_base_url)
    folder = form.get_by_role("textbox", name="Folder", exact=True)
    folder.fill(str(tmp_path))
    form.get_by_role("button", name="Browse…", exact=True).click()
    call = code_control.run(code_control.folder_picker.calls.get())
    if failure:
        code_control.run(code_control.folder_picker.fail(call))
        expect(
            form.get_by_text(
                "Could not open the folder picker. Enter the path manually.", exact=True
            )
        ).to_be_visible()
    else:
        code_control.run(code_control.folder_picker.finish(call, None))
    expect(folder).to_have_value(str(tmp_path))
    expect(folder).to_be_enabled()
    form.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))


@pytest.mark.parametrize("leave", ["cancel", "escape", "refresh", "navigate", "view"])
def test_dismissing_creation_closes_native_picker(
    page, admin_base_url, code_control, leave
):
    form = open_creation(page, admin_base_url)
    form.get_by_role("button", name="Browse…", exact=True).click()
    call = code_control.run(code_control.folder_picker.calls.get())
    if leave == "cancel":
        form.get_by_role("button", name="Cancel", exact=True).click()
    elif leave == "escape":
        page.keyboard.press("Escape")
    elif leave == "refresh":
        page.reload()
    elif leave == "view":
        page.evaluate(
            "history.pushState({}, '', '/admin'); window.dispatchEvent(new PopStateEvent('popstate'))"
        )
    else:
        # Browser history works even with the modal's background made inert.
        page.go_back()
    code_control.run(call[2].wait())
    expect(form).to_have_count(0)


def test_closed_form_ignores_a_late_picker_response(
    page, admin_base_url, tmp_path, code_control
):
    form = open_creation(page, admin_base_url)
    page.evaluate("""() => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        const response = await original(...args);
        if (String(args[0]).endsWith('/folder-picker')) {
          window.pickerCaptured = true;
          await new Promise(resolve => { window.releasePicker = resolve; });
        }
        return response;
      };
    }""")
    form.get_by_role("button", name="Browse…", exact=True).click()
    call = code_control.run(code_control.folder_picker.calls.get())
    code_control.run(code_control.folder_picker.finish(call, str(tmp_path)))
    page.wait_for_function("window.pickerCaptured === true")
    form.get_by_role("button", name="Cancel", exact=True).click()
    page.get_by_role("button", name="New code session", exact=True).click()
    folder = page.get_by_role("textbox", name="Folder", exact=True)
    folder.fill("new form")
    page.evaluate("window.releasePicker()")
    expect(folder).to_have_value("new form")


def test_existing_session_keeps_streaming_while_folder_picker_is_open(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Keep running")
    connection = code_control.connection()
    observer = context.new_page()
    try:
        observer.goto(url)
        form = open_creation(page, admin_base_url)
        form.get_by_role("button", name="Browse…", exact=True).click()
        call = code_control.run(code_control.folder_picker.calls.get())
        code_control.run(
            connection.text(
                "turn-1", "reply", "Update during folder selection", complete=True
            )
        )
        expect(
            observer.get_by_text("Update during folder selection", exact=True)
        ).to_be_visible()
        code_control.run(connection.finish("turn-1"))
        expect(form).to_be_visible()
        expect(form.get_by_role("textbox", name="Folder", exact=True)).to_be_disabled()
        code_control.run(code_control.folder_picker.finish(call, None))
        expect(form.get_by_role("textbox", name="Folder", exact=True)).to_be_enabled()
    finally:
        observer.close()


@pytest.mark.parametrize("terminal", ["completed", "failed", "interrupted"])
def test_thinking_fills_quiet_gaps_without_changing_transcript(
    page, admin_base_url, tmp_path, code_control, terminal
):
    create_session(page, admin_base_url, tmp_path)
    page.clock.install(time=1000)
    page.clock.pause_at(1000)
    send(page, "Inspect this project")
    connection = code_control.connection()
    expect(page.locator("#codeStop")).to_be_visible()
    activity = page.locator("#codeActivity")
    expect(activity).to_have_count(1)
    expect(page.locator("#codeComposerStatus")).to_have_count(0)
    page.clock.run_for(499)
    expect(activity).to_be_hidden()
    page.clock.run_for(1)
    expect(activity).to_have_text("Thinking…")
    expect(activity).to_be_visible()
    for chunk in ("A", "AB", "ABC"):
        code_control.run(connection.text("turn-1", "text", chunk))
        expect(page.locator('.code-item[data-kind="text"] .code-prose')).to_have_text(
            chunk
        )
        page.clock.run_for(400)
        expect(activity).to_be_hidden()
    for kind in ("text", "reasoning", "tool"):
        for complete, text in ((False, "Streaming"), (True, "Finished")):
            code_control.run(
                connection.text("turn-1", kind, text, complete=complete, kind=kind)
            )
            item = page.locator(f'.code-item[data-kind="{kind}"]')
            expect(item.locator(".code-prose")).to_have_text(text)
            expect(activity).to_be_hidden()
            page.clock.run_for(499)
            expect(activity).to_be_hidden()
            page.clock.run_for(1)
            expect(activity).to_be_visible()
    # A duplicate update and an unrelated composer render are not streaming.
    code_control.run(
        connection.text("turn-1", "tool", "Finished", complete=True, kind="tool")
    )
    page.locator("#codeComposer").fill("Next question")
    expect(activity).to_be_visible()
    assert page.locator(".code-items #codeActivity").count() == 0
    assert (
        activity.evaluate("node => node.previousElementSibling.className")
        == "code-items"
    )
    code_control.run(connection.finish("turn-1", status=terminal))
    expect(page.locator("#codeSend")).to_be_visible()
    expect(activity).to_be_hidden()
    page.clock.run_for(1000)
    expect(activity).to_be_hidden()
    expect(page.locator(".code-item")).to_have_count(4)


@pytest.mark.parametrize("quiet_elapsed", [200, 500])
@pytest.mark.parametrize("recovered_item", ["reply", "new-reply"])
def test_recovered_output_restarts_thinking_quiet_interval(
    page, admin_base_url, tmp_path, code_control, quiet_elapsed, recovered_item
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    page.clock.install(time=1000)
    page.clock.pause_at(1000)
    send(page, "Inspect")
    connection = code_control.connection()
    code_control.run(
        connection.text("turn-1", "reply", "Before disconnect", complete=True)
    )
    expect(page.get_by_text("Before disconnect", exact=True)).to_be_visible()
    page.clock.run_for(quiet_elapsed)
    activity = page.locator("#codeActivity")
    if quiet_elapsed == 500:
        expect(activity).to_be_visible()
    else:
        expect(activity).to_be_hidden()

    page.evaluate("window.dropCodeEvents = ['item.updated']")
    code_control.run(
        connection.text("turn-1", recovered_item, "Recovered output", complete=True)
    )
    expect(page.get_by_text("Recovered output", exact=True)).to_have_count(0)
    page.evaluate("window.replayCodeReady()")
    expect(page.get_by_text("Recovered output", exact=True)).to_be_visible()
    expect(activity).to_be_hidden()
    page.clock.run_for(499)
    expect(activity).to_be_hidden()
    page.clock.run_for(1)
    expect(activity).to_be_visible()


def test_thinking_respects_prompts_refresh_navigation_and_stop(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    page.clock.install(time=1000)
    page.clock.pause_at(1000)
    send(page, "Inspect")
    connection = code_control.connection()
    code_control.run(connection.prompt(0))
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_visible()
    page.clock.run_for(1000)
    expect(page.locator("#codeActivity")).to_be_hidden()
    page.get_by_role("button", name="Allow", exact=True).click()
    expect(page.locator(".code-prompt-state")).to_have_text("Resolved")
    page.clock.run_for(499)
    expect(page.locator("#codeActivity")).to_be_hidden()
    page.clock.run_for(1)
    expect(page.locator("#codeActivity")).to_be_visible()
    code_control.run(
        connection.sink(
            HarnessEvent(
                connection.generation,
                connection.thread_id,
                "error",
                turn_id="turn-1",
                message="Upstream paused",
                will_retry=True,
            )
        )
    )
    notice = page.locator('.code-item[data-kind="notice"] .code-prose')
    expect(notice).to_have_text("Retrying… Upstream paused")
    expect(page.locator("#codeActivity")).to_be_hidden()
    page.reload()
    expect(notice).to_have_text("Retrying… Upstream paused")
    expect(page.locator(".code-prompt-state")).to_have_text("Resolved")
    page.clock.run_for(500)
    expect(page.locator("#codeActivity")).to_be_visible()
    # Replaying the connection snapshot does not count as fresh output.
    page.evaluate("window.replayCodeReady()")
    expect(page.locator("#codeStop")).to_be_enabled()
    expect(page.locator("#codeActivity")).to_be_visible()
    page.get_by_role("button", name="Providers", exact=True).click()
    page.clock.run_for(1000)
    assert page.locator("#codeActivity").evaluate("node => node.hidden")
    page.get_by_role("button", name="Code sessions", exact=True).click()
    # Return through the library if the main tab opens the library route.
    if page.url != url:
        page.locator(".session-card").first.click()
    expect(page.locator("#codeStop")).to_be_enabled()
    page.clock.run_for(499)
    expect(page.locator("#codeActivity")).to_be_hidden()
    page.clock.run_for(1)
    expect(page.locator("#codeActivity")).to_be_visible()
    code_control.harness.interrupt_gate.clear()
    page.locator("#codeStop").click()
    expect(page.locator("#codeStop")).to_be_disabled()
    page.clock.run_for(500)
    expect(page.locator("#codeActivity")).to_be_visible()
    code_control.run(release_interrupt(code_control))
    expect(page.locator(".code-outcome")).to_contain_text("Turn stopped.")
    page.clock.run_for(1000)
    expect(page.locator("#codeActivity")).to_be_hidden()


async def release_interrupt(code_control):
    code_control.harness.interrupt_gate.set()


def test_thinking_preserves_expanded_items_and_scroll_position(
    page, admin_base_url, tmp_path, code_control
):
    create_session(page, admin_base_url, tmp_path)
    page.clock.install(time=1000)
    page.clock.pause_at(1000)
    send(page, "Inspect")
    connection = code_control.connection()
    code_control.run(connection.text("turn-1", "reason", "Reading", kind="reasoning"))
    reason = page.locator(".session-thinking")
    expect(reason).to_have_count(1)
    reason.locator("summary").click()
    page.evaluate("window.savedReason = document.querySelector('.session-thinking')")
    code_control.run(connection.text("turn-1", "text", "Paragraph\n\n" * 100))
    expect(page.locator('.code-item[data-kind="text"] .code-prose')).to_contain_text(
        "Paragraph"
    )
    transcript = page.locator("#codeTranscript")
    transcript.evaluate("node => node.scrollTop = 0")
    page.clock.run_for(500)
    expect(page.locator("#codeActivity")).to_be_visible()
    assert transcript.evaluate("node => node.scrollTop") == 0
    assert reason.evaluate("node => node === window.savedReason && node.open")
    transcript.evaluate("node => node.scrollTop = node.scrollHeight")
    code_control.run(
        connection.text("turn-1", "reason", "Reading more", kind="reasoning")
    )
    expect(reason.locator(".code-prose")).to_have_text("Reading more")
    page.clock.run_for(500)
    assert (
        transcript.evaluate(
            "node => node.scrollHeight - node.scrollTop - node.clientHeight"
        )
        < 2
    )
    assert reason.evaluate("node => node === window.savedReason && node.open")


def test_deletion_error_stays_beside_delete_and_survives_refresh(
    page, admin_base_url, tmp_path, code_control
):
    create_session(page, admin_base_url, tmp_path)
    send(page, "Inspect")
    connection = code_control.connection()
    code_control.run(connection.finish("turn-1"))
    code_control.harness.delete_error = RuntimeError("Delete failed")
    page.locator("#codeDelete").click()
    page.get_by_role("button", name="Delete session", exact=True).click()
    error = page.locator("#codeDeletionError")
    expect(error).to_contain_text("You can retry deletion")
    page.reload()
    expect(error).to_contain_text("You can retry deletion")
    expect(page.locator("#codeDelete")).to_have_attribute(
        "aria-describedby", "codeDeletionError"
    )
    expect(page.locator("#codeComposerStatus")).to_have_count(0)
    code_control.harness.delete_error = None
    page.locator("#codeDelete").click()
    page.get_by_role("button", name="Delete session", exact=True).click()
    expect(page).to_have_url(f"{admin_base_url}/admin/code")


def test_code_streams_survive_refresh_and_all_viewers_leaving(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Inspect this project")
    connection = code_control.connection()
    code_control.run(
        connection.text(
            "turn-1", "reason", "Reading the files", complete=True, kind="reasoning"
        )
    )
    code_control.run(
        connection.text(
            "turn-1", "command", "directory output", complete=True, kind="tool"
        )
    )
    code_control.run(connection.text("turn-1", "reply", "First finding", complete=True))
    expect(page.get_by_text("First finding", exact=True)).to_be_visible()
    page.locator("summary").filter(has_text="Thinking").click()
    expect(page.get_by_text("Reading the files", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Regenerate", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Edit message", exact=True)).to_have_count(0)
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.get_by_text("First finding", exact=True)).to_be_visible()
        page.goto("about:blank")
        second.close()
        code_control.run(
            connection.text("turn-1", "reply", "Final finding", complete=True)
        )
        code_control.run(connection.finish("turn-1"))
        page.goto(url)
        expect(page.get_by_text("Final finding", exact=True)).to_be_visible()
        page.get_by_role("textbox", name="Message", exact=True).fill("Follow up")
        expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
        page.reload()
        expect(page.get_by_text("Final finding", exact=True)).to_be_visible()
        assert len(connection.inputs) == 1
    finally:
        if not second.is_closed():
            second.close()


def test_prompt_claim_syncs_tabs_and_stop_keeps_output(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Run a command")
    connection = code_control.connection()
    code_control.run(
        connection.text("turn-1", "reply", "Output before approval", complete=True)
    )
    code_control.run(connection.prompt(0))
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.get_by_role("button", name="Allow", exact=True)).to_be_enabled()
        page.get_by_role("button", name="Allow", exact=True).click()
        expect(second.get_by_role("button", name="Allow", exact=True)).to_be_disabled()
        code_control.run(
            connection.text("turn-1", "after", "Output after approval", complete=True)
        )
        expect(
            second.locator(
                "#codeTranscript .code-prose, #codeTranscript .code-prompt h3"
            )
        ).to_have_text(
            [
                "Run a command",
                "Output before approval",
                "Run command?",
                "Output after approval",
            ]
        )
        page.get_by_role("button", name="Stop", exact=True).click()
        page.get_by_role("textbox", name="Message", exact=True).fill("Follow up")
        expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
        expect(second.get_by_text("Output before approval", exact=True)).to_be_visible()
        assert len(connection.answers) == 1
    finally:
        second.close()


def test_old_detail_cannot_replace_streamed_output(
    page, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Keep the latest output")
    connection = code_control.connection()
    code_control.run(connection.text("turn-1", "reply", "Old output", complete=True))
    code_control.run(connection.prompt(0))
    expect(page.get_by_text("Old output", exact=True)).to_be_visible()
    page.add_init_script("""(() => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        const result = await original(...args);
        if (/\\/api\\/code\\/sessions\\/[0-9a-f-]+$/.test(new URL(String(args[0]), location.origin).pathname)) {
          window.detailCaptured = true;
          await new Promise(resolve => { window.releaseDetail = resolve; });
        }
        return result;
      };
    })();""")
    page.goto(url)
    page.wait_for_function("window.detailCaptured === true")
    code_control.run(connection.text("turn-1", "reply", "Newest output", complete=True))
    code_control.run(connection.resolve(0))
    code_control.run(connection.finish("turn-1"))
    page.evaluate("window.releaseDetail()")
    expect(page.get_by_text("Newest output", exact=True)).to_be_visible()
    expect(page.get_by_text("Old output", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_disabled()

    expect(page.locator(".code-prompt")).to_have_count(1)
    expect(page.locator(".code-run-items > .code-prompt")).to_have_count(1)
    expect(page.locator(".code-prompt-state")).to_have_text("Resolved")


def test_competing_tabs_keep_the_rejected_draft(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    second = context.new_page()
    try:
        second.goto(url)
        first_input = page.get_by_role("textbox", name="Message", exact=True)
        other_input = second.get_by_role("textbox", name="Message", exact=True)
        first_input.fill("First tab")
        other_input.fill("Second tab draft")
        expect(second.get_by_role("button", name="Send", exact=True)).to_be_enabled()
        code_control.run(code_control.hold_send())
        page.get_by_role("button", name="Send", exact=True).click()
        second.get_by_role("button", name="Send", exact=True).click()
        code_control.run(code_control.release_send())
        expect(page.get_by_role("button", name="Stop", exact=True)).to_be_visible()
        expect(second.get_by_role("button", name="Stop", exact=True)).to_be_visible()
        expect(first_input).to_have_value("")
        expect(other_input).to_have_value("Second tab draft")
        second.reload()
        expect(second.get_by_role("textbox", name="Message", exact=True)).to_have_value(
            "Second tab draft"
        )
        assert (
            sum(
                len(connection.inputs)
                for connection in code_control.harness.connections
            )
            == 1
        )
    finally:
        code_control.run(code_control.release_send())
        second.close()


def test_lost_send_response_keeps_later_typing_and_does_not_replay(
    page, admin_base_url, tmp_path, code_control
):
    create_session(page, admin_base_url, tmp_path)
    page.evaluate("""() => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        const result = await original(...args);
        if (String(args[0]).endsWith('/turns')) {
          window.sendCaptured = true;
          await new Promise((resolve, reject) => { window.loseSend = () => reject(new TypeError('Connection lost')); });
        }
        return result;
      };
    }""")
    send(page, "Run once")
    page.wait_for_function("window.sendCaptured === true")
    draft = page.get_by_role("textbox", name="Message", exact=True)
    draft.fill("Keep my next message")
    page.evaluate("window.loseSend()")
    connection = code_control.connection()
    code_control.run(connection.finish("turn-1"))
    page.reload()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_have_value(
        "Keep my next message"
    )
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
    assert len(connection.inputs) == 1


def test_rename_and_delete_sync_library_without_touching_project(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    project = tmp_path / "project.txt"
    project.write_text("unchanged")
    second = context.new_page()
    try:
        second.goto(url)
        page.get_by_role("textbox", name="Code title", exact=True).fill("My project")
        page.get_by_role("textbox", name="Code title", exact=True).press("Enter")
        expect(
            second.get_by_role("textbox", name="Code title", exact=True)
        ).to_have_value("My project")
        page.get_by_role("button", name="Delete", exact=True).click()
        page.get_by_role("button", name="Delete session", exact=True).click()
        expect(page).to_have_url(f"{admin_base_url}/admin/code")
        expect(second).to_have_url(f"{admin_base_url}/admin/code")
        expect(second.locator(".session-card")).to_have_count(0)
        assert project.read_text() == "unchanged"
    finally:
        second.close()


def test_answered_question_keeps_its_place_across_turns_tabs_and_refresh(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First request")
    connection = code_control.connection()
    code_control.run(
        connection.text("turn-1", "before", "Before question", complete=True)
    )
    prompt = PromptRequest(
        7,
        "questions",
        {
            "title": "Question at this point",
            "questions": [
                {
                    "id": "task",
                    "label": "Which task?",
                    "options": [{"label": "Search", "description": "Search docs"}],
                    "allow_other": False,
                    "secret": False,
                }
            ],
        },
        {},
        "turn-1",
        "question-without-a-native-transcript-item",
    )

    async def ask():
        connection.requests[7] = prompt
        await connection.sink(
            HarnessEvent(
                connection.generation,
                connection.thread_id,
                "prompt",
                turn_id="turn-1",
                prompt=prompt,
            )
        )

    code_control.run(ask())
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.get_by_text("Which task?", exact=True)).to_be_visible()
        page.get_by_role("radio").check()
        page.get_by_role("button", name="Submit answers", exact=True).click()
        expect(second.locator(".code-prompt-state")).to_have_text("Resolved")
        code_control.run(
            connection.text("turn-1", "after", "After answer", complete=True)
        )
        code_control.run(connection.finish("turn-1"))
        send(page, "Second request")
        code_control.run(code_control.harness.wait_inputs(2))
        code_control.run(
            connection.text("turn-2", "reply", "Second reply", complete=True)
        )
        code_control.run(connection.finish("turn-2"))
        expected = [
            "First request",
            "Before question",
            "Which task?",
            "After answer",
            "Second request",
            "Second reply",
        ]
        for tab in (page, second):
            expect(tab.get_by_text("Second reply", exact=True)).to_be_visible()
            entries = tab.locator(
                "#codeTranscript .code-prose, #codeTranscript .code-prompt legend"
            )
            expect(entries).to_have_text(expected)
            tab.reload()
            expect(entries).to_have_text(expected)
            expect(tab.locator(".code-prompt-state")).to_have_text("Resolved")
        assert connection.answers == [(7, {"answers": {"task": ["Search"]}})]
    finally:
        second.close()


def test_question_input_survives_streaming_and_secret_answer_is_not_stored(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    send(page, "Ask me a question")
    connection = code_control.connection()
    assert_autofill_opt_out(page)
    prompt = PromptRequest(
        7,
        "questions",
        {
            "title": "Codex needs your input",
            "questions": [
                {
                    "id": "destination",
                    "label": "Which destination?",
                    "header": "Destination",
                    "options": [
                        {
                            "label": "Default",
                            "description": "Use the current destination",
                        }
                    ],
                    "allow_other": True,
                    "secret": True,
                }
            ],
        },
        {"question": "destination"},
        "turn-1",
    )

    async def ask():
        connection.requests[7] = prompt
        await connection.sink(
            HarnessEvent(
                connection.generation,
                connection.thread_id,
                "prompt",
                turn_id="turn-1",
                prompt=prompt,
            )
        )

    code_control.run(ask())
    secret = page.get_by_label("Your answer", exact=True)
    secret.fill("private-answer")
    expect(secret).to_have_attribute("type", "password")
    assert_autofill_opt_out(page)
    secret.focus()
    secret.evaluate(
        "input => { window.savedPromptInput = input; input.setSelectionRange(2, 6); }"
    )
    code_control.run(
        connection.text("turn-1", "stream", "Still checking", complete=True)
    )
    expect(page.get_by_text("Still checking", exact=True)).to_be_visible()
    expect(secret).to_have_value("private-answer")
    expect(secret).to_be_focused()
    assert secret.evaluate(
        "input => input === window.savedPromptInput && input.selectionStart === 2 && input.selectionEnd === 6"
    )
    page.evaluate("window.codeFeed.onerror(new Event('error'))")
    expect(
        page.get_by_role("button", name="Submit answers", exact=True)
    ).to_be_enabled()
    expect(secret).to_have_value("private-answer")
    assert secret.evaluate(
        "input => input === window.savedPromptInput && input.selectionStart === 2 && input.selectionEnd === 6"
    )
    expect(
        page.locator("#codeTranscript .code-prompt legend, #codeTranscript .code-prose")
    ).to_have_text(
        [
            "Ask me a question",
            "Which destination?",
            "Still checking",
        ]
    )
    page.get_by_role("button", name="Submit answers", exact=True).click()
    expect(
        page.get_by_role("button", name="Submit answers", exact=True)
    ).to_be_disabled()
    code_control.run(code_control.harness.answered.wait())
    assert connection.answers == [(7, {"answers": {"destination": ["private-answer"]}})]
    assert "private-answer" not in page.evaluate("JSON.stringify(sessionStorage)")
    detail = code_control.run(
        code_control.service.get_detail(page.url.rsplit("/", 1)[1])
    )
    assert all(
        "private-answer" not in value.model_dump_json()
        for value in (*detail.prompts, *detail.items)
    )


def delete_from_other_client(code_control, session_id):
    async def remove():
        detail = await code_control.service.get_detail(session_id)
        await code_control.service.delete_session(session_id, detail.session.revision)
        await code_control.service.wait_idle(session_id)

    code_control.run(remove())


def hold_deleted_detail(page, code_control, session_id):
    page.evaluate("window.dropCodeEvents = ['session.deleted']")
    delete_from_other_client(code_control, session_id)
    endpoint = f"/admin/api/code/sessions/{session_id}"
    hold_code_reads(page, endpoint)
    page.evaluate("window.replayCodeReady()")
    page.wait_for_function(
        "path => window.codeReadHolds.some(held => held.path === path && held.status === 404)",
        arg=endpoint,
    )


@pytest.mark.parametrize("source", ["event", "detail404"])
def test_hidden_code_removal_preserves_providers_and_deleted_history(
    page, admin_base_url, tmp_path, code_control, source
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    session_id = url.rsplit("/", 1)[1]
    if source == "detail404":
        hold_deleted_detail(page, code_control, session_id)
    page.get_by_role("button", name="Providers", exact=True).click()
    providers_url = f"{admin_base_url}/admin"
    expect(page).to_have_url(providers_url)
    field = page.locator("#field-MISTRAL_API_KEY")
    field.fill("Keep this provider edit")
    field.evaluate("input => input.setSelectionRange(2, 7)")
    history_length = page.evaluate("history.length")
    if source == "event":
        delete_from_other_client(code_control, session_id)
        page.wait_for_function("id => window.lastCodeDeleted === id", arg=session_id)
    else:
        page.evaluate("window.releaseCodeReads()")
        expect(page.locator("#codeLibrary")).to_have_count(1)
    expect(page).to_have_url(providers_url)
    assert page.evaluate("history.length") == history_length
    expect(page.locator("#view-providers")).to_be_visible()
    expect(page.locator('.nav-link[data-view="providers"]')).to_have_attribute(
        "aria-current", "page"
    )
    expect(field).to_be_focused()
    expect(field).to_have_value("Keep this provider edit")
    assert field.evaluate("input => [input.selectionStart,input.selectionEnd]") == [
        2,
        7,
    ]
    reads = []
    endpoint = f"/admin/api/code/sessions/{session_id}"
    page.on("request", lambda request: reads.append(request.url))
    page.go_back()
    expect(page).to_have_url(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeLibrary .session-card")).to_have_count(0)
    assert not any(request.endswith(endpoint) for request in reads)
    assert page.evaluate("history.length") == history_length
    page.go_forward()
    expect(page).to_have_url(providers_url)
    expect(field).to_have_value("Keep this provider edit")


@pytest.mark.parametrize("source", ["event", "detail404", "returned_detail404"])
def test_visible_code_removal_replaces_history(
    page, admin_base_url, tmp_path, code_control, source
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    session_id = url.rsplit("/", 1)[1]
    if source != "event":
        hold_deleted_detail(page, code_control, session_id)
    if source == "returned_detail404":
        page.get_by_role("button", name="Providers", exact=True).click()
        page.go_back()
        expect(page).to_have_url(url)
        expect(page.locator("#view-code")).to_be_visible()
    history_length = page.evaluate("history.length")
    if source == "event":
        delete_from_other_client(code_control, session_id)
        page.wait_for_function("id => window.lastCodeDeleted === id", arg=session_id)
    else:
        page.evaluate("window.releaseCodeReads()")
    expect(page).to_have_url(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNotice")).to_contain_text("deleted")
    expect(page.locator("#codeNew")).to_be_enabled()
    assert page.evaluate("history.length") == history_length
    if source == "returned_detail404":
        page.go_forward()
        expect(page).to_have_url(f"{admin_base_url}/admin")
        page.go_back()
        expect(page).to_have_url(f"{admin_base_url}/admin/code")
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    page.go_back()
    expect(page).to_have_url(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNew")).to_be_enabled()
    assert not any(request.endswith(f"/sessions/{session_id}") for request in reads)


@pytest.mark.parametrize("source", ["event", "detail404"])
def test_code_removal_does_not_redirect_another_code_session(
    page, admin_base_url, tmp_path, code_control, source
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    session_id = url.rsplit("/", 1)[1]
    other = code_control.run(
        code_control.service.create_session(str(uuid.uuid4()), str(tmp_path))
    )
    if source == "detail404":
        hold_deleted_detail(page, code_control, session_id)
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    page.locator(f'#codeLibrary [data-id="{other.id}"]').click()
    other_url = f"{admin_base_url}/admin/code/{other.id}"
    expect(page).to_have_url(other_url)
    composer = page.locator("#codeComposer")
    composer.fill("Keep B's draft")
    composer.evaluate("input => input.setSelectionRange(2, 7)")
    history_length = page.evaluate("history.length")
    if source == "event":
        delete_from_other_client(code_control, session_id)
        page.wait_for_function("id => window.lastCodeDeleted === id", arg=session_id)
    else:
        page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeSend")).to_be_enabled()
    expect(page).to_have_url(other_url)
    expect(composer).to_have_value("Keep B's draft")
    expect(composer).to_be_focused()
    assert composer.evaluate("input => [input.selectionStart,input.selectionEnd]") == [
        2,
        7,
    ]
    assert page.evaluate("history.length") == history_length
    page.reload()
    expect(page).to_have_url(other_url)
    expect(composer).to_have_value("Keep B's draft")


def control_feed(page):
    page.add_init_script("""(() => {
      const Native = window.EventSource;
      window.EventSource = class extends Native {
        constructor(...args) {
          super(...args);
          this.isCode = String(args[0]).includes('/api/code/events');
          if (this.isCode) window.codeFeed = this;
        }
        addEventListener(type, listener, ...options) {
          return super.addEventListener(type, event => {
            if (this.isCode && window.dropCodeEvents?.includes(type)) return;
            if (this.isCode && type === 'feed.ready')
              window.replayCodeReady = () => listener(event);
            listener(event);
            if (this.isCode && type === 'run.updated')
              window.lastCodeRun = JSON.parse(event.data).run;
            if (this.isCode && type === 'session.deleted')
              window.lastCodeDeleted = JSON.parse(event.data).session_id;
          }, ...options);
        }
      };
    })();""")


def hold_code_reads(page, endpoint=None):
    page.evaluate(
        """endpoint => {
      const original = window.fetch;
      window.codeReadHolds = [];
      window.holdCodeReads = true;
      window.fetch = async (...args) => {
        const response = await original(...args);
        const path = new URL(String(args[0]), location.origin).pathname;
        if (window.holdCodeReads && path.startsWith('/admin/api/code/') &&
            (!endpoint || path === endpoint) &&
            (!args[1]?.method || args[1].method === 'GET'))
          await new Promise(resolve => window.codeReadHolds.push({path, status: response.status, resolve}));
        return response;
      };
      window.releaseCodeReads = () => {
        window.holdCodeReads = false;
        for (const held of window.codeReadHolds.splice(0)) held.resolve();
      };
    }""",
        endpoint,
    )


def send_from_other_client(page, url, text):
    endpoint = url.replace("/admin/code/", "/admin/api/code/sessions/")
    detail = page.request.get(endpoint).json()
    response = page.request.post(
        f"{endpoint}/turns",
        data={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": detail["session"]["revision"],
            "expected_epoch": detail["epoch"],
            "text": text,
        },
    )
    assert response.ok, response.text()
    return response.json()["id"]


@pytest.mark.parametrize("choice", ["off", "low", "medium", "high", "xhigh", "max"])
def test_effort_options_persist_and_reach_the_harness(
    page, admin_base_url, tmp_path, code_control, choice
):
    create_session(page, admin_base_url, tmp_path)
    effort = page.get_by_role("combobox", name="Effort", exact=True)
    expect(effort.locator("option")).to_have_text(
        ["off", "low", "medium", "high", "xhigh", "max"]
    )
    expect(effort).to_have_value("medium")
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH" and "/api/code/sessions/" in response.url
        )
    ) as changed:
        effort.select_option(choice)
    assert changed.value.json()["reasoning_effort"] == choice
    page.reload()
    expect(effort).to_have_value(choice)
    send(page, "Use this effort")
    assert code_control.connection().efforts == [choice]


def test_off_clears_effort_when_reasoning_becomes_unavailable(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    effort = page.locator("#codeReasoning")
    effort.select_option("high")
    expect(effort).to_be_enabled()
    page.locator("#codeComposer").fill("Preserve this draft")
    code_control.harness.efforts = ("off",)
    code_control.harness.default_effort = "off"
    page.evaluate("window.CodeSessions.refresh()")
    expect(page.locator("#codeSelectionError")).to_contain_text("unavailable")
    expect(effort).to_have_value("high")
    expect(effort.locator("option:checked")).to_have_text("high")
    expect(effort.locator("option:disabled")).to_have_text(
        ["low", "medium", "high", "xhigh", "max"]
    )
    expect(page.locator("#codeSend")).to_be_disabled()
    expect(effort).to_be_enabled()
    patches = []
    page.on(
        "request",
        lambda request: (
            patches.append(request.post_data_json)
            if request.method == "PATCH" and "/api/code/sessions/" in request.url
            else None
        ),
    )
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH" and "/api/code/sessions/" in response.url
        )
    ) as reset:
        effort.select_option("off")
    assert reset.value.json()["reasoning_effort"] == "off"
    assert len(patches) == 1 and patches[0]["reasoning_effort"] == "off"
    expect(page.locator("#codeSelectionError")).to_be_hidden()
    expect(page.locator("#codeSend")).to_be_enabled()
    page.reload()
    expect(effort).to_have_value("off")
    expect(page.locator("#codeComposer")).to_have_value("Preserve this draft")
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.locator("#codeReasoning")).to_have_value("off")
        expect(second.locator("#codeModel")).to_have_value("model")
        page.locator("#codeSend").click()
        connection = code_control.connection()
        assert connection.efforts == ["off"]
        assert connection.inputs[0][1:] == ("Preserve this draft", "provider/model")
    finally:
        second.close()


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_library_reconnect_retires_missed_completion_and_loads_real_outcome(
    page, admin_base_url, tmp_path, code_control, status
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    send(page, "Cached task")
    connection = code_control.connection()
    code_control.run(
        connection.text("turn-1", "reply", "Preserved output", complete=True)
    )
    expect(page.get_by_text("Preserved output", exact=True)).to_be_visible()
    page.locator("#codeComposer").fill("Next draft")
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    card = page.locator("#codeLibrary .session-card")
    expect(card).to_contain_text("Running")
    page.evaluate("window.dropCodeEvents = ['run.updated', 'session.updated']")
    code_control.run(
        connection.finish(
            "turn-1", status, "Actual failure" if status == "failed" else None
        )
    )
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(card).not_to_contain_text("Running")
    hold_code_reads(page)
    card.click()
    page.wait_for_function(
        "window.codeReadHolds.some(held => /\\/sessions\\/[0-9a-f-]+$/.test(held.path))"
    )
    expect(page.locator("#codeSend")).to_be_disabled()
    expect(page.locator("#codeModel")).to_be_disabled()
    expect(page.get_by_text("Preserved output", exact=True)).to_be_visible()
    expect(page.locator("#codeComposer")).to_have_value("Next draft")
    page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeSend")).to_be_enabled()
    expect(page.locator(".code-run")).to_have_count(1)
    outcome = page.locator(".code-outcome")
    if status == "failed":
        expect(outcome.locator(".session-generation-status")).to_have_text(
            "Actual failure"
        )
        expect(outcome).to_be_visible()
    else:
        expect(outcome).to_be_hidden()


@pytest.mark.parametrize("snapshot_active", [False, True])
def test_new_run_survives_delayed_reconnect_reads(
    page, admin_base_url, tmp_path, code_control, snapshot_active
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First run")
    connection = code_control.connection()
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    if not snapshot_active:
        page.evaluate("window.dropCodeEvents = ['run.updated', 'session.updated']")
        code_control.run(connection.finish("turn-1"))
    hold_code_reads(page)
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    page.wait_for_function(
        "window.codeReadHolds.some(held => held.path === '/admin/api/code/sessions')"
    )
    if snapshot_active:
        expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
        code_control.run(connection.finish("turn-1"))
    run_id = send_from_other_client(page, url, "Newer run")
    code_control.run(code_control.harness.wait_inputs(2))
    page.wait_for_function(
        "id => window.lastCodeRun?.id === id && window.lastCodeRun.status === 'running'",
        arg=run_id,
    )
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")


@pytest.mark.parametrize("source", ["sse", "detail"])
def test_stale_ready_snapshot_preserves_newer_activity(
    page, admin_base_url, tmp_path, code_control, source
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First run")
    connection = code_control.connection()
    code_control.run(connection.finish("turn-1"))
    expect(page.locator("#codeStop")).to_be_hidden()
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    page.evaluate("window.codeFeed.onerror(new Event('error'))")
    expect(page.locator("#codeNew")).to_be_enabled()
    if source == "detail":
        page.evaluate(
            "window.dropCodeEvents = ['run.updated', 'session.updated', 'item.updated']"
        )
    run_id = send_from_other_client(page, url, "Newer run")
    code_control.run(code_control.harness.wait_inputs(2))
    if source == "detail":
        page.locator("#codeLibrary .session-card").click()
        expect(page.locator("#codeStop")).to_be_visible()
        expect(page.get_by_text("Newer run", exact=True)).to_be_visible()
        page.get_by_role("button", name="← Code sessions", exact=True).click()
    else:
        page.wait_for_function(
            "id => window.lastCodeRun?.id === id && window.lastCodeRun.status === 'running'",
            arg=run_id,
        )
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    hold_code_reads(page)
    page.evaluate("window.replayCodeReady()")
    page.wait_for_function("window.codeReadHolds.length >= 2")
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")


@pytest.mark.parametrize("endpoint", ["sessions", "bootstrap"])
def test_first_http_sync_failure_recovers_without_reloading(
    page, admin_base_url, endpoint
):
    page.add_init_script(
        """(endpoint => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        if (!window.failedCodeLoad && new URL(String(args[0]), location.origin).pathname === `/admin/api/code/${endpoint}`) {
          window.failedCodeLoad = true;
          throw new TypeError('Initial snapshot failed');
        }
        return original(...args);
      };
    })("""
        + repr(endpoint)
        + ");"
    )
    page.goto(f"{admin_base_url}/admin/code")
    expect(
        page.get_by_role("button", name="New code session", exact=True)
    ).to_be_enabled()
    assert page.evaluate("window.failedCodeLoad")
    expect(page.locator("#codeNotice")).to_be_hidden()


def test_unavailable_feed_shows_reason_and_clears_it_on_recovery(
    page, admin_base_url, code_control
):
    async def availability(available):
        code_control.service._accepting = available
        code_control.service._message = (
            None if available else "Code storage is unavailable"
        )

    code_control.run(availability(False))
    page.goto(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNotice")).to_contain_text("Code storage is unavailable")
    expect(page.locator("#codeNew")).to_be_disabled()
    code_control.run(availability(True))
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeNotice")).to_be_hidden()


def test_reconnect_retires_out_of_page_prompt_without_discarding_live_form_input(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    send(page, "Work")
    connection = code_control.connection()
    code_control.run(connection.prompt(0, turn_id=None))
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_enabled()
    page.evaluate("window.dropCodeEvents = ['prompt.updated']")
    code_control.run(connection.resolve(0))
    for index in range(55):
        code_control.run(
            connection.text("turn-1", str(index), f"Line {index}", complete=True)
        )
    code_control.run(connection.finish("turn-1"))
    page.get_by_role("textbox", name="Message", exact=True).fill("Keep draft")
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_disabled()
    expect(page.get_by_text("No longer active", exact=True)).to_be_visible()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_have_value(
        "Keep draft"
    )


def test_failed_reply_stays_after_its_output_once_and_not_in_composer(
    page, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First")
    connection = code_control.connection()
    code_control.run(
        connection.text(
            "turn-1", "reasoning", "Reasoning so far", kind="reasoning", complete=True
        )
    )
    code_control.run(
        connection.text("turn-1", "tool", "Tool so far", kind="tool", complete=True)
    )
    code_control.run(connection.text("turn-1", "reply", "Partial reply", complete=True))
    code_control.run(
        connection.finish("turn-1", "failed", "Provider refused the request")
    )
    expect(
        page.locator(".code-outcome").filter(has_text="Provider refused the request")
    ).to_have_count(1)
    expect(page.locator("#codeComposerStatus")).to_have_count(0)
    expect(page.locator("#codeNotice")).to_be_hidden()
    send(page, "Second")
    code_control.run(connection.finish("turn-2"))
    page.goto(url)
    expect(page.locator(".code-run")).to_have_count(2)
    expect(page.locator(".code-run").first).to_contain_text("Partial reply")
    expect(page.locator(".code-run").first.locator(".code-outcome")).to_contain_text(
        "Provider refused"
    )
    expect(page.locator(".code-run").last).to_contain_text("Second")


@pytest.mark.parametrize("refresh_first", [True, False])
def test_model_effort_picker_syncs_tabs_and_keeps_missing_selection(
    page, context, admin_base_url, tmp_path, code_control, refresh_first
):
    code_control.harness.configurations["provider/other"] = "other"
    url = create_session(page, admin_base_url, tmp_path)
    second = context.new_page()
    try:
        second.goto(url)
        page.get_by_role("combobox", name="Selected model", exact=True).fill("other")
        page.get_by_role("option", name="other", exact=True).click()
        expect(second.locator("#codeModel")).to_have_value("other")
        page.locator("#codeReasoning").select_option("high")
        expect(second.locator("#codeReasoning")).to_have_value("high")
        send(page, "Use choice")
        connection = code_control.connection()
        assert connection.inputs[0][2] == "provider/other"
        assert connection.efforts == ["high"]
        expect(page.locator("#codeModel")).to_be_disabled()
        expect(second.locator("#codeReasoning")).to_be_disabled()
        code_control.run(connection.finish("turn-1"))
        page.locator("#codeReasoning").select_option("medium")
        expect(second.locator("#codeReasoning")).to_have_value("medium")
        page.get_by_role("textbox", name="Message", exact=True).fill("Preserve me")
        code_control.harness.configurations.pop("provider/other")
        if refresh_first:
            page.evaluate("window.CodeSessions.refresh()")
        else:
            page.locator("#codeSend").click()
            expect(page.locator("#codeNotice")).to_contain_text("unavailable")
        expect(page.locator("#codeModel")).to_have_value("other")
        expect(page.locator("#codeSend")).to_be_disabled()
        expect(page.locator("#codeComposer")).to_have_value("Preserve me")
        expect(page.locator("#codeSelectionError")).to_contain_text("unavailable")
    finally:
        second.close()


def test_library_reconnect_removes_missed_deletion_and_searches_beyond_first_page(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)

    async def seed():
        first = await code_control.service.create_session(
            str(uuid.uuid4()), str(tmp_path)
        )
        await code_control.service.update_settings(
            first.id, first.revision, {"title": "Needle project"}
        )
        for _ in range(26):
            await code_control.service.create_session(str(uuid.uuid4()), str(tmp_path))
        return first.id

    # Start the isolated service before seeding it.
    page.goto(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNew")).to_be_enabled()
    session_id = code_control.run(seed())
    page.get_by_role("searchbox", name="Search titles and folders").fill("Needle")
    expect(page.locator(".session-card")).to_have_count(1)
    expect(page.locator(".session-card")).to_contain_text("Needle project")
    page.evaluate("window.dropCodeEvents = ['session.deleted']")

    async def remove_seed():
        detail = await code_control.service.get_detail(session_id)
        await code_control.service.delete_session(session_id, detail.session.revision)
        await code_control.service.wait_idle(session_id)

    code_control.run(remove_seed())
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    expect(page.locator(".session-card")).to_have_count(0)
    expect(page.get_by_text("No matching sessions.", exact=True)).to_be_visible()


@pytest.mark.parametrize("width,height", [(1440, 1000), (390, 844)])
def test_code_composer_fits_viewport_and_preserves_focus(
    page, admin_base_url, tmp_path, code_control, width, height
):
    page.set_viewport_size({"width": width, "height": height})
    create_session(page, admin_base_url, tmp_path)
    composer = page.locator("#codeComposer")
    composer.fill("a draft\nwith a second line")
    composer.evaluate("input => input.setSelectionRange(2, 6)")
    page.locator("#codeModel").click()
    page.keyboard.press("Escape")
    composer.focus()
    composer.evaluate("input => input.setSelectionRange(2, 6)")
    page.evaluate("window.CodeSessions.refresh()")
    expect(composer).to_be_focused()
    assert composer.evaluate("input => [input.selectionStart,input.selectionEnd]") == [
        2,
        6,
    ]
    assert (
        page.locator("#codeComposer").bounding_box()["y"]
        + page.locator("#codeComposer").bounding_box()["height"]
        <= height
    )
