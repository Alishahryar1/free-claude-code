from free_claude_code.messaging.rendering.profiles import build_rendering_profile


def test_discord_rendering_profile_has_plain_parse_mode():
    profile = build_rendering_profile("discord")

    assert profile.parse_mode is None
    assert profile.limit_chars == 1900
    assert profile.format_status("x", "Working", None).startswith("x")


def test_telegram_rendering_profile_uses_markdown_v2():
    profile = build_rendering_profile("telegram")

    assert profile.parse_mode == "MarkdownV2"
    assert profile.limit_chars == 3900
    assert profile.format_status("x", "Working", None).startswith("x")
