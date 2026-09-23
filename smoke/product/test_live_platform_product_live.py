import os
import time

import httpx
import pytest

from smoke.lib.config import SmokeConfig

pytestmark = [pytest.mark.live]


@pytest.mark.smoke_target("telegram")
def test_telegram_live_permissions_e2e(smoke_config: SmokeConfig) -> None:
    token = smoke_config.settings.telegram_bot_token
    if not token:
        pytest.skip("missing_env: TELEGRAM_BOT_TOKEN is not configured")

    base_url = f"https://api.telegram.org/bot{token}"
    get_me = httpx.get(f"{base_url}/getMe", timeout=smoke_config.timeout_s)
    assert get_me.status_code == 200, get_me.text
    assert get_me.json()["ok"] is True

    chat_id = os.getenv("FCC_SMOKE_TELEGRAM_CHAT_ID") or (
        smoke_config.settings.allowed_telegram_user_id or ""
    )
    if not chat_id:
        pytest.skip(
            "missing_env: FCC_SMOKE_TELEGRAM_CHAT_ID or "
            "ALLOWED_TELEGRAM_USER_ID required"
        )

    marker = f"FCC product smoke {int(time.time())}"
    sent = httpx.post(
        f"{base_url}/sendMessage",
        json={"chat_id": chat_id, "text": marker},
        timeout=smoke_config.timeout_s,
    )
    assert sent.status_code == 200, sent.text
    message_id = sent.json()["result"]["message_id"]

    edited = httpx.post(
        f"{base_url}/editMessageText",
        json={"chat_id": chat_id, "message_id": message_id, "text": marker + " edit"},
        timeout=smoke_config.timeout_s,
    )
    assert edited.status_code == 200, edited.text

    deleted = httpx.post(
        f"{base_url}/deleteMessage",
        json={"chat_id": chat_id, "message_id": message_id},
        timeout=smoke_config.timeout_s,
    )
    assert deleted.status_code == 200, deleted.text


@pytest.mark.smoke_target("discord")
def test_discord_live_permissions_e2e(smoke_config: SmokeConfig) -> None:
    token = smoke_config.settings.discord_bot_token
    channel_id = os.getenv("FCC_SMOKE_DISCORD_CHANNEL_ID")
    if not channel_id and smoke_config.settings.allowed_discord_channels:
        channel_id = smoke_config.settings.allowed_discord_channels.split(",", 1)[0]
    if not token:
        pytest.skip("missing_env: DISCORD_BOT_TOKEN is not configured")
    if not channel_id:
        pytest.skip(
            "missing_env: FCC_SMOKE_DISCORD_CHANNEL_ID or "
            "ALLOWED_DISCORD_CHANNELS required"
        )

    headers = {"authorization": f"Bot {token}"}
    base_url = "https://discord.com/api/v10"

    channel = httpx.get(
        f"{base_url}/channels/{channel_id}",
        headers=headers,
        timeout=smoke_config.timeout_s,
    )
    assert channel.status_code == 200, channel.text

    marker = f"FCC product smoke {int(time.time())}"
    sent = httpx.post(
        f"{base_url}/channels/{channel_id}/messages",
        headers=headers,
        json={"content": marker},
        timeout=smoke_config.timeout_s,
    )
    assert sent.status_code == 200, sent.text
    message_id = sent.json()["id"]

    edited = httpx.patch(
        f"{base_url}/channels/{channel_id}/messages/{message_id}",
        headers=headers,
        json={"content": marker + " edit"},
        timeout=smoke_config.timeout_s,
    )
    assert edited.status_code == 200, edited.text

    deleted = httpx.delete(
        f"{base_url}/channels/{channel_id}/messages/{message_id}",
        headers=headers,
        timeout=smoke_config.timeout_s,
    )
    assert deleted.status_code in {200, 204}, deleted.text


@pytest.mark.smoke_target("telegram")
def test_telegram_live_manual_inbound_e2e(smoke_config: SmokeConfig) -> None:
    if not smoke_config.interactive:
        pytest.skip("missing_env: set FCC_SMOKE_INTERACTIVE=1 for manual inbound smoke")
    pytest.skip(
        "manual product smoke: send the printed nonce to Telegram and verify "
        "progress edits, final transcript, /stop, /clear, /stats, and voice note "
        "behavior from the real client"
    )


@pytest.mark.smoke_target("discord")
def test_discord_live_manual_inbound_e2e(smoke_config: SmokeConfig) -> None:
    if not smoke_config.interactive:
        pytest.skip("missing_env: set FCC_SMOKE_INTERACTIVE=1 for manual inbound smoke")
    pytest.skip(
        "manual product smoke: send the printed nonce to Discord and verify "
        "progress edits, final transcript, /stop, /clear, /stats, and voice note "
        "behavior from the real client"
    )
