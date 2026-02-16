"""Messaging platform factory.

Creates the appropriate messaging platform adapter based on configuration.
To add a new platform (e.g. Discord, Slack):
1. Create a new class implementing MessagingPlatform in messaging/
2. Add a case to create_messaging_platform() below
"""

from typing import Optional

from loguru import logger

from .base import MessagingPlatform


def create_messaging_platform(
    platform_type: str,
    **kwargs,
) -> Optional[MessagingPlatform]:
    """Create a messaging platform instance based on type.

    Args:
        platform_type: Platform identifier ("telegram", "discord", etc.)
        **kwargs: Platform-specific configuration passed to the constructor.

    Returns:
        Configured MessagingPlatform instance, or None if not configured.
    """
    if platform_type == "telegram":
        bot_token = kwargs.get("bot_token")
        if not bot_token:
            logger.info("No Telegram bot token configured, skipping platform setup")
            return None

        from .telegram import TelegramPlatform

        return TelegramPlatform(
            bot_token=bot_token,
            allowed_user_id=kwargs.get("allowed_user_id"),
        )

    # Add new platforms here:
    # elif platform_type == "discord":
    #     from .discord import DiscordPlatform
    #     return DiscordPlatform(...)

    logger.warning(
        f"Unknown messaging platform: '{platform_type}'. Supported: 'telegram'"
    )
    return None
