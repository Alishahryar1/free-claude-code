"""Messaging platform component factory."""

from dataclasses import dataclass

from loguru import logger

from ..limiter import MessagingRateLimiter
from ..voice import Transcriber
from .ports import MessagingPlatformComponents, MessagingStartupNotice


@dataclass(frozen=True, slots=True)
class MessagingPlatformOptions:
    """Typed wiring from app settings into messaging platform runtimes."""

    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    telegram_proxy_url: str | None = None
    discord_bot_token: str | None = None
    allowed_discord_channels: str | None = None
    transcriber: Transcriber | None = None
    messaging_rate_limit: int = 1
    messaging_rate_window: float = 1.0
    log_raw_messaging_content: bool = False
    log_messaging_error_details: bool = False
    log_api_error_tracebacks: bool = False


def create_messaging_components(
    platform_type: str,
    options: MessagingPlatformOptions | None = None,
) -> MessagingPlatformComponents | None:
    """Create runtime/outbound components for the configured messaging platform."""
    opts = options or MessagingPlatformOptions()
    if platform_type == "none":
        logger.info("Messaging platform disabled by configuration")
        return None

    if platform_type == "telegram":
        bot_token = opts.telegram_bot_token
        if not bot_token:
            logger.info("No Telegram bot token configured, skipping platform setup")
            return None

        from .telegram import TelegramRuntime

        limiter = MessagingRateLimiter(
            rate_limit=opts.messaging_rate_limit,
            rate_window=opts.messaging_rate_window,
            log_error_details=opts.log_messaging_error_details,
        )
        runtime = TelegramRuntime(
            bot_token=bot_token,
            allowed_user_id=opts.allowed_telegram_user_id,
            telegram_proxy_url=opts.telegram_proxy_url,
            limiter=limiter,
            transcriber=opts.transcriber,
            log_raw_messaging_content=opts.log_raw_messaging_content,
            log_api_error_tracebacks=opts.log_api_error_tracebacks,
        )
        startup_notice = (
            MessagingStartupNotice(
                chat_id=opts.allowed_telegram_user_id,
                transport_label="Bot API",
            )
            if opts.allowed_telegram_user_id
            else None
        )
        return MessagingPlatformComponents(
            name=runtime.name,
            runtime=runtime,
            outbound=runtime.outbound,
            voice_cancellation=runtime,
            startup_notice=startup_notice,
        )

    if platform_type == "discord":
        bot_token = opts.discord_bot_token
        if not bot_token:
            logger.info("No Discord bot token configured, skipping platform setup")
            return None

        from .discord import DiscordRuntime

        limiter = MessagingRateLimiter(
            rate_limit=opts.messaging_rate_limit,
            rate_window=opts.messaging_rate_window,
            log_error_details=opts.log_messaging_error_details,
        )
        runtime = DiscordRuntime(
            bot_token=bot_token,
            allowed_channel_ids=opts.allowed_discord_channels,
            limiter=limiter,
            transcriber=opts.transcriber,
            log_raw_messaging_content=opts.log_raw_messaging_content,
            log_api_error_tracebacks=opts.log_api_error_tracebacks,
        )
        return MessagingPlatformComponents(
            name=runtime.name,
            runtime=runtime,
            outbound=runtime.outbound,
            voice_cancellation=runtime,
        )

    logger.warning(
        "Unknown messaging platform: '{}'. Supported: 'none', 'telegram', 'discord'",
        platform_type,
    )
    return None
