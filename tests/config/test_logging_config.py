"""Tests for config/logging_config.py."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from free_claude_code.config import logging_config
from free_claude_code.config.logging_config import configure_logging


def test_configure_logging_creates_parent_directories(tmp_path) -> None:
    """Nested log path: parent directories are created before truncating."""
    log_file = tmp_path / "nested" / "dir" / "app.log"
    configure_logging(str(log_file), force=True)
    assert log_file.is_file()


def test_configure_logging_writes_json_to_file(tmp_path):
    """configure_logging writes JSON lines to the specified file."""
    log_file = str(tmp_path / "test.log")
    configure_logging(log_file, force=True)

    # Emit a log via stdlib (intercepted to loguru)
    logger = logging.getLogger("test.module")
    logger.info("Test message for JSON")

    # Force flush - loguru may buffer
    from loguru import logger as loguru_logger

    loguru_logger.complete()

    content = Path(log_file).read_text(encoding="utf-8")
    lines = [line for line in content.strip().split("\n") if line]
    assert len(lines) >= 1

    # Each line should be valid JSON
    for line in lines:
        record = json.loads(line)
        assert "text" in record or "message" in record or "record" in record


def test_configure_logging_idempotent(tmp_path):
    """configure_logging is idempotent - safe to call twice with force."""
    log_file = str(tmp_path / "test.log")
    configure_logging(log_file, force=True)
    configure_logging(log_file, force=True)  # Should not raise

    logger = logging.getLogger("test.idempotent")
    logger.info("After second configure")


def test_configure_logging_replaces_sink_when_path_changes(tmp_path):
    """A path-only restart switches destinations without truncating either file."""
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "nested" / "second.log"
    configure_logging(first_log, force=True)

    logger.info("first destination")
    from loguru import logger as loguru_logger

    loguru_logger.complete()
    configure_logging(second_log)
    logger.info("second destination")
    loguru_logger.complete()

    first_text = first_log.read_text(encoding="utf-8")
    second_text = second_log.read_text(encoding="utf-8")
    assert "first destination" in first_text
    assert "second destination" not in first_text
    assert "second destination" in second_text


def test_telegram_bot_token_redacted_in_message_field(tmp_path) -> None:
    log_file = str(tmp_path / "redact.log")
    configure_logging(log_file, force=True, verbose_third_party=False)
    token = "123456:ABCDEF-ghij-klm"
    logger.info("Calling {}", f"https://api.telegram.org/bot{token}/getMe")
    logger.complete()
    text = Path(log_file).read_text(encoding="utf-8")
    assert token not in text
    assert "bot<redacted>/" in text or "redacted" in text


def test_bearer_substring_redacted_in_log_file(tmp_path) -> None:
    log_file = str(tmp_path / "bearer.log")
    configure_logging(log_file, force=True, verbose_third_party=False)
    secret = "ya29.secret-token-abc"
    logger.info("Request headers: Authorization: Bearer {}", secret)
    logger.complete()
    text = Path(log_file).read_text(encoding="utf-8")
    assert secret not in text
    assert "Bearer" in text


def test_noisy_third_party_loggers_quieted_when_not_verbose(tmp_path) -> None:
    log_file = str(tmp_path / "quiet.log")
    configure_logging(log_file, force=True, verbose_third_party=False)
    assert logging.getLogger("openai").level >= logging.WARNING
    assert (
        logging.getLogger("openai._base_client").getEffectiveLevel() >= logging.WARNING
    )
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING


def test_noisy_third_party_loggers_reset_when_verbose(tmp_path) -> None:
    log_file = str(tmp_path / "verbose.log")
    configure_logging(log_file, force=True, verbose_third_party=True)
    assert logging.getLogger("openai").level == logging.NOTSET
    assert logging.getLogger("httpx").level == logging.NOTSET


def test_openai_request_payload_debug_requires_verbose_third_party(tmp_path) -> None:
    log_file = tmp_path / "openai.log"
    marker = "Request options: sensitive-prompt"
    openai_logger = logging.getLogger("openai._base_client")

    configure_logging(
        log_file,
        force=True,
        level="DEBUG",
        verbose_third_party=False,
    )
    openai_logger.debug(marker)
    logger.complete()
    assert marker not in log_file.read_text(encoding="utf-8")

    configure_logging(
        log_file,
        force=True,
        level="DEBUG",
        verbose_third_party=True,
    )
    openai_logger.debug(marker)
    logger.complete()
    assert marker in log_file.read_text(encoding="utf-8")


def test_configure_logging_respects_level(tmp_path) -> None:
    """INFO level suppresses DEBUG messages from the file sink."""
    log_file = str(tmp_path / "level.log")
    configure_logging(log_file, force=True, level="INFO")

    logger.debug("should not appear")
    logger.info("should appear")
    logger.complete()

    text = Path(log_file).read_text(encoding="utf-8")
    assert "should appear" in text
    assert "should not appear" not in text


def test_configure_logging_defaults_to_info(tmp_path) -> None:
    """Customer default keeps lifecycle logs while suppressing debug diagnostics."""
    log_file = str(tmp_path / "default.log")
    configure_logging(log_file, force=True)

    logger.debug("debug message")
    logger.info("info message")
    logger.complete()

    text = Path(log_file).read_text(encoding="utf-8")
    assert "debug message" not in text
    assert "info message" in text


def test_file_sink_bounds_rotated_log_retention(tmp_path) -> None:
    """Five archives plus the active 50 MB file bound normal disk usage."""
    log_file = tmp_path / "bounded.log"

    with patch.object(logging_config.logger, "add", return_value=1) as add:
        logging_config._add_file_sink(log_file, "INFO")

    assert add.call_args.kwargs["rotation"] == "50 MB"
    assert add.call_args.kwargs["retention"] == 5


def test_configure_logging_handles_level_change_on_restart(tmp_path) -> None:
    """Restart with a different level replaces the sink without truncating."""
    log_file = str(tmp_path / "restart.log")

    # First call: DEBUG level
    configure_logging(log_file, force=True, level="DEBUG")
    logger.debug("first debug")
    logger.complete()

    # Second call: change to INFO (simulates supervised restart)
    configure_logging(log_file, level="INFO")
    logger.debug("second debug")
    logger.info("second info")
    logger.complete()

    text = Path(log_file).read_text(encoding="utf-8")
    # First DEBUG message was written before the level change
    assert "first debug" in text
    # Second DEBUG is suppressed by the new INFO level
    assert "second debug" not in text
    # INFO messages appear regardless
    assert "second info" in text


def test_configure_logging_skips_when_level_unchanged(tmp_path) -> None:
    """When level hasn't changed, the existing sink is reused."""
    log_file = str(tmp_path / "same.log")
    configure_logging(log_file, force=True, level="WARNING")
    logger.warning("first warning")
    logger.complete()

    # Second call with same level — should be a no-op for the sink
    configure_logging(log_file, level="WARNING")
    logger.warning("second warning")
    logger.complete()

    text = Path(log_file).read_text(encoding="utf-8")
    assert "first warning" in text
    assert "second warning" in text


def test_configure_logging_updates_verbosity_on_same_level(tmp_path) -> None:
    """When verbose_third_party changes but level stays the same, third-party
    logger levels are updated without touching the file sink."""
    log_file = str(tmp_path / "verbosity.log")

    # Start with verbosity off (third-party loggers at WARNING)
    configure_logging(log_file, force=True, level="DEBUG", verbose_third_party=False)
    assert logging.getLogger("openai").level >= logging.WARNING
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
    assert logging.getLogger("telegram").level >= logging.WARNING

    # Restart with same level but verbosity on
    configure_logging(log_file, level="DEBUG", verbose_third_party=True)
    assert logging.getLogger("openai").level == logging.NOTSET
    assert logging.getLogger("httpx").level == logging.NOTSET
    assert logging.getLogger("httpcore").level == logging.NOTSET
    assert logging.getLogger("telegram").level == logging.NOTSET

    # Log file sink still works
    logger.info("still logging")
    logger.complete()
    assert "still logging" in Path(log_file).read_text(encoding="utf-8")
