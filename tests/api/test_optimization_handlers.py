"""Tests for api/optimization_handlers.py."""

from unittest.mock import patch

from api.models.anthropic import ContentBlockText, Message, MessagesRequest
from api.optimization_handlers import (
    try_filepath_mock,
    try_optimizations,
    try_prefix_detection,
    try_quota_mock,
    try_suggestion_skip,
    try_title_skip,
)
from config.settings import Settings


def _make_request(
    messages_content: str, max_tokens: int | None = None
) -> MessagesRequest:
    """Create a MessagesRequest with a single user message."""
    return MessagesRequest(
        model="claude-3-sonnet",
        max_tokens=max_tokens if max_tokens is not None else 100,
        messages=[Message(role="user", content=messages_content)],
    )


class TestTryPrefixDetection:
    def test_disabled_returns_none(self):
        settings = Settings()
        settings.fast_prefix_detection = False
        req = _make_request("x")
        with patch(
            "api.optimization_handlers.is_prefix_detection_request",
            return_value=(True, "/ask"),
        ):
            assert try_prefix_detection(req, settings) is None

    def test_enabled_and_match_returns_response(self):
        settings = Settings()
        settings.fast_prefix_detection = True
        req = _make_request("x")
        with (
            patch(
                "api.optimization_handlers.is_prefix_detection_request",
                return_value=(True, "/ask"),
            ),
            patch(
                "api.optimization_handlers.extract_command_prefix",
                return_value="/ask",
            ),
            patch("api.optimization_handlers.logger.info") as mock_log_info,
        ):
            result = try_prefix_detection(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert block.text == "/ask"
        mock_log_info.assert_called_once_with(
            "Optimization: Fast prefix detection request"
        )

    def test_enabled_but_no_match_returns_none(self):
        settings = Settings()
        settings.fast_prefix_detection = True
        req = _make_request("x")
        with patch(
            "api.optimization_handlers.is_prefix_detection_request",
            return_value=(False, ""),
        ):
            assert try_prefix_detection(req, settings) is None


class TestTryQuotaMock:
    def test_disabled_returns_none(self):
        settings = Settings()
        settings.enable_network_probe_mock = False
        req = _make_request("quota", max_tokens=1)
        with patch(
            "api.optimization_handlers.is_quota_check_request",
            return_value=True,
        ):
            assert try_quota_mock(req, settings) is None

    def test_enabled_and_match_returns_response(self):
        settings = Settings()
        settings.enable_network_probe_mock = True
        req = _make_request("quota", max_tokens=1)
        with patch(
            "api.optimization_handlers.is_quota_check_request",
            return_value=True,
        ):
            result = try_quota_mock(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert "Quota check passed" in block.text


class TestTryTitleSkip:
    def test_disabled_returns_none(self):
        settings = Settings()
        settings.enable_title_generation_skip = False
        req = _make_request("write a 5-10 word title")
        with patch(
            "api.optimization_handlers.is_title_generation_request",
            return_value=True,
        ):
            assert try_title_skip(req, settings) is None

    def test_enabled_and_match_returns_response(self):
        settings = Settings()
        settings.enable_title_generation_skip = True
        req = _make_request("x")
        with patch(
            "api.optimization_handlers.is_title_generation_request",
            return_value=True,
        ):
            result = try_title_skip(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert block.text == "Conversation"


class TestTrySuggestionSkip:
    def test_disabled_returns_none(self):
        settings = Settings()
        settings.enable_suggestion_mode_skip = False
        req = _make_request("[SUGGESTION MODE: x]")
        with patch(
            "api.optimization_handlers.is_suggestion_mode_request",
            return_value=True,
        ):
            assert try_suggestion_skip(req, settings) is None

    def test_enabled_and_match_returns_response(self):
        settings = Settings()
        settings.enable_suggestion_mode_skip = True
        req = _make_request("x")
        with patch(
            "api.optimization_handlers.is_suggestion_mode_request",
            return_value=True,
        ):
            result = try_suggestion_skip(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert block.text == ""


class TestTryFilepathMock:
    def test_disabled_returns_none(self):
        settings = Settings()
        settings.enable_filepath_extraction_mock = False
        req = _make_request("Command:\nls\nOutput:\nfilepaths")
        with patch(
            "api.optimization_handlers.is_filepath_extraction_request",
            return_value=(True, "ls", "out"),
        ):
            assert try_filepath_mock(req, settings) is None

    def test_enabled_and_match_returns_response(self):
        settings = Settings()
        settings.enable_filepath_extraction_mock = True
        req = _make_request("x")
        with (
            patch(
                "api.optimization_handlers.is_filepath_extraction_request",
                return_value=(True, "ls", "a.txt b.txt"),
            ),
            patch(
                "api.optimization_handlers.extract_filepaths_from_command",
                return_value="a.txt\nb.txt",
            ),
        ):
            result = try_filepath_mock(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert block.text == "a.txt\nb.txt"

    def test_extract_filepaths_empty_list_still_returns_response(self):
        settings = Settings()
        settings.enable_filepath_extraction_mock = True
        req = _make_request("x")
        with (
            patch(
                "api.optimization_handlers.is_filepath_extraction_request",
                return_value=(True, "ls", "out"),
            ),
            patch(
                "api.optimization_handlers.extract_filepaths_from_command",
                return_value="",
            ),
        ):
            result = try_filepath_mock(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert block.text == ""


class TestTryOptimizations:
    def test_first_match_wins(self):
        """Quota mock is first in OPTIMIZATION_HANDLERS; it should win over prefix."""
        settings = Settings()
        settings.enable_network_probe_mock = True
        settings.fast_prefix_detection = True
        req = _make_request("quota", max_tokens=1)
        with patch(
            "api.optimization_handlers.is_quota_check_request",
            return_value=True,
        ):
            result = try_optimizations(req, settings)
        assert result is not None
        block = result.content[0]
        assert isinstance(block, ContentBlockText)
        assert "Quota check passed" in block.text

    def test_no_match_returns_none(self):
        settings = Settings()
        settings.fast_prefix_detection = False
        settings.enable_network_probe_mock = False
        settings.enable_title_generation_skip = False
        settings.enable_suggestion_mode_skip = False
        settings.enable_filepath_extraction_mock = False
        req = _make_request("random user message")
        assert try_optimizations(req, settings) is None
