"""Optimization handlers for fast-path API responses.

Each handler returns a MessagesResponse if the request matches and the
optimization is enabled, otherwise None.
"""

import uuid
from typing import Optional

from .models.anthropic import MessagesRequest
from .models.responses import MessagesResponse, Usage
from .detection import (
    is_quota_check_request,
    is_title_generation_request,
    is_prefix_detection_request,
    is_suggestion_mode_request,
    is_filepath_extraction_request,
)
from .command_utils import extract_command_prefix, extract_filepaths_from_command
from config.settings import Settings
from loguru import logger


def try_prefix_detection(
    request_data: MessagesRequest, settings: Settings
) -> Optional[MessagesResponse]:
    """Fast prefix detection - return command prefix without API call."""
    if not settings.fast_prefix_detection:
        return None

    is_prefix_req, command = is_prefix_detection_request(request_data)
    if not is_prefix_req:
        return None

    return MessagesResponse(
        id=f"msg_{uuid.uuid4()}",
        model=request_data.model,
        content=[{"type": "text", "text": extract_command_prefix(command)}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=100, output_tokens=5),
    )


def try_quota_mock(
    request_data: MessagesRequest, settings: Settings
) -> Optional[MessagesResponse]:
    """Mock quota probe requests."""
    if not settings.enable_network_probe_mock:
        return None
    if not is_quota_check_request(request_data):
        return None

    logger.info("Optimization: Intercepted and mocked quota probe")
    return MessagesResponse(
        id=f"msg_{uuid.uuid4()}",
        model=request_data.model,
        role="assistant",
        content=[{"type": "text", "text": "Quota check passed."}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def try_title_skip(
    request_data: MessagesRequest, settings: Settings
) -> Optional[MessagesResponse]:
    """Skip title generation requests."""
    if not settings.enable_title_generation_skip:
        return None
    if not is_title_generation_request(request_data):
        return None

    logger.info("Optimization: Skipped title generation request")
    return MessagesResponse(
        id=f"msg_{uuid.uuid4()}",
        model=request_data.model,
        role="assistant",
        content=[{"type": "text", "text": "Conversation"}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=100, output_tokens=5),
    )


def try_suggestion_skip(
    request_data: MessagesRequest, settings: Settings
) -> Optional[MessagesResponse]:
    """Skip suggestion mode requests."""
    if not settings.enable_suggestion_mode_skip:
        return None
    if not is_suggestion_mode_request(request_data):
        return None

    logger.info("Optimization: Skipped suggestion mode request")
    return MessagesResponse(
        id=f"msg_{uuid.uuid4()}",
        model=request_data.model,
        role="assistant",
        content=[{"type": "text", "text": ""}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=100, output_tokens=1),
    )


def try_filepath_mock(
    request_data: MessagesRequest, settings: Settings
) -> Optional[MessagesResponse]:
    """Mock filepath extraction requests."""
    if not settings.enable_filepath_extraction_mock:
        return None

    is_fp, cmd, output = is_filepath_extraction_request(request_data)
    if not is_fp:
        return None

    filepaths = extract_filepaths_from_command(cmd, output)
    logger.info("Optimization: Mocked filepath extraction")
    return MessagesResponse(
        id=f"msg_{uuid.uuid4()}",
        model=request_data.model,
        role="assistant",
        content=[{"type": "text", "text": filepaths}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=100, output_tokens=10),
    )


# Cheapest/most common optimizations first for faster short-circuit.
OPTIMIZATION_HANDLERS = [
    try_quota_mock,
    try_prefix_detection,
    try_title_skip,
    try_suggestion_skip,
    try_filepath_mock,
]


def try_optimizations(
    request_data: MessagesRequest, settings: Settings
) -> Optional[MessagesResponse]:
    """Run optimization handlers in order. Returns first match or None."""
    for handler in OPTIMIZATION_HANDLERS:
        result = handler(request_data, settings)
        if result is not None:
            return result
    return None
