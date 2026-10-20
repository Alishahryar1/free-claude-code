"""Anthropic server-tool block types shared by decoders and formatters."""

from typing import Final

SERVER_TOOL_USE: Final = "server_tool_use"
WEB_SEARCH_TOOL_RESULT: Final = "web_search_tool_result"
WEB_FETCH_TOOL_RESULT: Final = "web_fetch_tool_result"
WEB_SEARCH_TOOL_RESULT_ERROR: Final = "web_search_tool_result_error"
WEB_FETCH_TOOL_ERROR: Final = "web_fetch_tool_error"
