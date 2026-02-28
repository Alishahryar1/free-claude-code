import re
import uuid
from enum import Enum
from typing import Any

from loguru import logger

# Some OpenAI-compatible backends/models occasionally leak internal sentinel tokens
# into `delta.content` (e.g. "<|tool_call_end|>"). These should never be shown to
# end users, and they can disrupt downstream parsing if left in place.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{1,80}\|>")
_CONTROL_TOKEN_START = "<|"
_CONTROL_TOKEN_END = "|>"


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


class HeuristicToolParser:
    """
    Stateful parser that detects raw text tool calls in the format:
    ● <function=Name><parameter=key>value</parameter>...

    This is used as a fallback for models that emit tool calls as text
    instead of using the structured API.
    """

    # Class-level compiled patterns (compiled once, not per instance)
    _FUNC_START_PATTERN = re.compile(r"●\s*<function=([^>]+)>")
    _PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL
    )

    def __init__(self):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id = None
        self._current_function_name = None
        self._current_parameters = {}

    def _strip_control_tokens(self, text: str) -> str:
        # Remove complete sentinel tokens. If a token is split across chunks it
        # will be removed once the buffer contains the full token.
        return _CONTROL_TOKEN_RE.sub("", text)

    def _split_incomplete_control_token_tail(self) -> str:
        """
        If the buffer ends with an incomplete "<|...|>" sentinel token, keep that
        fragment in the buffer and return the safe-to-emit prefix.

        This prevents leaking raw sentinel fragments to the user when streaming.
        """
        start = self._buffer.rfind(_CONTROL_TOKEN_START)
        if start == -1:
            return ""
        end = self._buffer.find(_CONTROL_TOKEN_END, start)
        if end != -1:
            return ""

        prefix = self._buffer[:start]
        self._buffer = self._buffer[start:]
        return prefix

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Feed text into the parser.
        Returns a tuple of (filtered_text, detected_tool_calls).

        filtered_text: Text that should be passed through as normal message content.
        detected_tools: List of Anthropic-format tool_use blocks.
        """
        self._buffer += text
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        filtered_output_parts: list[str] = []

        while True:
            if self._state == ParserState.TEXT:
                # Look for the trigger character
                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    filtered_output_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    # Avoid emitting an incomplete "<|...|>" sentinel fragment if the
                    # token got split across streaming chunks.
                    safe_prefix = self._split_incomplete_control_token_tail()
                    if safe_prefix:
                        filtered_output_parts.append(safe_prefix)
                        break

                    filtered_output_parts.append(self._buffer)
                    self._buffer = ""
                    break

            if self._state == ParserState.MATCHING_FUNCTION:
                # We need enough buffer to match the function tag
                # e.g. "● <function=Grep>"
                match = self._FUNC_START_PATTERN.search(self._buffer)
                if match:
                    self._current_function_name = match.group(1).strip()
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}

                    # Consume the function start from buffer
                    self._buffer = self._buffer[match.end() :]
                    self._state = ParserState.PARSING_PARAMETERS
                    logger.debug(
                        "Heuristic bypass: Detected start of tool call '{}'",
                        self._current_function_name,
                    )
                else:
                    # If we have "●" but not the full tag yet, wait for more data
                    # Unless the buffer has grown too large without a match
                    if len(self._buffer) > 100:
                        # Probably not a tool call, treat as text
                        filtered_output_parts.append(self._buffer[0])
                        self._buffer = self._buffer[1:]
                        self._state = ParserState.TEXT
                    else:
                        break

            if self._state == ParserState.PARSING_PARAMETERS:
                # Look for parameters. We look for </parameter> to know a param is complete.
                # Or wait for another <parameter or the end of the text if it seems complete.

                # If we see a newline followed by anything other than <parameter or spaces,
                # we might be done with the tool call.

                finished_tool_call = False

                # Check if we have any complete parameters
                while True:
                    param_match = self._PARAM_PATTERN.search(self._buffer)
                    if param_match and "</parameter>" in param_match.group(0):
                        # Detect any content before the parameter match and preserve it
                        pre_match_text = self._buffer[: param_match.start()]
                        if pre_match_text:
                            filtered_output_parts.append(pre_match_text)

                        key = param_match.group(1).strip()
                        val = param_match.group(2).strip()
                        self._current_parameters[key] = val
                        self._buffer = self._buffer[param_match.end() :]
                    else:
                        break

                # Heuristic for completion:
                # 1. We have at least one param and we see a character that doesn't belong to the format
                # 2. Significant pause (not handled here, handled by caller via flush if needed)
                # 3. Another ● character (start of NEXT tool call)

                if "●" in self._buffer:
                    # Next tool call starting or something else, close current
                    # But first, capture any text before the ●
                    idx = self._buffer.find("●")
                    if idx > 0:
                        filtered_output_parts.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    finished_tool_call = True
                elif len(self._buffer) > 0 and not self._buffer.strip().startswith("<"):
                    # We have text that doesn't look like a tag, and we already parsed some or are in param state
                    # Let's see if we have trailing param starts
                    if "<parameter=" not in self._buffer:
                        # Treat the buffer as text (it's not a parameter)
                        # But wait, we are in PARSING_PARAMETERS.
                        # If we have " some text", we should emit it and finish tool call.
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True

                if finished_tool_call:
                    # Emit the tool call
                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": self._current_function_name,
                            "input": self._current_parameters,
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        self._current_function_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                    # Continue loop to process remaining buffer (which is empty or starts with ●)
                else:
                    break

        return "".join(filtered_output_parts), detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """
        Flush any remaining tool calls in the buffer.
        """
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        if self._state == ParserState.PARSING_PARAMETERS:
            # Try to extract any partial parameters remaining in buffer
            # Even without </parameter>
            partial_matches = re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            )
            for m in partial_matches:
                key = m.group(1).strip()
                val = m.group(2).strip()
                self._current_parameters[key] = val

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": self._current_tool_id,
                    "name": self._current_function_name,
                    "input": self._current_parameters,
                }
            )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
