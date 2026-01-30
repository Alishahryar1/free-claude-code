"""NVIDIA NIM provider - converts Anthropic format to OpenAI format for NIM."""

import logging
import os
import json
import uuid
from typing import Dict, Any, AsyncIterator

import httpx
from httpx import TimeoutException, ReadTimeout, ConnectTimeout

from .base import BaseProvider, ProviderConfig
from .utils import (
    SSEBuilder,
    map_stop_reason,
    ThinkTagParser,
    HeuristicToolParser,
    ContentType,
    extract_think_content,
    extract_reasoning_from_delta,
    AnthropicToOpenAIConverter,
)
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    OverloadedError,
    APIError,
)
from .rate_limit import GlobalRateLimiter

logger = logging.getLogger(__name__)


class NvidiaNimProvider(BaseProvider):
    """NVIDIA NIM provider using direct httpx requests."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = config.api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
        self._base_url = (
            config.base_url
            or os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        ).rstrip("/")
        self._nim_params = self._load_nim_params()
        self._global_rate_limiter = GlobalRateLimiter.get_instance()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0, read=60.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    def _load_nim_params(self) -> Dict[str, Any]:
        """Load NIM-specific parameters from environment."""
        params: Dict[str, Any] = {}
        if val := os.getenv("NVIDIA_NIM_TEMPERATURE"):
            params["temperature"] = float(val)
        if val := os.getenv("NVIDIA_NIM_TOP_P"):
            params["top_p"] = float(val)
        if val := os.getenv("NVIDIA_NIM_MAX_TOKENS"):
            params["max_tokens"] = int(val)
        return params

    def _build_request_body(self, request_data: Any, stream: bool = False) -> dict:
        """Build OpenAI-format request body from Anthropic request."""
        messages = AnthropicToOpenAIConverter.convert_messages(request_data.messages)

        # Add system prompt
        if request_data.system:
            system_msg = AnthropicToOpenAIConverter.convert_system_prompt(
                request_data.system
            )
            if system_msg:
                messages.insert(0, system_msg)

        body = {
            "model": request_data.model,
            "messages": messages,
            "max_tokens": request_data.max_tokens,
            "stream": stream,
        }

        if request_data.temperature is not None:
            body["temperature"] = request_data.temperature
        if request_data.top_p is not None:
            body["top_p"] = request_data.top_p
        if request_data.stop_sequences:
            body["stop"] = request_data.stop_sequences
        if request_data.tools:
            body["tools"] = AnthropicToOpenAIConverter.convert_tools(request_data.tools)

        # Handle thinking/reasoning mode
        extra_body = request_data.extra_body.copy() if request_data.extra_body else {}
        if request_data.thinking and getattr(request_data.thinking, "enabled", True):
            extra_body.setdefault("thinking", {"type": "enabled"})
            extra_body.setdefault("reasoning_split", True)

        body.update(extra_body)

        # Apply NIM defaults
        for key, val in self._nim_params.items():
            if key not in body:
                body[key] = val

        return body

    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        # Wait if globally rate limited
        blocked = await self._global_rate_limiter.wait_if_blocked()
        if blocked:
            # Yield error event for rate limit blocking
            message_id = f"msg_{uuid.uuid4()}"
            sse = SSEBuilder(message_id, request.model, input_tokens)
            error_msg = "⏱️ Rate limit exceeded. Please try again in a minute."
            logger.warning(f"NIM_STREAM: Rate limit blocked, yielding error event")
            yield sse.message_start()
            for event in sse.emit_error(error_msg):
                yield event
            yield sse.message_delta("stop", 0)
            yield sse.message_stop()
            yield sse.done()
            return

        body = self._build_request_body(request, stream=True)
        # Log compact request summary
        logger.info(
            f"NIM_STREAM: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}"
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)
        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()

        yield sse.message_start()

        finish_reason = None
        usage_info = None
        error_occurred = False
        error_message = ""

        try:
            async for chunk_json in self._stream_chunks(headers, body):
                # Process metadata
                if "usage" in chunk_json:
                    usage_info = chunk_json["usage"]

                if "choices" not in chunk_json or not chunk_json["choices"]:
                    continue

                choice = chunk_json["choices"][0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                    logger.debug(f"NIM finish_reason: {finish_reason}")

                # Handle reasoning content from delta
                reasoning = extract_reasoning_from_delta(delta)
                if reasoning:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(reasoning)

                # Handle text content with think tag and heuristic tool parsing
                if delta.get("content"):
                    for chunk in think_parser.feed(delta["content"]):
                        if chunk.type == ContentType.THINKING:
                            for event in sse.ensure_thinking_block():
                                yield event
                            yield sse.emit_thinking_delta(chunk.content)
                        else:
                            # Pass non-thinking text through heuristic tool parser
                            filtered_text, detected_tools = heuristic_parser.feed(
                                chunk.content
                            )

                            # Emit filtered text if any
                            if filtered_text:
                                for event in sse.ensure_text_block():
                                    yield event
                                yield sse.emit_text_delta(filtered_text)

                            # Emit detected heuristic tool calls
                            for tool_use in detected_tools:
                                for event in sse.close_content_blocks():
                                    yield event

                                block_idx = sse.blocks.allocate_index()
                                yield sse.content_block_start(
                                    block_idx,
                                    "tool_use",
                                    id=tool_use["id"],
                                    name=tool_use["name"],
                                )
                                yield sse.content_block_delta(
                                    block_idx,
                                    "input_json_delta",
                                    json.dumps(tool_use["input"]),
                                )
                                yield sse.content_block_stop(block_idx)

                # Handle native tool calls
                if delta.get("tool_calls"):
                    for event in sse.close_content_blocks():
                        yield event
                    for tc in delta["tool_calls"]:
                        for event in self._process_tool_call(tc, sse):
                            yield event

        except TimeoutException as e:
            timeout_type = "connect" if isinstance(e, ConnectTimeout) else "read"
            logger.error(f"NIM_TIMEOUT: type={timeout_type} model={body.get('model')}")
            error_occurred = True
            error_message = (
                f"⏱️ API Timeout ({timeout_type}): Request exceeded time limit"
            )
            logger.info(f"NIM_STREAM: Emitting SSE error event for timeout")
        except Exception as e:
            logger.error(f"NIM_ERROR: {type(e).__name__}: {e}")
            error_occurred = True
            error_message = str(e)
            logger.info(f"NIM_STREAM: Emitting SSE error event for exception")

        # Handle errors
        if error_occurred:
            for event in sse.emit_error(error_message):
                yield event
            logger.info(f"NIM_STREAM: Error event yielded, total events emitted")

        # Flush remaining content from parsers
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                for event in sse.ensure_thinking_block():
                    yield event
                yield sse.emit_thinking_delta(remaining.content)
            else:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(remaining.content)

        # Flush heuristic tool calls
        for tool_use in heuristic_parser.flush():
            for event in sse.close_content_blocks():
                yield event

            block_idx = sse.blocks.allocate_index()
            yield sse.content_block_start(
                block_idx,
                "tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
            )
            yield sse.content_block_delta(
                block_idx,
                "input_json_delta",
                json.dumps(tool_use["input"]),
            )
            yield sse.content_block_stop(block_idx)

        # Ensure at least one text block exists to avoid "(no content)" in Claude Code
        # This handles cases where model only returns thinking or empty responses
        if (
            not error_occurred
            and sse.blocks.text_index == -1
            and not sse.blocks.tool_indices
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")  # Single space placeholder

        # Close all blocks
        for event in sse.close_all_blocks():
            yield event

        # Final events
        output_tokens = (
            usage_info.get("completion_tokens", 0)
            if usage_info
            else sse.estimate_output_tokens()
        )
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
        yield sse.done()

    def _map_error(self, response_status: int, error_text: str) -> Exception:
        """Map HTTP status and error body to specific ProviderError."""
        try:
            error_data = json.loads(error_text)
            message = error_data.get("error", {}).get("message", error_text)
        except Exception:
            message = error_text

        if response_status == 401:
            return AuthenticationError(message, raw_error=error_text)
        if response_status == 429:
            # Trigger global rate limit block
            self._global_rate_limiter.set_blocked(60)  # Default 60s cooldown
            return RateLimitError(message, raw_error=error_text)
        if response_status in (400, 422):
            return InvalidRequestError(message, raw_error=error_text)
        if response_status >= 500:
            if "overloaded" in message.lower() or "capacity" in message.lower():
                return OverloadedError(message, raw_error=error_text)
            return APIError(message, status_code=response_status, raw_error=error_text)

        return APIError(message, status_code=response_status, raw_error=error_text)

    async def _stream_chunks(self, headers: dict, body: dict):
        """Generator that yields parsed SSE data chunks from the API."""
        async with self._client.stream(
            "POST", f"{self._base_url}/chat/completions", headers=headers, json=body
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise self._map_error(
                    response_status=response.status_code,
                    error_text=error_text.decode("utf-8", errors="replace"),
                )

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event_end = buffer.index("\n\n")
                    event_data = buffer[:event_end]
                    buffer = buffer[event_end + 2 :]

                    parsed = self._parse_sse_event(event_data)
                    if parsed is not None:
                        yield parsed

            # Flush remaining non-empty buffer
            if buffer.strip():
                parsed = self._parse_sse_event(buffer)
                if parsed is not None:
                    yield parsed

    def _parse_sse_event(self, event_data: str) -> Any:
        """Parse a single SSE event, return None if invalid/done."""
        if not event_data.strip():
            return None

        for line in event_data.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_content = line[5:].lstrip()
                if data_content == "[DONE]":
                    return None
                try:
                    return json.loads(data_content)
                except json.JSONDecodeError:
                    logger.debug(f"JSON decode failed for SSE data: {data_content}")
                    return None
        return None

    def _process_tool_call(self, tc: dict, sse: SSEBuilder):
        """Process a single tool call delta and yield SSE events."""
        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_indices)

        # Update accumulated name if present
        fn_delta = tc.get("function", {})
        if fn_delta.get("name") is not None:
            sse.blocks.tool_names[tc_index] = (
                sse.blocks.tool_names.get(tc_index, "") + fn_delta["name"]
            )

        # Check if we should start the tool block
        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            # Only start if name is non-empty or we have an ID (start of tool call)
            if name or tc.get("id"):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(
            tc_index
        ):
            # Block index exists (due to ID in previous chunk) but not started due to empty name
            tool_id = f"tool_{uuid.uuid4()}"  # Should ideally reuse ID if we saved it
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool_block(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.get("arguments", "")
        if args:
            # Ensure block is started before emitting args (with a fallback name if still empty)
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True

            yield sse.emit_tool_delta(tc_index, args)

    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request."""
        # Wait if globally rate limited
        wait_time = await self._global_rate_limiter.wait_if_blocked()
        if wait_time > 0:
            # Raise error for rate limit blocking in non-streaming mode
            error_msg = "⏱️ Rate limit exceeded. Please try again in a minute."
            logger.warning(f"NIM_COMPLETE: Rate limit blocked for {wait_time:.1f}s, raising error")
            raise APIError(error_msg, status_code=429)

        body = self._build_request_body(request, stream=False)
        # Log compact request summary
        logger.info(
            f"NIM_COMPLETE: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}"
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=body
            )
        except TimeoutException as e:
            timeout_type = "connect" if isinstance(e, ConnectTimeout) else "read"
            logger.error(f"NIM_TIMEOUT: type={timeout_type} model={body.get('model')}")
            raise APIError(
                f"API Timeout ({timeout_type}): Request exceeded time limit",
                status_code=504,
            )

        if response.status_code != 200:
            raise self._map_error(
                response_status=response.status_code, error_text=response.text
            )
        return response.json()

    def convert_response(self, response_json: dict, original_request: Any) -> dict:
        """Convert OpenAI response to Anthropic format."""
        choice = response_json["choices"][0]
        message = choice["message"]
        content = []

        # Extract reasoning from various sources
        reasoning = message.get("reasoning_content")
        if not reasoning:
            reasoning_details = message.get("reasoning_details")
            if reasoning_details and isinstance(reasoning_details, list):
                reasoning = "\n".join(
                    item.get("text", "")
                    for item in reasoning_details
                    if isinstance(item, dict)
                )

        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})

        # Extract text content (with think tag handling)
        if message.get("content"):
            raw_content = message["content"]
            if isinstance(raw_content, str):
                if not reasoning:
                    think_content, raw_content = extract_think_content(raw_content)
                    if think_content:
                        content.append({"type": "thinking", "thinking": think_content})
                if raw_content:
                    content.append({"type": "text", "text": raw_content})
            elif isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content.append(item)

        # Extract tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = tc["function"].get("arguments", {})
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    }
                )

        if not content:
            # NIM models (especially Mistral-based) often require non-empty content.
            # Adding a single space satisfies this requirement while avoiding
            # the "(no content)" display issue in Claude Code.
            content.append({"type": "text", "text": " "})

        usage = response_json.get("usage", {})

        return {
            "id": response_json.get("id", f"msg_{uuid.uuid4()}"),
            "type": "message",
            "role": "assistant",
            "model": original_request.model,
            "content": content,
            "stop_reason": map_stop_reason(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
