"""Shared base class for OpenAI-compatible providers (NIM, OpenRouter, LM Studio)."""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from providers.base import BaseProvider, ProviderConfig
from providers.common import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_error,
    map_stop_reason,
)
from providers.rate_limit import GlobalRateLimiter


class OpenAICompatibleProvider(BaseProvider):
    """Base class for providers using OpenAI-compatible chat completions API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        nim_settings: Any | None = None,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._nim_settings = nim_settings
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    def _build_request_body(self, request: Any) -> dict:
        """Build request body. Override in subclasses."""
        raise NotImplementedError

    def _handle_extra_reasoning(self, delta: Any, sse: SSEBuilder) -> Iterator[str]:
        """Hook for provider-specific reasoning (e.g. OpenRouter reasoning_details)."""
        return iter(())

    def _process_tool_call(self, tc: dict, sse: Any) -> Iterator[str]:
        """Process a single tool call delta and yield SSE events."""
        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_indices)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        if incoming_name is not None:
            sse.blocks.register_tool_name(tc_index, incoming_name)

        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            if name or tc.get("id"):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(
            tc_index
        ):
            tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool_block(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.get("arguments", "")
        if args:
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True

            current_name = sse.blocks.tool_names.get(tc_index, "")
            if current_name == "Task":
                parsed = sse.blocks.buffer_task_args(tc_index, args)
                if parsed is not None:
                    yield sse.emit_tool_delta(tc_index, json.dumps(parsed))
                return

            yield sse.emit_tool_delta(tc_index, args)

    def _flush_task_arg_buffers(self, sse: Any) -> Iterator[str]:
        """Emit buffered Task args as a single JSON delta (best-effort)."""
        for tool_index, out in sse.blocks.flush_task_arg_buffers():
            yield sse.emit_tool_delta(tool_index, out)

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(
                request, input_tokens, request_id
            ):
                yield event

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
    ) -> AsyncIterator[str]:
        """Shared streaming implementation."""
        tag = self._provider_name
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

        body = self._build_request_body(request)
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "%s_STREAM:%s model=%s msgs=%d tools=%d",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()

        finish_reason = None
        usage_info = None
        error_occurred = False
        error_message = ""

        async with self._global_rate_limiter.concurrency_slot():
            try:
                stream = await self._global_rate_limiter.execute_with_retry(
                    self._client.chat.completions.create, **body, stream=True
                )
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage_info = chunk.usage

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        continue

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.debug("%s finish_reason: %s", tag, finish_reason)

                    # Handle reasoning_content (OpenAI extended format)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        for event in sse.ensure_thinking_block():
                            yield event
                        yield sse.emit_thinking_delta(reasoning)

                    # Provider-specific extra reasoning (e.g. OpenRouter reasoning_details)
                    for event in self._handle_extra_reasoning(delta, sse):
                        yield event

                    # Handle text content
                    if delta.content:
                        for part in think_parser.feed(delta.content):
                            if part.type == ContentType.THINKING:
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(part.content)
                            else:
                                filtered_text, detected_tools = heuristic_parser.feed(
                                    part.content
                                )

                                if filtered_text:
                                    for event in sse.ensure_text_block():
                                        yield event
                                    yield sse.emit_text_delta(filtered_text)

                                for tool_use in detected_tools:
                                    for event in sse.close_content_blocks():
                                        yield event

                                    block_idx = sse.blocks.allocate_index()
                                    if tool_use.get("name") == "Task" and isinstance(
                                        tool_use.get("input"), dict
                                    ):
                                        tool_use["input"]["run_in_background"] = False
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
                    if delta.tool_calls:
                        for event in sse.close_content_blocks():
                            yield event
                        for tc in delta.tool_calls:
                            tc_info = {
                                "index": tc.index,
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for event in self._process_tool_call(tc_info, sse):
                                yield event

            except Exception as e:
                req_tag = f" request_id={request_id}" if request_id else ""
                logger.error("%s_ERROR:%s %s: %s", tag, req_tag, type(e).__name__, e)
                mapped_e = map_error(e)
                error_occurred = True
                error_message = str(mapped_e)
                logger.info(
                    "%s_STREAM: Emitting SSE error event for %s%s",
                    tag,
                    type(e).__name__,
                    req_tag,
                )
                for event in sse.close_content_blocks():
                    yield event
                for event in sse.emit_error(error_message):
                    yield event

        # Flush remaining content
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
            if tool_use.get("name") == "Task" and isinstance(
                tool_use.get("input"), dict
            ):
                tool_use["input"]["run_in_background"] = False
            yield sse.content_block_delta(
                block_idx,
                "input_json_delta",
                json.dumps(tool_use["input"]),
            )
            yield sse.content_block_stop(block_idx)

        if (
            not error_occurred
            and sse.blocks.text_index == -1
            and not sse.blocks.tool_indices
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in self._flush_task_arg_buffers(sse):
            yield event

        for event in sse.close_all_blocks():
            yield event

        output_tokens = (
            usage_info.completion_tokens
            if usage_info and hasattr(usage_info, "completion_tokens")
            else sse.estimate_output_tokens()
        )
        if usage_info and hasattr(usage_info, "prompt_tokens"):
            provider_input = usage_info.prompt_tokens
            if isinstance(provider_input, int):
                logger.debug(
                    "TOKEN_ESTIMATE: our=%d provider=%d diff=%+d",
                    input_tokens,
                    provider_input,
                    provider_input - input_tokens,
                )
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
