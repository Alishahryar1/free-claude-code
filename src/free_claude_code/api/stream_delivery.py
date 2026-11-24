"""Run the public response iterator and commit each projected packet."""

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from copy import deepcopy

from free_claude_code.core.async_iterators import try_close_async_iterator
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.stream_output import StreamOutput

from .generation_paths import WireApi
from .stream_publication import PacketPublication


class DeliveryObservedStream(AsyncIterator[StreamOutput]):
    """Bind observation for each read/close, including prefetch in another task."""

    def __init__(
        self, body: AsyncIterator[StreamOutput], state: StreamDeliveryState
    ) -> None:
        self._body = body
        self._state = state

    def __aiter__(self) -> DeliveryObservedStream:
        return self

    async def __anext__(self) -> StreamOutput:
        with bind_stream_delivery(self._state):
            return await anext(self._body)

    async def aclose(self) -> None:
        with bind_stream_delivery(self._state):
            error = await try_close_async_iterator(self._body)
        if error is not None:
            raise error


class PublicResponseStream(AsyncIterator[str]):
    """Publish one response while providers retain their physical request loops."""

    def __init__(
        self,
        body: AsyncIterator[StreamOutput],
        state: StreamDeliveryState,
        *,
        wire_api: WireApi,
        first_chunk: StreamOutput,
        terminal_frame: Callable[[str, str, BaseException], str] | None,
        terminal_failure_observer: Callable[[BaseException], None] | None,
        hide_reasoning: bool = False,
    ) -> None:
        self._body = body
        self.state = state
        self._wire_api = wire_api
        self._revision = state.attempt_revision
        self._hide_reasoning = hide_reasoning
        self._publication = PacketPublication(wire_api, hide_reasoning=hide_reasoning)
        state.bind_prefix(lambda: self._publication.snapshot())
        self._initial_chunk = "".join(first_chunk.frames)
        self._latest_chunk = ""
        self._terminal_frame = terminal_frame
        self._terminal_failure_observer = terminal_failure_observer
        self._done = False
        self._closed = False
        self._events = self._iterate()

    def __aiter__(self) -> PublicResponseStream:
        return self

    async def __anext__(self) -> str:
        if self._closed or self._done:
            raise StopAsyncIteration
        try:
            while True:
                packet, synthetic = await anext(self._events)
                value = self._publish_packet(packet, synthetic=synthetic)
                if value is not None:
                    self._latest_chunk = value
                    return value
        except StopAsyncIteration:
            self._done = True
            raise
        except (Exception, BaseExceptionGroup) as exc:
            self._done = True
            self._publication.discard_held_starts()
            failure = find_execution_failure(exc) or exc
            if self._terminal_frame is None:
                raise
            if self._terminal_failure_observer is not None:
                self._terminal_failure_observer(failure)
            frame = self._terminal_frame(
                self._publication.start_frame or self._initial_chunk,
                self._latest_chunk,
                failure,
            )
            return self._publish_packet(StreamOutput((frame,)), synthetic=True) or frame

    def _publish_packet(self, packet: StreamOutput, *, synthetic: bool) -> str | None:
        draft = deepcopy(self._publication)
        output, released = draft.apply(
            packet,
            synthetic=synthetic,
            content_released=self.state.content_released,
            transition_pending=self.state.transition is not None,
        )
        self._publication = draft
        if released:
            self.state.release_content(synthetic=synthetic)
        return output

    async def _iterate(self) -> AsyncGenerator[tuple[StreamOutput, bool]]:
        while True:
            try:
                packet = await anext(self._body)
            except StopAsyncIteration:
                for frame in self._synchronize():
                    yield StreamOutput((frame,)), True
                for frame in self._publication.handoff_frames():
                    yield StreamOutput((frame,)), True
                return
            for frame in self._synchronize():
                yield StreamOutput((frame,)), True
            yield packet, False

    def _synchronize(self) -> Iterator[str]:
        transition = self.state.transition
        if self._revision == self.state.attempt_revision and transition is None:
            return iter(())
        self._revision = self.state.attempt_revision
        self.state.transition = None
        return self._publication.reset_attempt(
            transition, content_released=self.state.content_released
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = self._done = True
        await self._events.aclose()
        self.state.bind_prefix(None)
        self._publication = PacketPublication(
            self._wire_api, hide_reasoning=self._hide_reasoning
        )
        self._initial_chunk = self._latest_chunk = ""
        error = await try_close_async_iterator(self._body)
        if error is not None:
            raise error
