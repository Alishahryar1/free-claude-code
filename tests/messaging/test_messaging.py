"""Tests for messaging/ module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# --- Existing Tests ---


class TestMessagingModels:
    """Test messaging models."""

    def test_incoming_message_creation(self):
        """Test IncomingMessage dataclass."""
        from free_claude_code.messaging.models import IncomingMessage

        msg = IncomingMessage(
            text="Hello",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="telegram",
        )
        assert msg.text == "Hello"
        assert msg.chat_id == "123"
        assert msg.platform == "telegram"
        assert msg.is_reply() is False

    def test_incoming_message_with_reply(self):
        """Test IncomingMessage as a reply."""
        from free_claude_code.messaging.models import IncomingMessage

        msg = IncomingMessage(
            text="Reply text",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="discord",
            reply_to_message_id="100",
        )
        assert msg.is_reply() is True
        assert msg.reply_to_message_id == "100"


class TestMessagingPorts:
    """Test explicit messaging platform component ports."""

    def test_components_bundle_runtime_and_outbound(self):
        """Verify the factory handoff shape is explicit."""
        from free_claude_code.messaging.platforms.ports import (
            MessagingPlatformComponents,
        )

        runtime = MagicMock()
        runtime.name = "telegram"
        runtime.start = AsyncMock()
        runtime.quiesce = AsyncMock()
        runtime.close = AsyncMock()
        runtime.on_message = MagicMock()
        outbound = MagicMock()
        outbound.queue_send_message = AsyncMock()
        outbound.queue_edit_message = AsyncMock()
        outbound.queue_delete_messages = AsyncMock()
        outbound.fire_and_forget = MagicMock()
        components = MessagingPlatformComponents(
            name="telegram",
            runtime=runtime,
            outbound=outbound,
            voice_cancellation=None,
        )
        assert components.runtime is runtime
        assert components.outbound is outbound


class TestTreeQueueManager:
    """Test TreeQueueManager."""

    def test_tree_queue_manager_init(self):
        from free_claude_code.messaging.trees import TreeQueueManager

        async def process(_claim):
            return None

        mgr = TreeQueueManager(process, store=AsyncMock())
        assert mgr.get_tree_count() == 0

    @pytest.mark.asyncio
    async def test_admit_creates_tree_and_claim(self):
        from free_claude_code.messaging.models import IncomingMessage
        from free_claude_code.messaging.trees import TreeQueueManager

        processed = asyncio.Event()

        async def processor(claim):
            assert claim.node.node_id == "1"
            processed.set()

        incoming = IncomingMessage(
            text="test",
            chat_id="1",
            user_id="1",
            message_id="1",
            platform="test",
        )

        mgr = TreeQueueManager(processor, store=AsyncMock())
        decision = await mgr.admit(incoming, "status_1")

        assert decision.accepted is True
        assert decision.claim is not None
        await processed.wait()

    @pytest.mark.asyncio
    async def test_cancel_unknown_node_is_empty(self):
        from free_claude_code.messaging.models import MessageScope
        from free_claude_code.messaging.trees import TreeQueueManager

        async def process(_claim):
            return None

        mgr = TreeQueueManager(process, store=AsyncMock())
        scope = MessageScope(platform="test", chat_id="1")
        cancelled = await mgr.cancel_node(scope, "nonexistent")
        assert cancelled.effects == ()
