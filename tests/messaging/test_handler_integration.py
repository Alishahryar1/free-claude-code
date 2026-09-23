import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.trees import MessageState
from free_claude_code.messaging.workflow import MessagingWorkflow

_SCOPE = MessageScope(platform="telegram", chat_id="chat_1")


@pytest.fixture
def handler_integration(mock_platform, mock_cli_manager, mock_session_store):
    return MessagingWorkflow(mock_platform, mock_cli_manager, mock_session_store)


async def _events(events):
    for event in events:
        yield event


async def _wait_for_idle(handler: MessagingWorkflow) -> None:
    for _ in range(100):
        if handler.tree_queue.task_count() == 0:
            return
        await asyncio.sleep(0)
    raise AssertionError("messaging claims did not finish")


@pytest.mark.asyncio
async def test_full_conversation_flow_single_user(
    handler_integration,
    mock_platform,
    mock_cli_manager,
    incoming_message_factory,
) -> None:
    mock_platform.queue_send_message = AsyncMock(side_effect=["s1", "s2"])
    root_session = MagicMock()
    root_session.start_task.return_value = _events(
        [
            {"type": "session_info", "session_id": "sess1"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Reply 1"}]},
            },
            {"type": "exit", "code": 0, "stderr": None},
        ]
    )
    reply_session = MagicMock()
    reply_session.start_task.return_value = _events(
        [
            {"type": "session_info", "session_id": "sess2"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Reply 2"}]},
            },
            {"type": "exit", "code": 0, "stderr": None},
        ]
    )
    mock_cli_manager.get_or_create_session.side_effect = [
        (root_session, "pending_1", True),
        (reply_session, "pending_2", True),
    ]

    await handler_integration.handle_message(
        incoming_message_factory(text="message 1", message_id="m1")
    )
    await _wait_for_idle(handler_integration)
    root = await handler_integration.tree_queue.get_node(_SCOPE, "m1")
    assert root is not None
    assert root.state is MessageState.COMPLETED
    assert root.session_id == "sess1"

    await handler_integration.handle_message(
        incoming_message_factory(
            text="message 2",
            message_id="m2",
            reply_to_message_id="m1",
        )
    )
    await _wait_for_idle(handler_integration)
    reply = await handler_integration.tree_queue.get_node(_SCOPE, "m2")
    assert reply is not None
    assert reply.state is MessageState.COMPLETED
    assert reply.parent_id == "m1"
    mock_cli_manager.get_or_create_session.assert_called_with(session_id="sess1")
    reply_session.start_task.assert_called_with(
        "message 2", session_id="sess1", fork_session=True
    )


@pytest.mark.asyncio
async def test_error_propagation_chain(
    handler_integration,
    mock_platform,
    mock_cli_manager,
    incoming_message_factory,
) -> None:
    started = asyncio.Event()
    release_error = asyncio.Event()

    async def failing_events():
        started.set()
        await release_error.wait()
        yield {"type": "error", "error": {"message": "failed"}}

    session = MagicMock()
    session.start_task.return_value = failing_events()
    mock_cli_manager.get_or_create_session.return_value = (session, "sess1", False)
    mock_platform.queue_send_message = AsyncMock(side_effect=["s1", "s2"])

    await handler_integration.handle_message(
        incoming_message_factory(text="m1", message_id="m1")
    )
    await started.wait()
    await handler_integration.handle_message(
        incoming_message_factory(text="m2", message_id="m2", reply_to_message_id="m1")
    )
    release_error.set()
    await _wait_for_idle(handler_integration)

    root = await handler_integration.tree_queue.get_node(_SCOPE, "m1")
    child = await handler_integration.tree_queue.get_node(_SCOPE, "m2")
    assert root is not None and root.state is MessageState.ERROR
    assert child is not None and child.state is MessageState.ERROR
    rendered = "\n".join(
        call.args[2] for call in mock_platform.queue_edit_message.call_args_list
    )
    assert "Parent task failed" in rendered


@pytest.mark.asyncio
async def test_different_trees_process_independently(
    handler_integration,
    mock_platform,
    mock_cli_manager,
    incoming_message_factory,
) -> None:
    session_one = MagicMock()
    session_one.start_task.return_value = _events([{"type": "exit", "code": 0}])
    session_two = MagicMock()
    session_two.start_task.return_value = _events([{"type": "exit", "code": 0}])
    mock_cli_manager.get_or_create_session.side_effect = [
        (session_one, "s1", False),
        (session_two, "s2", False),
    ]
    mock_platform.queue_send_message = AsyncMock(side_effect=["status-t1", "status-t2"])

    await asyncio.gather(
        handler_integration.handle_message(
            incoming_message_factory(text="t1", message_id="t1")
        ),
        handler_integration.handle_message(
            incoming_message_factory(text="t2", message_id="t2")
        ),
    )
    await _wait_for_idle(handler_integration)

    node_one = await handler_integration.tree_queue.get_node(_SCOPE, "t1")
    node_two = await handler_integration.tree_queue.get_node(_SCOPE, "t2")
    assert node_one is not None and node_one.state is MessageState.COMPLETED
    assert node_two is not None and node_two.state is MessageState.COMPLETED
