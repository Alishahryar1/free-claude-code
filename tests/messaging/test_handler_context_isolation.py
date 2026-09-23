import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.messaging.workflow import MessagingWorkflow


async def _wait_for_idle(handler: MessagingWorkflow) -> None:
    for _ in range(100):
        if handler.tree_queue.task_count() == 0:
            return
        await asyncio.sleep(0)
    raise AssertionError("messaging claims did not finish")


def _session_factory(calls: list[tuple[str, str | None, bool]]):
    async def get_or_create_session(session_id=None):
        session = MagicMock()

        async def start_task(prompt, session_id=None, fork_session=False):
            calls.append((prompt, session_id, fork_session))
            yield {"type": "session_info", "session_id": f"sess_{prompt}"}
            yield {"type": "exit", "code": 0, "stderr": None}

        session.start_task = start_task
        return session, f"pending_{len(calls)}", True

    return get_or_create_session


@pytest.mark.asyncio
async def test_sibling_replies_fork_from_parent_session_id(
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
) -> None:
    calls: list[tuple[str, str | None, bool]] = []
    mock_cli_manager.get_or_create_session = AsyncMock(
        side_effect=_session_factory(calls)
    )
    mock_platform.queue_send_message = AsyncMock(
        side_effect=["status_A", "status_R1", "status_R2"]
    )
    handler = MessagingWorkflow(mock_platform, mock_cli_manager, mock_session_store)

    await handler.handle_message(incoming_message_factory(text="A", message_id="A"))
    await _wait_for_idle(handler)
    await handler.handle_message(
        incoming_message_factory(text="R1", message_id="R1", reply_to_message_id="A")
    )
    await _wait_for_idle(handler)
    await handler.handle_message(
        incoming_message_factory(text="R2", message_id="R2", reply_to_message_id="A")
    )
    await _wait_for_idle(handler)

    assert calls == [
        ("A", None, False),
        ("R1", "sess_A", True),
        ("R2", "sess_A", True),
    ]


@pytest.mark.asyncio
async def test_grandchild_reply_forks_from_branch_session(
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
) -> None:
    calls: list[tuple[str, str | None, bool]] = []
    mock_cli_manager.get_or_create_session = AsyncMock(
        side_effect=_session_factory(calls)
    )
    mock_platform.queue_send_message = AsyncMock(
        side_effect=["status_A", "status_R1", "status_C1"]
    )
    handler = MessagingWorkflow(mock_platform, mock_cli_manager, mock_session_store)

    await handler.handle_message(incoming_message_factory(text="A", message_id="A"))
    await _wait_for_idle(handler)
    await handler.handle_message(
        incoming_message_factory(text="R1", message_id="R1", reply_to_message_id="A")
    )
    await _wait_for_idle(handler)
    await handler.handle_message(
        incoming_message_factory(text="C1", message_id="C1", reply_to_message_id="R1")
    )
    await _wait_for_idle(handler)

    assert calls[-1] == ("C1", "sess_R1", True)
