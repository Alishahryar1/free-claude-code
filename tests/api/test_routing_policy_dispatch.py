import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.providers.base import BaseProvider
from tests.api.support import create_test_app


def test_auto_alias_inspection_happens_before_native_contract_selection():
    primary = MagicMock(spec=BaseProvider)
    selected = MagicMock(spec=BaseProvider)

    async def emit(req, **kwargs):
        ledger = AnthropicStreamLedger(None, req.model, 1)
        yield ledger.message_start()
        for event in ledger.ensure_text_block():
            yield event
        yield ledger.emit_text_delta("safe selection")
        for event in ledger.close_all_blocks():
            yield event
        yield ledger.message_delta("end_turn", 2)
        yield ledger.message_stop()

    selected.stream_messages.side_effect = emit
    settings = Settings.model_validate(
        {
            "MODEL": "anthropic/never",
            "ROUTING_MODE": "auto",
            "ROUTING_CANDIDATES": json.dumps(
                [
                    {
                        "model": "ollama/local",
                        "context_tokens": 32000,
                        "supports_tools": True,
                        "supports_thinking": True,
                    }
                ]
            ),
        }
    )
    app = create_test_app(
        settings, providers={"anthropic": primary, "ollama": selected}
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert response.status_code == 200
    assert "safe selection" in response.text
    primary.stream_native_messages.assert_not_called()
    selected.stream_messages.assert_called_once()
