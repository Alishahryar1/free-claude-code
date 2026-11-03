import json

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


def test_history_is_bounded_redacted_and_returns_detached_rows():
    from free_claude_code.application.routing_decisions import DecisionLog

    log = DecisionLog(limit=2)
    for index in range(3):
        log.record(
            request_id=f"r-{index}",
            mode="shadow",
            outcome="ok",
            selected_model="ollama/test",
            prompt="private",
            api_key="secret",
        )
    rows = log.recent()
    assert len(rows) == 2 and rows[0]["request_id"] == "r-2"
    assert "private" not in json.dumps(rows) and "secret" not in json.dumps(rows)
    rows[0]["outcome"] = "changed"
    assert log.recent()[0]["outcome"] == "ok"


@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.asyncio
async def test_decisions_extract_numeric_usage_without_retaining_content(wire):
    from free_claude_code.application.routing_decisions import (
        DecisionLog,
        observe_stream,
    )

    async def source():
        if wire == "messages":
            yield 'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"private"}}\n\n'
            yield 'event: message_delta\ndata: {"usage":{"output_tokens":7}}\n\n'
        else:
            yield 'event: response.output_text.delta\ndata: {"delta":"private"}\n\n'
            yield 'event: response.completed\ndata: {"response":{"usage":{"output_tokens":7}}}\n\n'

    log = DecisionLog()
    _ = [
        chunk
        async for chunk in observe_stream(
            source(), {"mode": "shadow", "request_id": "r"}, log
        )
    ]
    row = log.recent()[0]
    usage = row["reported_usage"]
    assert isinstance(usage, dict)
    assert usage["output_tokens"] == 7
    assert row["first_output_ms"] is not None
    assert row["usage_kind"] == "normalized_stream_unverified"
    assert "private" not in json.dumps(row)


def test_current_admin_exposes_policy_fields_and_local_only_history():
    app = create_test_app(Settings())
    local = TestClient(app, base_url="http://localhost", client=("::1", 12345))
    remote = TestClient(
        app, base_url="http://localhost", client=("remote.example", 12345)
    )
    with local:
        response = local.get("/admin/api/config")
        keys = {field["key"] for field in response.json()["fields"]}
        assert "ROUTING_MODE" in keys
        assert local.get("/admin/api/routing/decisions").status_code == 200
    assert remote.get("/admin/api/routing/decisions").status_code == 403


def test_current_admin_renders_policy_section_and_decision_history():
    from pathlib import Path

    root = Path("src/free_claude_code/api/admin_static")
    script = (root / "admin.js").read_text()
    assert '"routing_policy"' in script
    assert "/admin/api/routing/decisions" in script
    assert 'id="routingDecisions"' in (root / "index.html").read_text()
