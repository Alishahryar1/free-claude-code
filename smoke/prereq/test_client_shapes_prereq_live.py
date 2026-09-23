import pytest

from smoke.lib.config import SmokeConfig, auth_headers
from smoke.lib.http import message_payload, post_json
from smoke.lib.server import RunningServer

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("clients")]


def test_vscode_and_jetbrains_shaped_requests(
    smoke_server: RunningServer,
    smoke_config: SmokeConfig,
) -> None:
    payload = message_payload("quota", max_tokens=1)

    vscode_headers = auth_headers()
    vscode_headers.update(
        {
            "anthropic-beta": "messages-2023-12-15",
            "user-agent": "Claude-Code-VSCode smoke",
        }
    )
    vscode = post_json(
        smoke_server,
        "/v1/messages?beta=true",
        payload,
        smoke_config,
        headers=vscode_headers,
    )
    assert vscode.status_code == 200, vscode.text
    assert vscode.json()["content"][0]["text"] == "Quota check passed."

    jetbrains_headers = auth_headers()
    jetbrains_headers["user-agent"] = "JetBrains-ACP smoke"
    jetbrains = post_json(
        smoke_server,
        "/v1/messages",
        payload,
        smoke_config,
        headers=jetbrains_headers,
    )
    assert jetbrains.status_code == 200, jetbrains.text
    assert jetbrains.json()["content"][0]["text"] == "Quota check passed."
