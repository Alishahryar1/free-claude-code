from pathlib import Path

import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.server import start_server

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("auth")]


def test_bearer_auth_is_the_only_supported_header_shape(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    token = "fcc-smoke-token"
    home = tmp_path / "home"
    env_file = home / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\n"
        "PROXY_AUTH_ENABLED=true\n"
        f'ANTHROPIC_AUTH_TOKEN="{token}"\n',
        encoding="utf-8",
    )

    with start_server(
        smoke_config,
        env_overrides={
            "HOME": str(home),
            "USERPROFILE": str(home),
            "MESSAGING_PLATFORM": "none",
        },
        env_unset={"ANTHROPIC_AUTH_TOKEN", "PROXY_AUTH_ENABLED", "FCC_ENV_FILE"},
        name="auth",
    ) as server:
        assert httpx.get(f"{server.base_url}/").status_code == 401
        assert (
            httpx.get(f"{server.base_url}/", headers={"x-api-key": "wrong"}).status_code
            == 401
        )
        assert (
            httpx.get(f"{server.base_url}/", headers={"x-api-key": token}).status_code
            == 401
        )
        assert (
            httpx.get(
                f"{server.base_url}/",
                headers={"authorization": f"Bearer {token}"},
            ).status_code
            == 200
        )
        assert (
            httpx.get(
                f"{server.base_url}/",
                headers={"anthropic-auth-token": token},
            ).status_code
            == 401
        )
        assert (
            httpx.get(
                f"{server.base_url}/",
                headers={
                    "authorization": f"Bearer {token}",
                    "x-api-key": "stale-provider-key",
                },
            ).status_code
            == 200
        )
