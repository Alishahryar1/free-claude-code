import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("auth")]


def test_api_bearer_auth_contract_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    token = "product-smoke-token"
    home = tmp_path / "home"
    env_file = home / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\n"
        "PROXY_AUTH_ENABLED=true\n"
        f'ANTHROPIC_AUTH_TOKEN="{token}"\n',
        encoding="utf-8",
    )
    with SmokeServerDriver(
        smoke_config,
        name="product-auth",
        env_overrides={
            "HOME": str(home),
            "USERPROFILE": str(home),
            "MESSAGING_PLATFORM": "none",
        },
        env_unset={"ANTHROPIC_AUTH_TOKEN", "PROXY_AUTH_ENABLED", "FCC_ENV_FILE"},
    ).run() as server:
        unauth = httpx.get(
            f"{server.base_url}/v1/models", timeout=smoke_config.timeout_s
        )
        x_api_key_only = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"x-api-key": token},
            timeout=smoke_config.timeout_s,
        )
        bearer = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"authorization": f"Bearer {token}"},
            timeout=smoke_config.timeout_s,
        )
        anthropic_auth_token_only = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"anthropic-auth-token": token},
            timeout=smoke_config.timeout_s,
        )
        bearer_with_stale_api_key = httpx.get(
            f"{server.base_url}/v1/models",
            headers={
                "authorization": f"Bearer {token}",
                "x-api-key": "stale-provider-key",
            },
            timeout=smoke_config.timeout_s,
        )
        invalid_bearer_with_matching_api_key = httpx.get(
            f"{server.base_url}/v1/models",
            headers={
                "authorization": "Bearer wrong",
                "x-api-key": token,
            },
            timeout=smoke_config.timeout_s,
        )

    assert unauth.status_code == 401
    assert x_api_key_only.status_code == 401
    assert bearer.status_code == 200
    assert anthropic_auth_token_only.status_code == 401
    assert bearer_with_stale_api_key.status_code == 200
    assert invalid_bearer_with_matching_api_key.status_code == 401
