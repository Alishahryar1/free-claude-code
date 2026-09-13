from free_claude_code.harnesses.environment import with_local_proxy_bypass


def test_child_proxy_bypass_preserves_existing_proxy_policy() -> None:
    base_env = {
        "HTTP_PROXY": "http://proxy.example:3128",
        "NO_PROXY": "example.com, localhost",
        "no_proxy": "10.0.0.0/8,EXAMPLE.COM",
        "KEEP_ME": "yes",
    }

    env = with_local_proxy_bypass(
        base_env,
        proxy_root_url="http://fcc.internal:8082",
    )

    assert env["HTTP_PROXY"] == "http://proxy.example:3128"
    assert env["KEEP_ME"] == "yes"
    assert env["NO_PROXY"] == (
        "example.com,localhost,10.0.0.0/8,127.0.0.1,::1,fcc.internal"
    )
    assert env["no_proxy"] == env["NO_PROXY"]
    assert base_env["NO_PROXY"] == "example.com, localhost"
    assert base_env["no_proxy"] == "10.0.0.0/8,EXAMPLE.COM"


def test_child_proxy_bypass_uses_all_loopback_spellings_without_duplicates() -> None:
    env = with_local_proxy_bypass(
        {"NO_PROXY": "127.0.0.1"},
        proxy_root_url="http://127.0.0.1:8082",
    )

    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]
