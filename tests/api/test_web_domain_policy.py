"""Scoped web domain constraints are enforced before DNS on every fetch hop."""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.api.web_tools.egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    domain_matches,
    get_validated_stream_addrinfos_for_egress,
    normalize_web_domain,
    web_url_allowed_by_domains,
)
from free_claude_code.api.web_tools.outbound import _run_web_fetch

_HTTP_SCHEMES = frozenset({"http", "https"})
_PUBLIC_ADDRINFOS = [
    (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))
]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Example.COM", "example.com"),
        ("Example.COM.", "example.com"),
        ("bücher.example", "xn--bcher-kva.example"),
        ("BÜCHER.example.", "xn--bcher-kva.example"),
        ("faß.example", "xn--fa-hia.example"),
        ("example。com。", "example.com"),
    ],
)
def test_normalize_web_domain(raw: str, expected: str) -> None:
    assert normalize_web_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        ".",
        "https://example.com",
        "//example.com",
        "example.com:443",
        "example.com/path",
        "example.com?query",
        "example.com#fragment",
        "*.example.com",
        "user@example.com",
        "user:secret@example.com",
        "example.com\\evil",
        "example%2ecom",
        "[::1]",
        "-example.com",
        "example-.com",
        "under_score.example",
        "example..com",
        "example.com..",
        " example.com",
        "example.com ",
        "exam ple.com",
        "exam\tple.com",
        "exam\nple.com",
        "example.com\x00",
        "​example.com",
        "xn--invalid-.example",
        f"{'a' * 64}.example",
        ".".join(["a" * 63] * 5),
    ],
)
def test_normalize_web_domain_rejects_malformed_options(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_web_domain(raw)


@pytest.mark.parametrize(
    "host,domain,expected",
    [
        ("example.com", "example.com", True),
        ("docs.example.com", "example.com", True),
        ("deep.docs.example.com", "example.com", True),
        ("EXAMPLE.COM.", "example.com", True),
        ("docs.example.com", "EXAMPLE.COM.", True),
        ("bücher.example.", "xn--bcher-kva.example", True),
        ("docs.xn--bcher-kva.example", "BÜCHER.example.", True),
        ("evil-example.com", "example.com", False),
        ("notexample.com", "example.com", False),
        ("example.com.attacker.test", "example.com", False),
        ("example.com", "docs.example.com", False),
        ("fass.example", "faß.example", False),
        ("8.8.8.8", "8.8.8.8", True),
        ("8.8.8.8", "8.8", False),
        ("docs.8.8.8.8", "8.8.8.8", False),
        ("https://example.com", "example.com", False),
        ("example.com", "*.example.com", False),
        ("example.com", "", False),
        ("", "example.com", False),
    ],
)
def test_domain_matches_requires_hostname_boundary(
    host: str, domain: str, expected: bool
) -> None:
    assert domain_matches(host, domain) is expected


@pytest.mark.parametrize(
    "url,allowed,blocked,expected",
    [
        ("https://example.com/path", (), (), True),
        ("http://docs.example.com/path", ("example.com",), (), True),
        ("https://EXAMPLE.COM./path", ("example.com",), (), True),
        ("https://bücher.example/path", ("xn--bcher-kva.example",), (), True),
        ("https://xn--bcher-kva.example/path", ("BÜCHER.example.",), (), True),
        ("https://example.com:8443/path", ("example.com",), (), True),
        ("https://evil-example.com/path", ("example.com",), (), False),
        ("https://example.com.evil.test/path", ("example.com",), (), False),
        ("https://other.test/path", ("example.com",), (), False),
        ("https://example.com/path", (), ("example.com",), False),
        ("https://docs.example.com/path", (), ("example.com",), False),
        ("https://notexample.com/path", (), ("example.com",), True),
        ("https://example.com/path", ("example.com",), ("example.com",), False),
        (
            "https://docs.example.com/path",
            ("example.com",),
            ("docs.example.com",),
            False,
        ),
        ("https://example.com/path", ("example.com",), ("docs.example.com",), True),
        ("https://bücher.example/path", (), ("XN--BCHER-KVA.example.",), False),
        ("https://other.test/path", ("*.test",), (), False),
        ("https://other.test/path", (), ("*.test",), False),
        ("https://example.com/path", ("example.com", "*.test"), (), False),
    ],
)
def test_web_url_allowed_by_domains(
    url: str,
    allowed: tuple[str, ...],
    blocked: tuple[str, ...],
    expected: bool,
) -> None:
    with patch("free_claude_code.api.web_tools.egress.socket.getaddrinfo") as dns:
        assert web_url_allowed_by_domains(url, allowed, blocked) is expected
    dns.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/path",
        "//example.com/path",
        "https:///path",
        "https://user@example.com/path",
        "https://user:secret@example.com/path",
        "https://@example.com/path",
        "https://example.com:invalid/path",
        "https://example.com:65536/path",
        "https://example.com:/path",
        "https://[broken/path",
        "https://example..com/path",
        "https://example.com../path",
        "https://example.com\\evil.test/path",
        "https://exam\tple.com/path",
        "\nhttps://example.com/path",
        "https://example.com\x00/path",
    ],
)
def test_web_url_allowed_by_domains_rejects_unsafe_or_invalid_urls(url: str) -> None:
    assert not web_url_allowed_by_domains(url, (), ())


def test_egress_policy_keeps_existing_two_positional_arguments() -> None:
    policy = WebFetchEgressPolicy(False, _HTTP_SCHEMES)
    assert policy.allow_private_network_targets is False
    assert policy.allowed_schemes == _HTTP_SCHEMES
    assert policy.allowed_domains == ()
    assert policy.blocked_domains == ()


def test_egress_policy_normalizes_domain_options() -> None:
    policy = WebFetchEgressPolicy(
        False, _HTTP_SCHEMES, ("EXAMPLE.COM.",), ("BÜCHER.example.",)
    )
    assert policy.allowed_domains == ("example.com",)
    assert policy.blocked_domains == ("xn--bcher-kva.example",)


@pytest.mark.parametrize("option", ["allowed_domains", "blocked_domains"])
def test_egress_policy_rejects_invalid_domains(option: str) -> None:
    with pytest.raises(ValueError):
        WebFetchEgressPolicy(False, _HTTP_SCHEMES, **{option: ("*.example.com",)})


@pytest.mark.parametrize("allow_private", [False, True])
@pytest.mark.parametrize(
    "url,allowed,blocked",
    [
        ("https://other.test/", ("example.com",), ()),
        ("https://evil-example.com/", ("example.com",), ()),
        ("https://example.com.evil.test/", ("example.com",), ()),
        ("https://docs.example.com/", (), ("example.com",)),
        ("https://example.com/", ("example.com",), ("example.com",)),
        ("https://127.0.0.1/", ("example.com",), ()),
    ],
)
def test_egress_domain_rejection_never_resolves_dns(
    allow_private: bool,
    url: str,
    allowed: tuple[str, ...],
    blocked: tuple[str, ...],
) -> None:
    policy = WebFetchEgressPolicy(allow_private, _HTTP_SCHEMES, allowed, blocked)
    with (
        patch("free_claude_code.api.web_tools.egress.socket.getaddrinfo") as dns,
        pytest.raises(WebFetchEgressViolation, match="domain"),
    ):
        get_validated_stream_addrinfos_for_egress(url, policy)
    dns.assert_not_called()


@pytest.mark.parametrize("allow_private", [False, True])
@pytest.mark.parametrize(
    "url,allowed,blocked,host",
    [
        ("https://example.com/", (), (), "example.com"),
        ("https://docs.example.com/", ("example.com",), (), "docs.example.com"),
        ("https://EXAMPLE.COM./", ("example.com",), (), "example.com"),
        (
            "https://bücher.example/",
            ("xn--bcher-kva.example",),
            (),
            "xn--bcher-kva.example",
        ),
        ("https://example.com/", (), ("other.test",), "example.com"),
    ],
)
def test_egress_resolves_allowed_normalized_domains(
    allow_private: bool,
    url: str,
    allowed: tuple[str, ...],
    blocked: tuple[str, ...],
    host: str,
) -> None:
    policy = WebFetchEgressPolicy(allow_private, _HTTP_SCHEMES, allowed, blocked)
    with patch(
        "free_claude_code.api.web_tools.egress.socket.getaddrinfo",
        return_value=_PUBLIC_ADDRINFOS,
    ) as dns:
        assert (
            get_validated_stream_addrinfos_for_egress(url, policy) == _PUBLIC_ADDRINFOS
        )
    dns.assert_called_once_with(
        host, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )


def test_egress_rechecks_domains_for_each_redirect_hop() -> None:
    policy = WebFetchEgressPolicy(True, _HTTP_SCHEMES, ("example.com",))
    with patch(
        "free_claude_code.api.web_tools.egress.socket.getaddrinfo",
        return_value=_PUBLIC_ADDRINFOS,
    ) as dns:
        get_validated_stream_addrinfos_for_egress("https://example.com/start", policy)
        get_validated_stream_addrinfos_for_egress(
            "https://docs.example.com/next", policy
        )
        with pytest.raises(WebFetchEgressViolation, match="domain"):
            get_validated_stream_addrinfos_for_egress("https://evil.test/final", policy)
    assert dns.call_count == 2


@pytest.mark.parametrize("allow_private", [False, True])
@pytest.mark.parametrize(
    "url", ["https://user:secret@example.com/", "https://example.com:invalid/"]
)
def test_egress_rejects_unsafe_urls_before_dns(allow_private: bool, url: str) -> None:
    policy = WebFetchEgressPolicy(allow_private, _HTTP_SCHEMES)
    with (
        patch("free_claude_code.api.web_tools.egress.socket.getaddrinfo") as dns,
        pytest.raises(WebFetchEgressViolation),
    ):
        get_validated_stream_addrinfos_for_egress(url, policy)
    dns.assert_not_called()


def test_egress_allowed_domain_still_checks_resolved_addresses() -> None:
    private_infos = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.1", 443))
    ]
    policy = WebFetchEgressPolicy(False, _HTTP_SCHEMES, ("example.com",))
    with (
        patch(
            "free_claude_code.api.web_tools.egress.socket.getaddrinfo",
            return_value=private_infos,
        ) as dns,
        pytest.raises(WebFetchEgressViolation, match="non-public"),
    ):
        get_validated_stream_addrinfos_for_egress("https://example.com/", policy)
    dns.assert_called_once()


def test_egress_private_opt_in_preserved_with_no_domain_constraints() -> None:
    policy = WebFetchEgressPolicy(True, _HTTP_SCHEMES)
    dns = MagicMock(return_value=_PUBLIC_ADDRINFOS)
    with patch("free_claude_code.api.web_tools.egress.socket.getaddrinfo", dns):
        get_validated_stream_addrinfos_for_egress("http://127.0.0.1/", policy)
        get_validated_stream_addrinfos_for_egress("http://[::1]/", policy)
    assert [call.args for call in dns.call_args_list] == [
        ("127.0.0.1", 80),
        ("::1", 80),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_private", [False, True])
async def test_fetch_redirect_to_disallowed_domain_never_resolves_or_connects(
    allow_private: bool,
) -> None:
    response = MagicMock()
    response.status = 302
    response.url = "https://example.com/start"
    response.headers = {"location": "https://evil.test/final"}

    async def empty_body(_chunk_size: int):
        yield b""

    response.content.iter_chunked = MagicMock(side_effect=empty_body)
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get = MagicMock(return_value=response_context)
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    policy = WebFetchEgressPolicy(allow_private, _HTTP_SCHEMES, ("example.com",))
    with (
        patch(
            "free_claude_code.api.web_tools.egress.socket.getaddrinfo",
            return_value=_PUBLIC_ADDRINFOS,
        ) as dns,
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=session_context,
        ),
        pytest.raises(WebFetchEgressViolation, match="domain"),
    ):
        await _run_web_fetch("https://example.com/start", policy)
    dns.assert_called_once_with(
        "example.com", 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    session.get.assert_called_once_with(
        "https://example.com/start", allow_redirects=False
    )
