"""The updater checks published versions before authorizing installation."""

from importlib.metadata import PackageNotFoundError

import httpx
import pytest

from free_claude_code.updater import check as update_check


def release(version="1.10.0", *, yanked=False):
    return {"info": {"version": version}, "urls": [{"yanked": yanked}]}


@pytest.fixture
def check(monkeypatch):
    original_client = httpx.Client

    def run(installed="1.9.0", payload=None, status=200, error=None, body=None):
        monkeypatch.setattr(update_check, "version", lambda name: installed)

        def respond(request):
            assert str(request.url) == update_check.PYPI_URL
            if error is not None:
                raise error
            if body is not None:
                return httpx.Response(status, content=body)
            return httpx.Response(status, json=payload)

        monkeypatch.setattr(
            update_check.httpx,
            "Client",
            lambda **kwargs: original_client(
                transport=httpx.MockTransport(respond), **kwargs
            ),
        )
        return update_check.main()

    return run


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("1.9.0", 10),
        ("1.10.0", 0),
        ("1.11.0", 0),
        ("1.10.0+local", 0),
        ("1.10.0.dev1", 10),
    ],
)
def test_version_comparison(check, capsys, installed, expected):
    assert check(installed, release()) == expected
    output = capsys.readouterr()
    assert installed in output.out
    assert not output.err


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"info": []},
        release("invalid"),
        release("2.0rc1"),
        release("2.0.dev1"),
        release(yanked=True),
        {"info": {"version": "2.0"}, "urls": []},
        {"info": {"version": "2.0"}, "urls": [{}]},
    ],
)
def test_invalid_release_does_not_authorize_installation(check, capsys, payload):
    assert check(payload=payload) == 1
    assert capsys.readouterr().err


@pytest.mark.parametrize("installed", ["invalid", "0+unknown"])
def test_unknown_installed_version(check, installed):
    assert check(installed, release()) == 1


def test_missing_distribution(monkeypatch, capsys):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(update_check, "version", missing)
    assert update_check.main() == 1
    assert capsys.readouterr().err


def test_http_error(check, capsys):
    assert check(payload=release(), status=503) == 1
    assert capsys.readouterr().err


def test_timeout(check, capsys):
    assert check(error=httpx.ReadTimeout("timed out")) == 1
    assert capsys.readouterr().err


def test_malformed_json(check, capsys):
    assert check(body="not JSON") == 1
    assert capsys.readouterr().err
