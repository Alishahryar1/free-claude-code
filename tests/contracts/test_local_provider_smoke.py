from dataclasses import dataclass

import httpx
import pytest

from smoke.lib.local_providers import first_local_provider_model_id


@dataclass(slots=True)
class FakeResponse:
    status_code: int
    payload: dict
    text: str = ""

    def json(self) -> dict:
        return self.payload


def test_local_provider_openai_models_returns_first_id(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        calls.append((url, timeout))
        return FakeResponse(200, {"data": [{"id": "local-model"}]})

    monkeypatch.setattr("smoke.lib.local_providers.httpx.get", fake_get)

    model = first_local_provider_model_id(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        timeout_s=45,
    )

    assert model == "local-model"
    assert calls == [("http://127.0.0.1:1234/v1/models", 1.5)]


@pytest.mark.parametrize(
    ("provider", "port", "model_id"),
    [
        ("llamacpp", 8080, "local-model"),
        ("ollama", 11434, "llama3.1"),
    ],
)
def test_local_provider_root_url_targets_openai_v1_models(
    monkeypatch, provider: str, port: int, model_id: str
) -> None:
    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        assert url == f"http://127.0.0.1:{port}/v1/models"
        assert timeout == 1.5
        return FakeResponse(200, {"data": [{"id": model_id}]})

    monkeypatch.setattr("smoke.lib.local_providers.httpx.get", fake_get)

    assert (
        first_local_provider_model_id(
            provider,
            f"http://127.0.0.1:{port}",
            timeout_s=45,
        )
        == model_id
    )


def test_local_provider_not_running_is_missing_env_skip(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("smoke.lib.local_providers.httpx.get", fake_get)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        first_local_provider_model_id(
            "llamacpp",
            "http://127.0.0.1:8080/v1",
            timeout_s=45,
        )

    assert "missing_env: llamacpp local server is not running" in str(excinfo.value)


def test_local_provider_empty_model_list_is_missing_env_skip(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        return FakeResponse(200, {"data": []})

    monkeypatch.setattr("smoke.lib.local_providers.httpx.get", fake_get)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        first_local_provider_model_id(
            "lmstudio",
            "http://127.0.0.1:1234/v1",
            timeout_s=45,
        )

    assert "missing_env: lmstudio local server has no loaded models" in str(
        excinfo.value
    )
