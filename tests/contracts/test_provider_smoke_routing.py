"""Provider smoke aliases cannot escape into user tiers or fallbacks."""

from contextlib import nullcontext

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.loader import clear_settings_cache, get_settings
from free_claude_code.config.reasoning import ReasoningPreference
from smoke.lib.config import ProviderModel, SmokeConfig
from smoke.lib.e2e import SmokeServerDriver
from smoke.product.test_provider_product_live import _server_for_provider


def test_provider_smoke_isolates_tiers_and_reasoning(monkeypatch):
    for tier in ("FABLE", "OPUS", "SONNET", "HAIKU"):
        monkeypatch.setenv(f"MODEL_{tier}", "groq/wrong-model")
        monkeypatch.setenv(f"REASONING_{tier}", "off")
    monkeypatch.setenv("MODEL_FALLBACKS", "deepseek/wrong-fallback")
    monkeypatch.setenv("REASONING_POLICY", "off")
    captured = {}

    def run(driver):
        captured.update(driver.env_overrides)
        return nullcontext()

    monkeypatch.setattr(SmokeServerDriver, "run", run)
    model = ProviderModel("nvidia_nim", "nvidia_nim/smoke-selected", "test")
    with _server_for_provider(SmokeConfig.load(), model, "routing"):
        pass
    with monkeypatch.context() as child:
        for key, value in captured.items():
            child.setenv(key, value)
        clear_settings_cache()
        router = ModelRouter(get_settings())
        for alias in (
            "fcc-smoke-default",
            "claude-fable-5",
            "claude-opus-4-7",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5",
        ):
            route = router.resolve(alias)
            assert route.primary.provider_model_ref == model.full_model
            assert not route.fallbacks
            assert route.reasoning_preference == ReasoningPreference.CLIENT
    clear_settings_cache()
