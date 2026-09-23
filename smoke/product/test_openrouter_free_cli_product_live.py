import shutil
from pathlib import Path

import pytest

from smoke.lib.claude_cli_matrix import (
    CliMatrixOutcome,
    regression_failures,
    run_cli_feature_probes,
    write_matrix_report,
)
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("openrouter_free_cli")]


def test_openrouter_free_cli_matrix_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    if not smoke_config.has_provider_configuration("open_router"):
        pytest.skip("missing_env: OPENROUTER_API_KEY is not configured")

    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")

    provider_models = smoke_config.openrouter_free_cli_models()
    if not provider_models:
        pytest.skip("missing_env: no OpenRouter free CLI smoke models configured")

    outcomes: list[CliMatrixOutcome] = []
    for provider_model in provider_models:
        with SmokeServerDriver(
            smoke_config,
            name=f"product-openrouter-free-cli-{_slug(provider_model.model_name)}",
            env_overrides={
                "MODEL": provider_model.full_model,
                "MESSAGING_PLATFORM": "none",
                "REASONING_POLICY": "high",
                "LOG_RAW_API_PAYLOADS": "true",
                "LOG_RAW_SSE_EVENTS": "true",
            },
        ).run() as server:
            outcomes.extend(
                run_cli_feature_probes(
                    claude_bin=claude_bin,
                    server=server,
                    smoke_config=smoke_config,
                    provider_model=provider_model,
                    model_dir=tmp_path / _slug(provider_model.model_name),
                    marker_prefix="OPENROUTER_FREE",
                )
            )

    report_path = write_matrix_report(
        smoke_config,
        outcomes,
        target="openrouter_free_cli",
        filename_prefix="openrouter-free-cli",
    )
    failures = regression_failures(outcomes)
    assert not failures, (
        f"OpenRouter free CLI matrix regressions written to {report_path}:\n"
        + "\n".join(failures)
    )


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")
