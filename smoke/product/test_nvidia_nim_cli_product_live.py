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

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("nvidia_nim_cli")]


def test_nvidia_nim_cli_matrix_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")

    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")

    provider_models = smoke_config.nvidia_nim_cli_models()
    if not provider_models:
        pytest.skip("missing_env: no NVIDIA NIM CLI smoke models configured")

    outcomes: list[CliMatrixOutcome] = []
    for provider_model in provider_models:
        with SmokeServerDriver(
            smoke_config,
            name=f"product-nvidia-nim-cli-{_slug(provider_model.model_name)}",
            env_overrides={
                "MODEL": provider_model.full_model,
                "MESSAGING_PLATFORM": "none",
                "REASONING_POLICY": "high",
                "LOG_LEVEL": "DEBUG",
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
                    marker_prefix="NIM",
                )
            )

    report_path = write_matrix_report(
        smoke_config,
        outcomes,
        target="nvidia_nim_cli",
        filename_prefix="nvidia-nim-cli",
    )
    failures = regression_failures(outcomes)
    assert not failures, (
        f"NVIDIA NIM CLI matrix regressions written to {report_path}:\n"
        + "\n".join(failures)
    )


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")
