"""Installed Aider launcher using native model settings for FCC routing."""

from collections.abc import Sequence

from free_claude_code.harnesses.environment import client_environment
from free_claude_code.harnesses.launch import PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .aider_config import AIDER_API_KEY_ENV_PREFIX, build_aider_config
from .runner import HarnessSpec, LaunchContext, launch_harness


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    key_env = f"{AIDER_API_KEY_ENV_PREFIX}{ctx.launch_id.upper()}"
    config = build_aider_config(
        catalog.models,
        messages_url=f"{ctx.proxy_root_url.rstrip('/')}/v1/messages",
        api_key_env=key_env,
        launch_id=ctx.launch_id,
    )
    settings_path = files.write_json("model-settings.yml", config.settings)
    metadata_path = files.write_json("model-metadata.json", config.metadata)
    return PreparedLaunch(
        [ctx.binary_path, "--set-env", "ANTHROPIC_API_KEY=fcc-local", *args],
        client_environment(
            ctx.base_env,
            proxy_root_url=ctx.proxy_root_url,
            remove_prefixes=(AIDER_API_KEY_ENV_PREFIX,),
            case_sensitive=False,
            updates={
                key_env: ctx.auth_token,
                "AIDER_MODEL": catalog.default_model_id,
                "AIDER_MODEL_SETTINGS_FILE": str(settings_path),
                "AIDER_MODEL_METADATA_FILE": str(metadata_path),
            },
        ),
    )


SPEC = HarnessSpec(
    binary_name="aider",
    display_name="Aider",
    install_hint="Install Aider from: https://aider.chat/docs/install.html",
    configure=_configure,
    catalog_view="messages",
)


def launch(argv: Sequence[str] | None = None) -> None:
    launch_harness(SPEC, argv)
