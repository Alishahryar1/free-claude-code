import plistlib

import pytest

from free_claude_code.runtime.integrations.discovery import LocalInstallations
from tests.integration_support import executable, installed_clients, write_json


def test_discovery_reads_installed_metadata_without_running_clients(
    tmp_path, monkeypatch
):
    locator = installed_clients(tmp_path)
    monkeypatch.setattr(
        "subprocess.run", lambda *_a, **_kw: pytest.fail("must not launch a client")
    )
    snapshot = locator.scan()
    assert snapshot.claude_vscode and snapshot.codex_vscode and snapshot.codex_app
    assert snapshot.jetbrains and snapshot.acp_command
    assert snapshot.fcc_command == tmp_path / "bin/fcc-codex"


def test_stray_extension_directory_does_not_count_as_profile_installation(tmp_path):
    locator = installed_clients(tmp_path)
    (tmp_path / ".vscode/extensions/extensions.json").unlink()
    snapshot = locator.scan()
    assert not snapshot.claude_vscode and not snapshot.codex_vscode


def test_index_requires_matching_package_identity_and_existing_entrypoint(tmp_path):
    locator = installed_clients(tmp_path)
    write_json(
        tmp_path / ".vscode/extensions/anthropic.claude-code-1.0.0/package.json",
        {"publisher": "wrong", "name": "claude-code"},
    )
    (
        tmp_path
        / "lib/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
    ).unlink()
    snapshot = locator.scan()
    assert not snapshot.claude_vscode
    assert not snapshot.acp_command


@pytest.mark.parametrize(
    "platform,suffix",
    [
        ("win32", "AppData/Roaming/Code/User/settings.json"),
        ("darwin", "Library/Application Support/Code/User/settings.json"),
        ("linux", ".config/Code/User/settings.json"),
    ],
)
def test_standard_settings_paths(tmp_path, platform, suffix):
    locator = LocalInstallations(
        tmp_path,
        platform,
        {"APPDATA": str(tmp_path / "AppData/Roaming")},
        tmp_path / "bin",
        (tmp_path / "apps",),
    )
    assert locator.vscode_settings == tmp_path / suffix


def test_macos_app_uses_bundle_identity_and_declared_executable(tmp_path):
    bundle = tmp_path / "apps/ChatGPT.app/Contents"
    executable(bundle / "MacOS/ChatGPT")
    (bundle / "Info.plist").write_bytes(
        plistlib.dumps(
            {"CFBundleIdentifier": "com.openai.codex", "CFBundleExecutable": "ChatGPT"}
        )
    )
    locator = LocalInstallations(
        tmp_path, "darwin", {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert locator.scan().codex_app
    (bundle / "MacOS/ChatGPT").unlink()
    assert not locator.scan().codex_app


def test_windows_app_uses_package_declared_executable(tmp_path, monkeypatch):
    package = tmp_path / "WindowsApps/OpenAI.Codex"
    executable(package / "app/ChatGPT.exe")
    (package / "AppxManifest.xml").write_text(
        '<Package><Identity Name="OpenAI.Codex"/><Applications><Application Executable="app/ChatGPT.exe"/></Applications></Package>'
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.integrations.discovery.windows_codex_locations",
        lambda: (package,),
    )
    locator = LocalInstallations(
        tmp_path, "win32", {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert locator.scan().codex_app


def test_wsl_scope_is_reported_without_configuring_another_environment(tmp_path):
    locator = installed_clients(tmp_path)
    locator.environ["WSL_DISTRO_NAME"] = "Ubuntu"
    assert "WSL" in locator.scan().scope_issue


def test_discovery_does_not_create_missing_settings_or_state(tmp_path):
    locator = LocalInstallations(
        tmp_path, "linux", {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert not locator.scan().claude_vscode
    assert list(tmp_path.iterdir()) == []


def test_installed_helper_symlink_resolves_to_its_executable(tmp_path):
    locator = installed_clients(tmp_path)
    helper = tmp_path / "bin/fcc-codex"
    helper.unlink()
    actual = tmp_path / "tool/bin/fcc-codex"
    executable(actual)
    try:
        helper.symlink_to(actual)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this host")
    assert locator.scan().fcc_command == actual
