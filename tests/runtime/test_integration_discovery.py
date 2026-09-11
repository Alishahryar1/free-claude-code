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
    assert snapshot.jetbrains and snapshot.acp.command
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
    assert not snapshot.acp.command


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


@pytest.mark.parametrize("override", [None, "", "relative/config", "~/.config"])
def test_invalid_xdg_override_uses_default_absolute_target(tmp_path, override):
    locator = installed_clients(tmp_path)
    if override is None:
        locator.environ.pop("XDG_CONFIG_HOME")
    else:
        locator.environ["XDG_CONFIG_HOME"] = override
    assert locator.vscode_settings == tmp_path / ".config/Code/User/settings.json"
    assert locator.vscode_settings.is_absolute()


def test_macos_jetbrains_launcher_is_relative_to_its_manifest(tmp_path):
    contents = tmp_path / "apps/IntelliJ IDEA.app/Contents"
    executable(contents / "MacOS/idea")
    write_json(
        contents / "Resources/product-info.json",
        {"productCode": "IU", "launch": [{"launcherPath": "../MacOS/idea"}]},
    )
    locator = LocalInstallations(
        tmp_path, "darwin", {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert locator.scan().jetbrains


@pytest.mark.parametrize("version", [(20, 19, 0), (21, 7, 3)])
def test_acp_rejects_node_below_its_runtime_requirement(tmp_path, monkeypatch, version):
    locator = installed_clients(tmp_path)
    package = tmp_path / "lib/node_modules/@agentclientprotocol/claude-agent-acp"
    write_json(
        package / "package.json",
        {
            "name": "@agentclientprotocol/claude-agent-acp",
            "bin": {"claude-agent-acp": "dist/index.js"},
            "engines": {"node": ">=22"},
        },
    )
    monkeypatch.setattr(
        type(locator), "_node_version", lambda _self, _node: version, raising=False
    )
    assert not locator.scan().acp.command


def test_absolute_xdg_override_is_used_without_changing_location(tmp_path):
    locator = installed_clients(tmp_path)
    locator.environ["XDG_CONFIG_HOME"] = str(tmp_path / "custom-config")
    assert locator.vscode_settings == tmp_path / "custom-config/Code/User/settings.json"


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_jetbrains_rejects_absolute_launcher_declarations(tmp_path, platform):
    base = tmp_path / "apps/idea"
    if platform == "darwin":
        base = tmp_path / "apps/IDEA.app/Contents/Resources"
    launcher = executable(base / "bin/idea")
    write_json(
        base / "product-info.json",
        {"productCode": "IU", "launch": [{"launcherPath": str(launcher)}]},
    )
    locator = LocalInstallations(
        tmp_path, platform, {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert not locator._jetbrains()


@pytest.mark.parametrize("symlink", [False, True])
def test_jetbrains_launcher_cannot_escape_its_installation(tmp_path, symlink):
    outside = executable(tmp_path / "outside/idea")
    base = tmp_path / "apps/IDEA.app/Contents"
    launcher = base / "MacOS/idea"
    launcher.parent.mkdir(parents=True)
    if symlink:
        try:
            launcher.symlink_to(outside)
        except OSError:
            pytest.skip("Symlink creation is not permitted on this host")
        relative = "../MacOS/idea"
    else:
        relative = "../../../../outside/idea"
    write_json(
        base / "Resources/product-info.json",
        {"productCode": "IU", "launch": [{"launcherPath": relative}]},
    )
    locator = LocalInstallations(
        tmp_path, "darwin", {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert not locator._jetbrains()


def test_macos_manifest_outside_bundle_does_not_expand_launcher_boundary(tmp_path):
    executable(tmp_path / "apps/other/idea")
    write_json(
        tmp_path / "apps/idea/product-info.json",
        {"productCode": "IU", "launch": [{"launcherPath": "../other/idea"}]},
    )
    locator = LocalInstallations(
        tmp_path, "darwin", {}, tmp_path / "bin", (tmp_path / "apps",)
    )
    assert not locator._jetbrains()


@pytest.mark.parametrize(
    "version,requirement,compatible",
    [
        ((22, 0, 0), ">=22", True),
        ((24, 2, 1), ">=22", True),
        ((22, 0, 0), ">=24", False),
        ((24, 1, 0), ">=24.1", True),
        ((24, 0, 9), ">=24.1", False),
        ((24, 1, 2), ">=24.1.2", True),
        ((24, 1, 1), ">=24.1.2", False),
        ((21, 0, 0), ">=20", False),
        ((22, 0, 0), None, False),
        ((22, 0, 0), "^22 || ^24", False),
        ((22, 0, 0), ">=22 <25", False),
        (None, ">=22", False),
        pytest.param((22, 0, 0), ">=" + "9" * 5000, False, id="oversized-minimum"),
    ],
)
def test_acp_honors_supported_package_engine_minimums(
    tmp_path, monkeypatch, version, requirement, compatible
):
    locator = installed_clients(tmp_path)
    package = tmp_path / "lib/node_modules/@agentclientprotocol/claude-agent-acp"
    write_json(
        package / "package.json",
        {
            "name": "@agentclientprotocol/claude-agent-acp",
            "bin": {"claude-agent-acp": "dist/index.js"},
            "engines": {"node": requirement},
        },
    )
    monkeypatch.setattr(
        type(locator), "_node_version", lambda _self, _node: version, raising=False
    )
    assert bool(locator.scan().acp.command) is compatible


@pytest.mark.parametrize("ineligible", ["missing_package", "wsl", "ambiguous_package"])
def test_ineligible_acp_installation_does_not_probe_node(
    tmp_path, monkeypatch, ineligible
):
    locator = installed_clients(tmp_path)
    if ineligible == "missing_package":
        (
            tmp_path
            / "lib/node_modules/@agentclientprotocol/claude-agent-acp/package.json"
        ).unlink()
    elif ineligible == "wsl":
        locator.environ["WSL_DISTRO_NAME"] = "fixture"
    else:
        executable(tmp_path / "bin/claude-agent-acp")
        write_json(
            tmp_path
            / "bin/node_modules/@agentclientprotocol/claude-agent-acp/package.json",
            {
                "name": "@agentclientprotocol/claude-agent-acp",
                "bin": {"claude-agent-acp": "dist/index.js"},
                "engines": {"node": ">=22"},
            },
        )
        executable(
            tmp_path
            / "bin/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
        )
    monkeypatch.setattr(
        type(locator),
        "_node_version",
        lambda *_a: pytest.fail("must not probe"),
        raising=False,
    )
    assert not locator.scan().acp.command
