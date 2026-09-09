"""Behavior contracts for FCC's standalone native-Windows Muse lifecycle."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _powershells() -> tuple[str, ...]:
    return tuple(
        executable
        for name in ("powershell", "pwsh")
        if (executable := shutil.which(name)) is not None
    )


POWERSHELLS = _powershells()


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_installer_functions(
    powershell: str, body: str
) -> subprocess.CompletedProcess[str]:
    script_path = _repo_root() / "scripts" / "install-muse.ps1"
    script = f"""$ErrorActionPreference = 'Stop'
. {_ps_literal(script_path)}
{body}
"""
    return subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("AMD64", "x86_windows"),
        ("X64", "x86_windows"),
        ("X86_64", "x86_windows"),
        ("ARM64", "aarch64_windows"),
    ],
)
def test_muse_installer_maps_native_windows_architecture(
    powershell: str, architecture: str, expected: str
) -> None:
    result = _run_installer_functions(
        powershell,
        f"Write-Output (Get-MuseArtifactKey -Architecture {_ps_literal(architecture)})",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize("architecture", ["X86", "ARM", "unknown"])
def test_muse_installer_rejects_unsupported_architecture(
    powershell: str, architecture: str
) -> None:
    result = _run_installer_functions(
        powershell,
        f"Get-MuseArtifactKey -Architecture {_ps_literal(architecture)}",
    )

    assert result.returncode != 0
    assert "supported Windows release" in result.stderr


def _channel() -> dict[str, object]:
    return {
        "channel": "muse-stable",
        "state": "public",
        "version": "0.2.1-R1215.1",
        "manifest_url": "https://lookaside.facebook.com/muse/releases/0.2.1.json",
    }


def _manifest() -> dict[str, object]:
    return {
        "version": "0.2.1-R1215.1",
        "checksum_algorithm": "sha256",
        "artifacts": {
            "x86_windows": {
                "url": "https://lookaside.facebook.com/muse/muse-x86-windows.exe",
                "checksum": "a" * 64,
                "size": 123,
            },
            "aarch64_windows": {
                "url": "https://lookaside.facebook.com/muse/muse-aarch64-windows.exe",
                "checksum": "b" * 64,
                "size": 456,
            },
        },
    }


def _resolve_artifact(
    powershell: str,
    *,
    channel: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    architecture: str = "X64",
) -> subprocess.CompletedProcess[str]:
    channel_json = json.dumps(channel or _channel(), separators=(",", ":"))
    manifest_json = json.dumps(manifest or _manifest(), separators=(",", ":"))
    return _run_installer_functions(
        powershell,
        f"""$channel = ConvertFrom-Json {_ps_literal(channel_json)}
$manifest = ConvertFrom-Json {_ps_literal(manifest_json)}
Resolve-MuseArtifact -Architecture {_ps_literal(architecture)} -Channel $channel -Manifest $manifest |
    ConvertTo-Json -Compress
""",
    )


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_resolves_version_bound_verified_artifact(
    powershell: str,
) -> None:
    result = _resolve_artifact(powershell)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "ReleaseVersion": "0.2.1-R1215.1",
        "ArtifactKey": "x86_windows",
        "Url": "https://lookaside.facebook.com/muse/muse-x86-windows.exe",
        "Sha256": "a" * 64,
        "Size": 123,
    }


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("channel", "channel", "preview"), "stable channel"),
        (("channel", "state", "private"), "public"),
        (("channel", "version", "latest"), "release version"),
        (
            ("channel", "manifest_url", "http://lookaside.facebook.com/muse.json"),
            "HTTPS",
        ),
        (
            ("channel", "manifest_url", "https://example.com/muse.json"),
            "lookaside.facebook.com",
        ),
        (("manifest", "version", "0.2.2-R1"), "does not match"),
        (("manifest", "checksum_algorithm", "sha512"), "sha256"),
        (("artifact", "checksum", "ABC"), "checksum"),
        (("artifact", "size", 0), "size"),
        (("artifact", "url", "https://example.com/muse.exe"), "lookaside.facebook.com"),
    ],
)
def test_muse_installer_rejects_untrusted_release_metadata(
    powershell: str, mutation: tuple[str, str, object], message: str
) -> None:
    channel = _channel()
    manifest = _manifest()
    target, field, value = mutation
    if target == "channel":
        channel[field] = value
    elif target == "manifest":
        manifest[field] = value
    else:
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, dict)
        artifact = artifacts["x86_windows"]
        assert isinstance(artifact, dict)
        artifact[field] = value

    result = _resolve_artifact(
        powershell,
        channel=channel,
        manifest=manifest,
    )

    assert result.returncode != 0
    assert message.lower() in result.stderr.lower()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_rejects_missing_architecture_artifact(powershell: str) -> None:
    manifest = _manifest()
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    del artifacts["x86_windows"]

    result = _resolve_artifact(powershell, manifest=manifest)

    assert result.returncode != 0
    assert "x86_windows" in result.stderr


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_validates_owner_record(tmp_path: Path, powershell: str) -> None:
    record_path = tmp_path / ".fcc-muse-install.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner": "free-claude-code-muse-installer",
                "release_version": "0.2.1-R1215.1",
                "artifact_key": "x86_windows",
                "sha256": "a" * 64,
                "size": 123,
            }
        ),
        encoding="utf-8",
    )
    result = _run_installer_functions(
        powershell,
        f"Read-MuseOwnershipRecord -RecordPath {_ps_literal(record_path)} | ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["owner"] == "free-claude-code-muse-installer"


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize(
    "record",
    [
        "not-json",
        json.dumps({"schema_version": 1, "owner": "someone-else"}),
        json.dumps(
            {
                "schema_version": 2,
                "owner": "free-claude-code-muse-installer",
            }
        ),
    ],
)
def test_muse_installer_rejects_malformed_or_foreign_owner_record(
    tmp_path: Path, powershell: str, record: str
) -> None:
    record_path = tmp_path / ".fcc-muse-install.json"
    record_path.write_text(record, encoding="utf-8")

    result = _run_installer_functions(
        powershell,
        f"Read-MuseOwnershipRecord -RecordPath {_ps_literal(record_path)}",
    )

    assert result.returncode != 0
    assert "ownership record" in result.stderr.lower()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_normalizes_managed_path_once_and_preserves_order(
    tmp_path: Path, powershell: str
) -> None:
    managed = tmp_path / "Tools" / "Muse" / "bin"
    first = tmp_path / "One"
    second = tmp_path / "Two"
    path_value = os.pathsep.join(
        (
            str(first),
            str(managed).swapcase() + os.sep,
            str(second),
            str(managed).upper(),
        )
    )
    result = _run_installer_functions(
        powershell,
        f"""$managed = {_ps_literal(managed)}
$path = {_ps_literal(path_value)}
Write-Output (Add-MusePathValue -PathValue $path -ManagedBin $managed)
Write-Output (Remove-MusePathValue -PathValue $path -ManagedBin $managed)
""",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        os.pathsep.join((str(managed), str(first), str(second))),
        os.pathsep.join((str(first), str(second))),
    ]


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_checks_exact_artifact_size_and_sha256(
    tmp_path: Path, powershell: str
) -> None:
    artifact = tmp_path / "muse.exe"
    payload = b"verified muse fixture\n"
    artifact.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    command = (
        f"Assert-MuseArtifactIntegrity -Path {_ps_literal(artifact)} "
        f"-ExpectedSize {len(payload)} -ExpectedSha256 {_ps_literal(checksum)}"
    )

    accepted = _run_installer_functions(powershell, command)
    wrong_size = _run_installer_functions(
        powershell,
        command.replace(f"-ExpectedSize {len(payload)}", "-ExpectedSize 1"),
    )
    wrong_hash = _run_installer_functions(
        powershell,
        command.replace(checksum, "f" * 64),
    )

    assert accepted.returncode == 0, accepted.stderr
    assert wrong_size.returncode != 0
    assert "size" in wrong_size.stderr.lower()
    assert wrong_hash.returncode != 0
    assert "sha-256" in wrong_hash.stderr.lower()


def _managed_paths(local_app_data: Path) -> tuple[Path, Path, Path]:
    root = local_app_data / "Programs" / "Muse Code"
    return root, root / "bin" / "muse.exe", root / ".fcc-muse-install.json"


def _owner_record(
    payload: bytes,
    *,
    release_version: str = "0.2.1-R1215.1",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "owner": "free-claude-code-muse-installer",
        "release_version": release_version,
        "artifact_key": "x86_windows",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _run_installer_lifecycle(
    tmp_path: Path,
    powershell: str,
    *,
    payload: bytes = b"new verified muse binary",
    release_version: str = "0.2.1-R1215.1",
    external_path: str = "",
    external_compatible: bool = True,
    publish_failure: bool = False,
    dry_run: bool = False,
    install_root_is_file: bool = False,
    empty_path_entries: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    local_app_data = tmp_path / "local-app-data"
    if install_root_is_file:
        root, _, _ = _managed_paths(local_app_data)
        root.parent.mkdir(parents=True)
        root.write_bytes(b"unowned file")
    source_artifact = tmp_path / "official-muse.exe"
    source_artifact.write_bytes(payload)
    call_log = tmp_path / "calls.log"
    record = _owner_record(payload, release_version=release_version)
    record_json = json.dumps(record, separators=(",", ":"))
    script_path = _repo_root() / "scripts" / "install-muse.ps1"
    process_path = os.pathsep.join(
        (str(tmp_path / "process-one"), str(tmp_path / "process-two"))
    )
    user_path = os.pathsep.join(
        (str(tmp_path / "user-one"), str(tmp_path / "user-two"))
    )
    if empty_path_entries:
        process_path = os.pathsep + process_path + os.pathsep
        user_path = os.pathsep + user_path + os.pathsep
    script = f"""$ErrorActionPreference = 'Stop'
$env:LOCALAPPDATA = {_ps_literal(local_app_data)}
$env:Path = {_ps_literal(process_path)}
. {_ps_literal(script_path)}
$DryRun = ${str(dry_run).lower()}
$script:TestUserPath = {_ps_literal(user_path)}
function Get-MuseNativeArchitecture {{ return 'X64' }}
function Resolve-MuseExternalCommand {{
    if ({_ps_literal(external_path)} -eq '') {{ return $null }}
    return [pscustomobject] @{{ Source = {_ps_literal(external_path)} }}
}}
function Get-MuseVersionProbe {{
    param([string] $Path)
    $isExternal = $Path -eq {_ps_literal(external_path)} -and {_ps_literal(external_path)} -ne ''
    if ($isExternal -and -not ${str(external_compatible).lower()}) {{
        return [pscustomobject] @{{ Compatible = $false; Version = $null; Reason = 'not Muse Code' }}
    }}
    return [pscustomobject] @{{ Compatible = $true; Version = [version] '0.2.1'; Reason = '' }}
}}
function Get-MuseReleaseArtifact {{
    Add-Content -LiteralPath {_ps_literal(call_log)} -Value 'metadata'
    $record = ConvertFrom-Json {_ps_literal(record_json)}
    return [pscustomobject] @{{
        ReleaseVersion = $record.release_version
        ArtifactKey = $record.artifact_key
        Url = 'https://lookaside.facebook.com/muse/fixture.exe'
        Sha256 = $record.sha256
        Size = [long] $record.size
    }}
}}
function Save-MuseArtifact {{
    param([string] $Url, [string] $Destination)
    Add-Content -LiteralPath {_ps_literal(call_log)} -Value 'download'
    Copy-Item -LiteralPath {_ps_literal(source_artifact)} -Destination $Destination
}}
function Get-MuseUserPathValue {{ return $script:TestUserPath }}
function Set-MuseUserPathValue {{
    param([AllowEmptyString()][string] $Value)
    Add-Content -LiteralPath {_ps_literal(call_log)} -Value 'path-write'
    $script:TestUserPath = $Value
}}
"""
    if publish_failure:
        script += (
            "function Publish-MuseExecutable { throw 'injected publication failure' }\n"
        )
    script += """Invoke-MuseInstaller
[pscustomobject] @{ UserPath = $script:TestUserPath; ProcessPath = $env:Path } |
    ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, local_app_data, call_log


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_preserves_compatible_external_installation(
    tmp_path: Path, powershell: str
) -> None:
    result, local_app_data, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        external_path=r"C:\External Muse\muse.exe",
    )
    root, executable, record_path = _managed_paths(local_app_data)

    assert result.returncode == 0, result.stderr
    assert "leaving it unchanged" in result.stdout
    assert not root.exists()
    assert not executable.exists()
    assert not record_path.exists()
    assert not call_log.exists()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_rejects_conflicting_external_command_before_fetch(
    tmp_path: Path, powershell: str
) -> None:
    result, local_app_data, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        external_path=r"C:\Unrelated\muse.exe",
        external_compatible=False,
    )

    assert result.returncode != 0
    assert r"C:\Unrelated\muse.exe" in result.stderr
    assert "conflict" in result.stderr.lower()
    assert not local_app_data.exists()
    assert not call_log.exists()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_rejects_file_at_managed_root_before_fetch(
    tmp_path: Path, powershell: str
) -> None:
    result, local_app_data, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        install_root_is_file=True,
    )
    root, executable, record_path = _managed_paths(local_app_data)

    assert result.returncode != 0
    assert "not owned by FCC" in result.stderr
    assert root.read_bytes() == b"unowned file"
    assert not executable.exists()
    assert not record_path.exists()
    assert not call_log.exists()


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize("empty_path_entries", [False, True])
def test_muse_installer_publishes_verified_binary_record_and_path(
    tmp_path: Path, powershell: str, empty_path_entries: bool
) -> None:
    payload = b"fresh Muse payload"
    result, local_app_data, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        payload=payload,
        empty_path_entries=empty_path_entries,
    )
    root, executable, record_path = _managed_paths(local_app_data)

    assert result.returncode == 0, result.stderr
    assert executable.read_bytes() == payload
    assert json.loads(record_path.read_text(encoding="utf-8-sig")) == _owner_record(
        payload
    )
    state = json.loads(result.stdout.splitlines()[-1])
    for field, prefix in (("UserPath", "user"), ("ProcessPath", "process")):
        expected = [str(tmp_path / f"{prefix}-one"), str(tmp_path / f"{prefix}-two")]
        if empty_path_entries:
            expected = ["", *expected, ""]
        assert state[field] == os.pathsep.join([str(root / "bin"), *expected])
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "metadata",
        "download",
        "path-write",
    ]


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_skips_download_only_after_revalidating_current_binary(
    tmp_path: Path, powershell: str
) -> None:
    payload = b"current Muse payload"
    local_app_data = tmp_path / "local-app-data"
    _, executable, record_path = _managed_paths(local_app_data)
    executable.parent.mkdir(parents=True)
    executable.write_bytes(payload)
    record_path.write_text(json.dumps(_owner_record(payload)), encoding="utf-8")

    result, _, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        payload=payload,
    )

    assert result.returncode == 0, result.stderr
    assert "already current and verified" in result.stdout
    assert executable.read_bytes() == payload
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "metadata",
        "path-write",
    ]


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_repairs_corrupt_managed_binary(
    tmp_path: Path, powershell: str
) -> None:
    payload = b"repaired Muse payload"
    local_app_data = tmp_path / "local-app-data"
    _, executable, record_path = _managed_paths(local_app_data)
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"corrupt")
    record_path.write_text(json.dumps(_owner_record(payload)), encoding="utf-8")

    result, _, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        payload=payload,
    )

    assert result.returncode == 0, result.stderr
    assert executable.read_bytes() == payload
    assert "download" in call_log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_updates_changed_release(
    tmp_path: Path, powershell: str
) -> None:
    old_payload = b"old Muse payload"
    new_payload = b"new Muse payload"
    local_app_data = tmp_path / "local-app-data"
    _, executable, record_path = _managed_paths(local_app_data)
    executable.parent.mkdir(parents=True)
    executable.write_bytes(old_payload)
    record_path.write_text(
        json.dumps(_owner_record(old_payload, release_version="0.2.1-R1")),
        encoding="utf-8",
    )

    result, _, _ = _run_installer_lifecycle(
        tmp_path,
        powershell,
        payload=new_payload,
        release_version="0.2.2-R2",
    )

    assert result.returncode == 0, result.stderr
    assert executable.read_bytes() == new_payload
    assert json.loads(record_path.read_text(encoding="utf-8-sig")) == _owner_record(
        new_payload,
        release_version="0.2.2-R2",
    )


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_failed_update_keeps_previous_binary_and_record(
    tmp_path: Path, powershell: str
) -> None:
    old_payload = b"old working Muse payload"
    new_payload = b"new Muse payload"
    old_record = _owner_record(old_payload, release_version="0.2.1-R1")
    local_app_data = tmp_path / "local-app-data"
    _, executable, record_path = _managed_paths(local_app_data)
    executable.parent.mkdir(parents=True)
    executable.write_bytes(old_payload)
    record_path.write_text(json.dumps(old_record), encoding="utf-8")

    result, _, _ = _run_installer_lifecycle(
        tmp_path,
        powershell,
        payload=new_payload,
        release_version="0.2.2-R2",
        publish_failure=True,
    )

    assert result.returncode != 0
    assert "injected publication failure" in result.stderr
    assert executable.read_bytes() == old_payload
    assert json.loads(record_path.read_text(encoding="utf-8-sig")) == old_record


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_refuses_unowned_managed_directory(
    tmp_path: Path, powershell: str
) -> None:
    local_app_data = tmp_path / "local-app-data"
    root, _, _ = _managed_paths(local_app_data)
    root.mkdir(parents=True)
    (root / "unowned.txt").write_text("keep", encoding="utf-8")

    result, _, call_log = _run_installer_lifecycle(tmp_path, powershell)

    assert result.returncode != 0
    assert "not owned" in result.stderr.lower()
    assert (root / "unowned.txt").read_text(encoding="utf-8") == "keep"
    assert not call_log.exists()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_installer_dry_run_has_no_network_or_state_mutation(
    tmp_path: Path, powershell: str
) -> None:
    result, local_app_data, call_log = _run_installer_lifecycle(
        tmp_path,
        powershell,
        dry_run=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert not local_app_data.exists()
    assert not call_log.exists()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_native_process_runner_times_out_and_stops_process(
    powershell: str,
) -> None:
    result = _run_installer_functions(
        powershell,
        f"""$result = Invoke-MuseNativeProcess `
    -FilePath {_ps_literal(powershell)} `
    -Arguments @('-NoProfile', '-Command', 'Start-Sleep -Seconds 10') `
    -TimeoutMilliseconds 50
Write-Output ($result | ConvertTo-Json -Compress)
""",
    )

    assert result.returncode == 0, result.stderr
    process_result = json.loads(result.stdout)
    assert process_result["TimedOut"] is True
    assert process_result["ExitCode"] is None


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize(
    ("started", "timed_out", "exit_code", "output", "compatible", "reason"),
    [
        (False, False, 0, "", False, "could not be started"),
        (True, True, 0, "", False, "timed out"),
        (True, False, 7, "", False, "returned exit code 7"),
        (
            True,
            False,
            0,
            "another muse 0.2.1",
            False,
            "did not identify itself as Muse Code",
        ),
        (
            True,
            False,
            0,
            "Muse Code 0.2.0",
            False,
            "is older than Muse Code 0.2.1",
        ),
        (True, False, 0, "Muse Code 0.2.1 (0.2.1-R1215.1)", True, ""),
    ],
)
def test_muse_version_probe_classifies_only_supported_native_binary(
    powershell: str,
    started: bool,
    timed_out: bool,
    exit_code: int,
    output: str,
    compatible: bool,
    reason: str,
) -> None:
    result = _run_installer_functions(
        powershell,
        f"""function Invoke-MuseNativeProcess {{
    return [pscustomobject] @{{
        Started = ${str(started).lower()}
        TimedOut = ${str(timed_out).lower()}
        ExitCode = {exit_code}
        Output = {_ps_literal(output)}
    }}
}}
$probe = Get-MuseVersionProbe -Path {_ps_literal(r"C:\fixture\muse.exe")}
[pscustomobject] @{{ Compatible = $probe.Compatible; Reason = $probe.Reason }} |
    ConvertTo-Json -Compress
""",
    )

    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)
    assert probe == {"Compatible": compatible, "Reason": reason}


def _run_uninstaller_lifecycle(
    tmp_path: Path,
    powershell: str,
    *,
    dry_run: bool = False,
    user_path: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    local_app_data = tmp_path / "local-app-data"
    root, _, _ = _managed_paths(local_app_data)
    managed_bin = root / "bin"
    path_value = user_path or os.pathsep.join(
        (
            str(tmp_path / "one"),
            str(managed_bin),
            str(tmp_path / "two"),
            str(managed_bin) + os.sep,
        )
    )
    script_path = _repo_root() / "scripts" / "uninstall-muse.ps1"
    script = f"""$ErrorActionPreference = 'Stop'
$env:LOCALAPPDATA = {_ps_literal(local_app_data)}
$env:Path = {_ps_literal(path_value)}
. {_ps_literal(script_path)}
$DryRun = ${str(dry_run).lower()}
$script:TestUserPath = {_ps_literal(path_value)}
function Get-MuseUserPathValue {{ return $script:TestUserPath }}
function Set-MuseUserPathValue {{
    param([AllowEmptyString()][string] $Value)
    $script:TestUserPath = $Value
}}
Invoke-MuseUninstaller
[pscustomobject] @{{ UserPath = $script:TestUserPath; ProcessPath = $env:Path }} |
    ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, local_app_data


def _create_managed_install(
    local_app_data: Path,
    *,
    unknown_file: bool = False,
    record: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    payload = b"managed Muse binary"
    root, executable, record_path = _managed_paths(local_app_data)
    executable.parent.mkdir(parents=True)
    executable.write_bytes(payload)
    record_path.write_text(
        json.dumps(record or _owner_record(payload)),
        encoding="utf-8",
    )
    residue_id = "0123456789abcdef0123456789abcdef"
    (executable.parent / f".muse-{residue_id}.staging.exe").write_bytes(b"staging")
    (executable.parent / f".muse-{residue_id}.backup.exe").write_bytes(b"backup")
    (root / f".fcc-muse-install.json.{residue_id}.tmp").write_bytes(b"temporary")
    (root / f".fcc-muse-install.json.{residue_id}.backup").write_bytes(b"backup")
    if unknown_file:
        near_match_id = "fedcba9876543210fedcba9876543210"
        (root / "keep.txt").write_text("not installer owned", encoding="utf-8")
        (executable.parent / ".muse-user-notes.staging.exe").write_bytes(b"keep")
        (executable.parent / f".MUSE-{near_match_id}.backup.exe").write_bytes(b"keep")
        (root / ".fcc-muse-install.json.user.tmp").write_bytes(b"keep")
        (root / f".FCC-MUSE-INSTALL.JSON.{near_match_id}.backup").write_bytes(b"keep")
    return root, executable, record_path


@pytest.mark.parametrize("powershell", POWERSHELLS)
@pytest.mark.parametrize("empty_path_entries", [False, True])
def test_muse_uninstaller_removes_only_managed_assets_and_exact_path_entries(
    tmp_path: Path, powershell: str, empty_path_entries: bool
) -> None:
    local_app_data = tmp_path / "local-app-data"
    root, executable, record_path = _create_managed_install(local_app_data)
    muse_state = tmp_path / "muse-state" / "sessions" / "state.json"
    muse_state.parent.mkdir(parents=True)
    muse_state.write_text('{"native":true}', encoding="utf-8")

    path_entries = [str(tmp_path / "one"), str(tmp_path / "two")]
    if empty_path_entries:
        path_entries = ["", *path_entries, "", ""]
    user_path = os.pathsep.join(
        [str(root / "bin"), *path_entries, str(root / "bin") + os.sep]
    )
    result, _ = _run_uninstaller_lifecycle(tmp_path, powershell, user_path=user_path)

    assert result.returncode == 0, result.stderr
    assert not executable.exists()
    assert not record_path.exists()
    assert not root.exists()
    assert muse_state.read_text(encoding="utf-8") == '{"native":true}'
    state = json.loads(result.stdout.splitlines()[-1])
    expected_path = os.pathsep.join(path_entries)
    assert state == {
        "UserPath": expected_path,
        "ProcessPath": expected_path,
    }


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_uninstaller_preserves_unknown_files_and_nonempty_root(
    tmp_path: Path, powershell: str
) -> None:
    local_app_data = tmp_path / "local-app-data"
    root, executable, record_path = _create_managed_install(
        local_app_data,
        unknown_file=True,
    )

    result, _ = _run_uninstaller_lifecycle(tmp_path, powershell)

    assert result.returncode == 0, result.stderr
    assert not executable.exists()
    assert not record_path.exists()
    assert (root / "keep.txt").read_text(encoding="utf-8") == "not installer owned"
    assert (root / "bin" / ".muse-user-notes.staging.exe").read_bytes() == b"keep"
    assert (
        root / "bin" / ".MUSE-fedcba9876543210fedcba9876543210.backup.exe"
    ).read_bytes() == b"keep"
    assert (root / ".fcc-muse-install.json.user.tmp").read_bytes() == b"keep"
    assert (
        root / ".FCC-MUSE-INSTALL.JSON.fedcba9876543210fedcba9876543210.backup"
    ).read_bytes() == b"keep"
    assert "unknown files" in result.stdout.lower()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_uninstaller_refuses_foreign_record_without_mutation(
    tmp_path: Path, powershell: str
) -> None:
    local_app_data = tmp_path / "local-app-data"
    foreign = _owner_record(b"managed Muse binary")
    foreign["owner"] = "another-installer"
    _, executable, record_path = _create_managed_install(
        local_app_data,
        record=foreign,
    )
    before = record_path.read_bytes()

    result, _ = _run_uninstaller_lifecycle(tmp_path, powershell)

    assert result.returncode != 0
    assert "ownership record" in result.stderr.lower()
    assert executable.read_bytes() == b"managed Muse binary"
    assert record_path.read_bytes() == before


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_uninstaller_is_idempotent_when_managed_install_is_absent(
    tmp_path: Path, powershell: str
) -> None:
    result, local_app_data = _run_uninstaller_lifecycle(tmp_path, powershell)

    assert result.returncode == 0, result.stderr
    assert "No FCC-managed Muse Code installation" in result.stdout
    assert not local_app_data.exists()


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_uninstaller_refuses_unowned_root(tmp_path: Path, powershell: str) -> None:
    local_app_data = tmp_path / "local-app-data"
    root, _, _ = _managed_paths(local_app_data)
    root.mkdir(parents=True)
    unknown = root / "keep.txt"
    unknown.write_text("keep", encoding="utf-8")

    result, _ = _run_uninstaller_lifecycle(tmp_path, powershell)

    assert result.returncode != 0
    assert "not owned" in result.stderr.lower()
    assert unknown.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("powershell", POWERSHELLS)
def test_muse_uninstaller_dry_run_is_non_mutating(
    tmp_path: Path, powershell: str
) -> None:
    local_app_data = tmp_path / "local-app-data"
    root, executable, record_path = _create_managed_install(local_app_data)
    managed_bin = root / "bin"

    result, _ = _run_uninstaller_lifecycle(
        tmp_path,
        powershell,
        dry_run=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert executable.exists()
    assert record_path.exists()
    assert (managed_bin / ".muse-0123456789abcdef0123456789abcdef.staging.exe").exists()
    state = json.loads(result.stdout.splitlines()[-1])
    assert str(managed_bin) in state["UserPath"]
    assert str(managed_bin) in state["ProcessPath"]
