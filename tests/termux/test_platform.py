import platform

from free_claude_code.platform import termux


def test_detects_native_termux_arm64(monkeypatch):
    monkeypatch.setattr(termux.sys, "getandroidapilevel", lambda: 35, raising=False)
    monkeypatch.setattr(termux.platform, "machine", lambda: "aarch64")
    environment = termux.detect_termux_environment(
        {
            "TERMUX_VERSION": "0.118.0",
            "PREFIX": "/data/data/com.termux/files/usr",
            "HOME": "/data/data/com.termux/files/home",
        }
    )

    assert environment.termux
    assert environment.android
    assert environment.bionic
    assert environment.arm64
    assert str(environment.prefix) == "/data/data/com.termux/files/usr"
    assert str(environment.home) == "/data/data/com.termux/files/home"
    assert environment.android_api_level == 35


def test_distinguishes_android_from_termux(monkeypatch):
    monkeypatch.setattr(termux.sys, "getandroidapilevel", lambda: 35, raising=False)
    monkeypatch.setattr(termux.platform, "machine", lambda: "x86_64")

    environment = termux.detect_termux_environment(
        {"PREFIX": "/usr", "HOME": "/home/user"}
    )

    assert not environment.termux
    assert environment.android
    assert environment.bionic
    assert not environment.arm64
    assert environment.prefix is None
    assert environment.home is None


def test_does_not_infer_termux_from_architecture_alone(monkeypatch):
    monkeypatch.delattr(termux.sys, "getandroidapilevel", raising=False)
    monkeypatch.setattr(termux.platform, "system", lambda: "Linux")
    monkeypatch.setattr(termux.platform, "release", lambda: "6.8.0-generic")
    monkeypatch.setattr(termux.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.setattr(termux.platform, "machine", lambda: "aarch64")

    environment = termux.detect_termux_environment(
        {
            "TERMUX_VERSION": "0.118.0",
            "PREFIX": "/data/data/com.termux/files/usr",
            "HOME": "/data/data/com.termux/files/home",
        }
    )

    assert not environment.termux
    assert not environment.android
    assert not environment.bionic
    assert environment.arm64


def test_android_kernel_release_is_detected_without_api_helper(monkeypatch):
    monkeypatch.delattr(termux.sys, "getandroidapilevel", raising=False)
    monkeypatch.setattr(termux.platform, "system", lambda: "Linux")
    monkeypatch.setattr(termux.platform, "release", lambda: "5.10.210-android13-8")
    monkeypatch.setattr(termux.platform, "machine", lambda: "arm64")

    assert termux.is_android_runtime()
    assert termux.is_bionic_runtime()
    assert termux.is_termux_runtime(
        {
            "TERMUX_VERSION": "0.118.0",
            "PREFIX": "/data/data/com.termux/files/usr",
            "HOME": "/data/data/com.termux/files/home",
        }
    )


def test_imported_platform_module_uses_the_host_platform():
    assert termux.platform is platform
