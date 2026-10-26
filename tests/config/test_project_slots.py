"""Tests for global per-project config slot paths and FCC_HOME relocation."""

from pathlib import Path

from free_claude_code.config import paths

# The suite-wide isolation fixture replaces ``paths.config_dir_path`` with a
# sandbox lambda before each test; the FCC_HOME override under test lives in
# the real implementation, so capture it while the module is still pristine.
_REAL_CONFIG_DIR_PATH = paths.config_dir_path


def test_slot_name_dashes_every_non_alphanumeric_character() -> None:
    assert paths._project_slot_name(Path(r"C:\work\my-app")) == "C--work-my-app"
    assert paths._project_slot_name(Path("/tmp/My Project")) == "-tmp-My-Project"


def test_slot_name_matches_claude_code_project_directory_naming() -> None:
    # Claude Code stores per-project state under ~/.claude/projects/<dashed>;
    # slots reuse the same collision-free shape so both tools key alike.
    assert paths.project_slot_path(Path(r"C:\work\my-app")).name == ("C--work-my-app")


def test_slot_path_lives_under_the_stock_projects_directory() -> None:
    slot = paths.project_slot_path(Path(r"C:\work\my-app"))
    assert slot == paths._default_config_dir_path() / "projects" / "C--work-my-app"


def test_slot_path_stays_under_default_home_when_fcc_home_is_set(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FCC_HOME", "C:/elsewhere")
    slot = paths.project_slot_path(Path(r"C:\work\my-app"))
    assert slot == paths._default_config_dir_path() / "projects" / "C--work-my-app"


def test_config_dir_path_defaults_to_stock_home(monkeypatch) -> None:
    monkeypatch.delenv("FCC_HOME", raising=False)
    assert _REAL_CONFIG_DIR_PATH() == paths._default_config_dir_path()


def test_config_dir_path_honors_explicit_fcc_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FCC_HOME", str(tmp_path / "slot-home"))
    assert _REAL_CONFIG_DIR_PATH() == tmp_path / "slot-home"


def test_config_dir_path_expands_user_in_fcc_home(monkeypatch) -> None:
    monkeypatch.setenv("FCC_HOME", "~/portable-slot")
    assert _REAL_CONFIG_DIR_PATH() == Path("~/portable-slot").expanduser()
