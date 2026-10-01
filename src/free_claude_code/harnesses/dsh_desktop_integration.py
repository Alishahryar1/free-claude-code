"""Coordinate recoverable, narrowly owned DSH Desktop configuration changes."""

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml.comments import CommentedSeq

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import dsh_desktop_profile as native
from free_claude_code.harnesses import dsh_desktop_state as journal
from free_claude_code.harnesses import dsh_files
from free_claude_code.harnesses.config_file import decode_json
from free_claude_code.harnesses.dsh_config import DSH_PROVIDER_ID, build_dsh_provider
from free_claude_code.harnesses.dsh_desktop_profile import DSH_DESKTOP_API_KEY
from free_claude_code.harnesses.dsh_files import DshConfigError


def config_home() -> Path:
    value = os.environ.get("DSH_HOME", "").strip()
    if value == "~":
        return Path.home()
    if value.startswith(("~/", "~\\")):
        return (Path.home() / value[2:]).resolve()
    return Path(value).resolve() if value else Path.home() / ".dsh"


@dataclass
class _Peers:
    paths: list[Path]
    unknown: bool = False


def _peer_paths(home: Path) -> list[Path]:
    return sorted(
        path
        for path in (home / "profiles").iterdir()
        if path.name != "desktop" and path.is_dir()
    )


@contextmanager
def _locked(
    home: Path, state: Path, *, strict: bool, peers: bool = False
) -> Iterator[tuple[native.Profile, _Peers]]:
    native.validate_paths(home)
    dsh_files.regular_path(state)
    if strict and not native._initialized(home):
        raise DshConfigError(
            "Install and open DeepSeek Harness Desktop once, then retry Configure."
        )
    with ExitStack() as locks:
        locks.enter_context(
            dsh_files.file_lock(home / "profiles/desktop/package.json.lock", wait=2)
        )
        inspection = _Peers(_peer_paths(home) if peers else [])
        for path in inspection.paths:
            try:
                dsh_files.regular_path(path / "package.json", root=home)
                locks.enter_context(
                    dsh_files.file_lock(path / "package.json.lock", wait=0)
                )
            except OSError, ValueError:
                inspection.unknown = True
        locks.enter_context(
            dsh_files.file_lock(home / ".credentials.yaml.lock", wait=30)
        )
        yield native.load(home, strict=strict), inspection


def _retention(profile: native.Profile, peers: _Peers) -> str | None:
    """Deletion is optional; an opaque or busy consumer keeps its credential."""
    if peers.unknown or peers.paths != _peer_paths(profile.home):
        return "The credential was retained because another DSH profile could not be inspected safely."
    try:
        higher_path = profile.home / "cordis.patch.yml"
        before = higher_path.read_bytes() if higher_path.exists() else None
        higher = dsh_files.read_yaml(higher_path, sequence=True)
        assert isinstance(higher, CommentedSeq)
        unknown = False
        for directory in [profile.home / "profiles/desktop", *peers.paths]:
            for name in ("package.json", "cordis.yml", "cordis.patch.yml"):
                dsh_files.regular_path(directory / name, root=profile.home)
            manifest = decode_json(
                (directory / "package.json").read_text(encoding="utf-8-sig")
            )
            if not isinstance(manifest, dict):
                return "The credential was retained because a DSH profile has an unknown configuration."
            metadata = manifest.get("dsh")
            selection = metadata.get("profile") if isinstance(metadata, dict) else None
            bundles = selection.get("bundles") if isinstance(selection, dict) else None
            if bundles != native.STOCK_BUNDLES or dsh_files.read_yaml(
                directory / "cordis.yml", sequence=True
            ):
                unknown = True
                continue
            rows = (
                profile.rows
                if directory.name == "desktop"
                else dsh_files.read_yaml(directory / "cordis.patch.yml", sequence=True)
            )
            assert isinstance(rows, CommentedSeq)
            usage = native.reference_usage(rows, higher)
            if usage == "used":
                return "The credential was retained because another DSH setting still uses it."
            unknown |= usage == "unknown"
        if before != (
            higher_path.read_bytes() if higher_path.exists() else None
        ) or peers.paths != _peer_paths(profile.home):
            unknown = True
        if unknown:
            return "The credential was retained because a custom or dynamic DSH setting may use it."
    except OSError, ValueError, UnicodeError:
        return "The credential was retained because another DSH configuration could not be read safely."
    return None


def _verify(home: Path, expected: journal.Projection) -> None:
    actual = native.load(home, strict=False).projection(expected.defaults)
    if actual != expected:
        raise DshConfigError(
            "DSH configuration changed during the operation. Retry to reconcile the saved changes."
        )


def configure(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    state_path: Path,
    provider_progress_timeout: float,
    only_existing: bool = False,
) -> JsonObject:
    home = home.resolve()
    if not auth_token.strip():
        raise DshConfigError(
            "Configure an FCC proxy authentication token before connecting DSH Desktop."
        )
    override = os.environ.get(DSH_DESKTOP_API_KEY)
    if override is not None and override != auth_token:
        raise DshConfigError(
            "FCC_DSH_DESKTOP_API_KEY in the environment overrides the saved Desktop credential. Remove or update it first."
        )
    provider = build_dsh_provider(
        catalog.models,
        proxy_root_url=proxy_root_url,
        credential_ref=DSH_DESKTOP_API_KEY,
        provider_progress_timeout=provider_progress_timeout,
    )
    if catalog.default_model_id not in {model.wire_slug for model in catalog.models}:
        raise DshConfigError("FCC's default model is missing from its DSH catalog.")
    changed = False
    with _locked(home, state_path, strict=True) as (profile, _):
        record = journal.read(state_path, home)
        if only_existing and (record is None or record.phase == "retained"):
            return {"changed": False, "connected": False}
        if record is not None and record.phase == "pending_disconnect":
            raise DshConfigError(
                "Finish disconnecting DSH Desktop before connecting again."
            )
        owned_defaults = record.owned_defaults if record else []
        before = profile.projection(owned_defaults)
        if record is None:
            if before.routes or before.credential_hash is not None:
                raise DshConfigError(
                    "A reserved FCC provider or credential already exists without an ownership record. Rename that entry before configuring FCC."
                )
        else:
            record.recognize(before)
            if (
                record.phase == "connected"
                and DSH_PROVIDER_ID not in profile.providers()
            ):
                raise DshConfigError(
                    "The FCC route is overridden in DSH. Disconnect its saved ownership before configuring again."
                )
        restoration = (
            record.restoration
            if record and record.restoration
            else profile.restoration()
        )
        if (
            record
            and record.phase != "retained"
            and before.effective_default
            not in [part.effective_default for part in record.projections]
        ):
            restoration = restoration.model_copy(update={"follow_default": False})
        default: JsonObject | None = (
            {"provider": DSH_PROVIDER_ID, "model": catalog.default_model_id}
            if restoration.follow_default
            else None
        )
        profile.configure(provider, default, owned_defaults)
        refs = profile.refs()
        refs[DSH_DESKTOP_API_KEY] = auth_token
        profile.credentials["version"] = 1
        profile.credentials["refs"] = refs
        after = profile.projection(
            [*owned_defaults, *([default] if default is not None else [])]
        )
        pending = journal.Ownership(
            home=str(home),
            phase="pending_connect",
            before=before,
            after=after,
            restoration=restoration,
            retention_reason=None,
        )
        journal.save(state_path, pending)
        changed |= profile.save_credentials()
        changed |= profile.save_profile()
        _verify(home, after)
        journal.save(
            state_path,
            pending.model_copy(
                update={"phase": "connected", "before": after, "after": None}
            ),
        )
    return status(home, proxy_root_url, auth_token, state_path=state_path) | {
        "changed": changed
    }


def disconnect(home: Path, *, state_path: Path) -> JsonObject:
    home = home.resolve()
    if journal.read(state_path, home) is None:
        return _status_record(home, None)
    with _locked(home, state_path, strict=False, peers=True) as (profile, peers):
        record = journal.read(state_path, home)
        if record is None:
            return _status_record(home, None)
        before = profile.projection(record.owned_defaults)
        record.recognize(before, removing=True)
        if record.restoration is not None:
            profile.disconnect(record.restoration, record.owned_defaults)
        reason = (
            _retention(profile, peers) if before.credential_hash is not None else None
        )
        if reason is None:
            profile.refs().pop(DSH_DESKTOP_API_KEY, None)
        after = profile.projection(record.owned_defaults)
        restoration = record.restoration or profile.restoration()
        pending = journal.Ownership(
            home=str(home),
            phase="pending_disconnect",
            before=before,
            after=after,
            restoration=restoration,
            retention_reason=reason,
        )
        journal.save(state_path, pending)
        profile.save_profile()
        if reason is None and "refs" in profile.credentials:
            profile.save_credentials()
        _verify(home, after)
        if reason is not None:
            retained = journal.Ownership(
                home=str(home),
                phase="retained",
                before=journal.Projection(
                    routes=[],
                    credential_hash=after.credential_hash,
                    defaults=[],
                    effective_default=None,
                ),
                after=None,
                restoration=None,
                retention_reason=reason,
            )
            journal.save(state_path, retained)
            return _status_record(home, retained)
        state_path.unlink()
    return _status_record(home, None)


def _status_record(home: Path, record: journal.Ownership | None) -> JsonObject:
    phase = record.phase if record else "disconnected"
    configured = phase in {"connected", "pending_connect", "pending_disconnect"}
    retained = phase == "retained"
    return {
        "connection_state": phase,
        "installed": (home / "profiles/desktop/package.json").is_file(),
        "connected": False,
        "configured": configured,
        "pending": phase.startswith("pending_"),
        "disconnect_pending": phase == "pending_disconnect",
        "credential_retained": retained,
        "retention_reason": record.retention_reason if retained and record else None,
        "inspection_error": None,
        "actions": {
            "configure": not configured,
            "disconnect": configured,
            "refresh": phase in {"connected", "pending_connect"},
        },
        "paths": {
            "desktop_profile": str(home / "profiles/desktop/cordis.patch.yml"),
            "credentials": str(home / ".credentials.yaml"),
        },
    }


def status(
    home: Path, proxy_root_url: str, auth_token: str, *, state_path: Path
) -> JsonObject:
    home = home.resolve()
    try:
        record = journal.read(state_path, home)
    except (ValueError, OSError) as exc:
        return _status_record(home, None) | {
            "connection_state": "blocked",
            "inspection_error": str(exc)
            if isinstance(exc, DshConfigError)
            else "Could not read FCC's DSH ownership record. Check permissions and retry.",
            "actions": {"configure": False, "disconnect": False, "refresh": False},
        }
    result = _status_record(home, record)
    if record is None and not result["installed"]:
        return result
    try:
        profile = native.load(
            home,
            strict=record is None or record.phase in {"connected", "pending_connect"},
        )
        if record:
            observed = profile.projection(record.owned_defaults)
            record.recognize(observed, removing=record.phase == "pending_disconnect")
            route = profile.providers().get(DSH_PROVIDER_ID)
            result["connected"] = (
                record.phase == "connected"
                and isinstance(route, dict)
                and route.get("baseURL") == proxy_root_url.rstrip("/") + "/v1"
                and observed.credential_hash == native._hash(auth_token)
            )
    except (ValueError, OSError, UnicodeError) as exc:
        result["inspection_error"] = (
            str(exc)
            if isinstance(exc, DshConfigError)
            else "Could not inspect DSH Desktop files. Check permissions and finish native edits, then retry."
        )
    return result


def refresh_connected(
    home: Path,
    proxy_root_url: str,
    auth_token: str,
    catalog: ModelCatalog,
    *,
    state_path: Path,
    provider_progress_timeout: float,
) -> bool:
    record = journal.read(state_path, home.resolve())
    if record is None or record.phase == "retained":
        return False
    if record.phase == "pending_disconnect":
        disconnect(home, state_path=state_path)
        return True
    result = configure(
        home,
        proxy_root_url,
        auth_token,
        catalog,
        state_path=state_path,
        provider_progress_timeout=provider_progress_timeout,
        only_existing=True,
    )
    return result["changed"] is True
