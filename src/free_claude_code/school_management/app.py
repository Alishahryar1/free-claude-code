"""Local FastAPI backend for the school-management application."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import sys
import threading
import uuid
import webbrowser
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)
SESSION_COOKIE = "school_management_session"
SESSION_DURATION = timedelta(hours=12)
PASSWORD_ITERATIONS = 310_000
MAX_STATE_BYTES = 64_000_000
DEFAULT_USERNAME = "Moussa"
DEFAULT_PASSWORD = "79408093Ms@"
if getattr(sys, "frozen", False):
    STATIC_DIR = Path(sys._MEIPASS) / "school-management"
else:
    STATIC_DIR = Path(__file__).resolve().parents[3] / "school-management"
STATIC_FILES = {"index.html", "styles.css", "app.js"}
_backup_lock = threading.Lock()
_automation_state: dict[str, Any] = {"last_error": None}

MODULES = (
    "dashboard",
    "enrollments",
    "students",
    "classes",
    "dossiers",
    "teachers",
    "payments",
    "expenses",
    "accounting",
    "grades",
    "bulletins",
    "curriculum",
    "attendance",
    "discipline",
    "documents",
    "cards",
    "accounts",
    "settings",
    "audit",
)
CRUD = ("create", "read", "update", "delete")
COLLECTION_MODULES = {
    "students": "students",
    "classes": "classes",
    "teachers": "teachers",
    "payments": "payments",
    "expenses": "expenses",
    "grades": "grades",
    "assessments": "grades",
    "incidents": "discipline",
    "attendance": "attendance",
}
STATE_COLLECTIONS = tuple(COLLECTION_MODULES)
ACADEMIC_LEVELS = {
    "Petit": ("Crèche",),
    "Grand": ("Crèche",),
    "1re": ("Primaire",),
    "2e": ("Primaire",),
    "3e": ("Primaire",),
    "4e": ("Primaire",),
    "5e": ("Primaire",),
    "6e": ("Primaire",),
    "7e": ("Secondaire",),
    "8e": ("Secondaire",),
    "9e": ("Secondaire",),
    "10e CM": ("Lycée",),
    "11e LL": ("Lycée",),
    "11e SES": ("Lycée",),
    "11e S": ("Lycée",),
    "12e TLL": ("Lycée",),
    "12e TAL": ("Lycée",),
    "12e TSS": ("Lycée",),
    "12e TSE": ("Lycée",),
    "12e TSEExp": ("Lycée",),
    "12e TSEco": ("Lycée",),
}
NEXT_ACADEMIC_LEVELS = {
    "Petit": ("Grand",),
    "Grand": ("1re",),
    "1re": ("2e",),
    "2e": ("3e",),
    "3e": ("4e",),
    "4e": ("5e",),
    "5e": ("6e",),
    "6e": ("7e",),
    "7e": ("8e",),
    "8e": ("9e",),
    "9e": ("10e CM",),
    "10e CM": ("11e LL", "11e SES", "11e S"),
    "11e LL": ("12e TLL", "12e TAL"),
    "11e SES": ("12e TSEco",),
    "11e S": ("12e TSS", "12e TSE", "12e TSEExp"),
}
TERMINAL_ACADEMIC_LEVELS = {
    "12e TLL",
    "12e TAL",
    "12e TSS",
    "12e TSE",
    "12e TSEExp",
}
ACADEMIC_LEVEL_ALIASES = {
    "12e TSExp": "12e TSEExp",
    "12e TEXP": "12e TSEExp",
    "TEXP": "12e TSEExp",
}
ACADEMIC_CYCLES_NAMES = {"Crèche", "Primaire", "Secondaire", "Lycée"}
LEGACY_CLASS_LEVELS = {"CP1": "1re", "CE1": "2e", "CE2": "3e", "CM1": "4e", "CM2": "5e"}
ACADEMIC_CYCLES = {
    "Crèche": ["Petit", "Grand"],
    "Primaire": ["1re", "2e", "3e", "4e", "5e", "6e"],
    "Secondaire": ["7e", "8e", "9e"],
    "Lycée": [
        "10e CM",
        "11e LL",
        "11e SES",
        "11e S",
        "12e TLL",
        "12e TAL",
        "12e TSS",
        "12e TSE",
        "12e TSEExp",
        "12e TSEco",
    ],
}


def _permissions(role: str) -> dict[str, dict[str, bool]]:
    permissions = {module: dict.fromkeys(CRUD, False) for module in MODULES}
    if role == "proviseur":
        permissions = {module: dict.fromkeys(CRUD, True) for module in MODULES}
        permissions["accounts"]["activate"] = True
    elif role == "admin":
        permissions = {
            module: {"create": True, "read": True, "update": True, "delete": True}
            for module in MODULES
        }
        for module in ("payments", "expenses", "accounting", "audit"):
            permissions[module]["delete"] = False
        permissions["accounts"]["activate"] = True
    elif role == "censeur":
        for module in ("dashboard", "students", "classes", "dossiers", "documents"):
            permissions[module]["read"] = True
        permissions["teachers"] = dict.fromkeys(CRUD, True)
        permissions["grades"] = dict.fromkeys(CRUD, True)
        permissions["bulletins"] = dict.fromkeys(CRUD, True)
        permissions["curriculum"] = dict.fromkeys(CRUD, True)
        permissions["discipline"] = dict.fromkeys(CRUD, True)
    elif role == "comptable":
        permissions["dashboard"]["read"] = True
        permissions["students"]["read"] = True
        permissions["payments"] = dict.fromkeys(CRUD, True)
        permissions["expenses"] = dict.fromkeys(CRUD, True)
        permissions["accounting"] = dict.fromkeys(CRUD, True)
        permissions["audit"]["read"] = True
    elif role == "econome":
        permissions["dashboard"]["read"] = True
        permissions["enrollments"] = {
            "create": True,
            "read": True,
            "update": True,
            "delete": False,
        }
        permissions["students"] = {
            "create": True,
            "read": True,
            "update": True,
            "delete": False,
        }
        for module in ("payments", "expenses", "accounting"):
            permissions[module] = {
                "create": True,
                "read": True,
                "update": True,
                "delete": False,
            }
        permissions["audit"]["read"] = True
    elif role == "surveillant":
        for module in ("dashboard", "dossiers", "documents"):
            permissions[module]["read"] = True
        permissions["students"] = {
            "create": False,
            "read": True,
            "update": True,
            "delete": False,
        }
        permissions["classes"] = dict.fromkeys(CRUD, True)
        permissions["attendance"] = dict.fromkeys(CRUD, True)
    elif role in {"directeur_primaire", "directeur_secondaire"}:
        for module in (
            "dashboard",
            "students",
            "classes",
            "dossiers",
            "teachers",
            "grades",
            "bulletins",
            "curriculum",
            "attendance",
            "documents",
            "discipline",
        ):
            permissions[module] = dict.fromkeys(CRUD, True)
        permissions["enrollments"] = {
            "create": True,
            "read": True,
            "update": True,
            "delete": False,
        }
        permissions["students"]["delete"] = False
    elif role == "enseignant":
        for module in ("dashboard", "classes", "students", "dossiers", "documents"):
            permissions[module]["read"] = True
        permissions["grades"] = {
            **dict.fromkeys(CRUD, False),
            "create": True,
            "read": True,
            "update": True,
        }
        permissions["bulletins"]["read"] = True
        permissions["bulletins"]["create"] = True
    elif role == "accueil":
        permissions["dashboard"]["read"] = True
        permissions["enrollments"] = {
            "create": True,
            "read": True,
            "update": False,
            "delete": False,
        }
        permissions["students"] = {
            "create": True,
            "read": True,
            "update": False,
            "delete": False,
        }
        permissions["classes"]["read"] = True
        permissions["dossiers"]["read"] = True
    elif role == "user":
        permissions["dashboard"]["read"] = True
        permissions["students"]["read"] = True
        permissions["classes"]["read"] = True
        permissions["dossiers"]["read"] = True
    return permissions


DIRECTOR_CYCLE_SCOPES = {
    "directeur_primaire": ["Crèche", "Primaire"],
    "directeur_secondaire": ["Secondaire", "Lycée"],
}


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp() -> str:
    return _now().isoformat(timespec="seconds")


def _is_iso_date(value: Any) -> bool:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date().isoformat() == value
    except ValueError:
        return False


def _validate_state_shape(state: dict[str, Any], require_all: bool = False) -> None:
    if "school" in state and not isinstance(state["school"], dict):
        raise HTTPException(
            status_code=422, detail="La fiche établissement est invalide."
        )
    for key in STATE_COLLECTIONS:
        if key not in state:
            if require_all:
                raise HTTPException(
                    status_code=422, detail=f"Le registre {key} est absent."
                )
            continue
        records = state[key]
        if not isinstance(records, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            for item in records
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Le registre {key} contient une fiche invalide.",
            )
        identifiers = [item["id"] for item in records]
        if len(identifiers) != len(set(identifiers)):
            raise HTTPException(
                status_code=422,
                detail=f"Le registre {key} contient des identifiants dupliqués.",
            )
    for item in state.get("attendance", []):
        if (
            not isinstance(item.get("studentId"), str)
            or not _is_iso_date(item.get("date"))
            or item.get("status") not in {"Présent", "Absent", "Retard", "Excusé"}
        ):
            raise HTTPException(
                status_code=422,
                detail="Une fiche de présence doit contenir un élève, une date et un statut valides.",
            )


def _database_path() -> Path:
    configured = os.environ.get("SCHOOL_MANAGEMENT_DB")
    if configured:
        return Path(configured)
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return root / "FreeClaudeCode" / "school-management.sqlite3"


def _backup_directory() -> Path:
    configured = os.environ.get("SCHOOL_MANAGEMENT_BACKUP_DIR")
    return (
        Path(configured)
        if configured
        else _database_path().parent / "school-management-backups"
    )


def _automation_settings() -> tuple[bool, int, int]:
    enabled_value = os.environ.get("SCHOOL_MANAGEMENT_AUTO_BACKUP", "true").lower()
    if enabled_value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError("SCHOOL_MANAGEMENT_AUTO_BACKUP doit être un booléen.")
    enabled = enabled_value in {"1", "true", "yes", "on"}
    try:
        interval_hours = int(
            os.environ.get("SCHOOL_MANAGEMENT_BACKUP_INTERVAL_HOURS", "24")
        )
        retention = int(os.environ.get("SCHOOL_MANAGEMENT_BACKUP_RETENTION", "14"))
    except ValueError as error:
        raise ValueError(
            "La configuration des sauvegardes automatiques est invalide."
        ) from error
    if not 1 <= interval_hours <= 8760:
        raise ValueError(
            "L'intervalle de sauvegarde doit être compris entre 1 et 8760 heures."
        )
    if not 1 <= retention <= 365:
        raise ValueError(
            "La rétention de sauvegarde doit être comprise entre 1 et 365 fichiers."
        )
    return enabled, interval_hours, retention


def _create_automatic_backup() -> dict[str, Any] | None:
    with _backup_lock:
        with _connect() as connection:
            row = connection.execute(
                "SELECT state_json, updated_at FROM app_state WHERE id = 1"
            ).fetchone()
        if not row:
            return None

        state = json.loads(row["state_json"])
        backup = {
            "format": "school-management-backup-v1",
            "created_at": _timestamp(),
            "state": state,
        }
        encoded = json.dumps(backup, ensure_ascii=False, indent=2)
        if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
            raise ValueError("Les données dépassent la limite de sauvegarde de 64 Mo.")

        directory = _backup_directory()
        directory.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC)
        filename = f"school-management-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        destination = directory / filename
        temporary = directory / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        _enabled, _interval_hours, retention = _automation_settings()
        backups = sorted(
            directory.glob("school-management-*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for old_backup in backups[retention:]:
            old_backup.unlink()

        result = {
            "filename": filename,
            "created_at": backup["created_at"],
            "size_bytes": destination.stat().st_size,
        }
        _automation_state.update(last_backup=result, last_error=None)
        return result


async def _automatic_backup_loop(interval_hours: int) -> None:
    while True:
        await asyncio.sleep(interval_hours * 60 * 60)
        try:
            await asyncio.to_thread(_create_automatic_backup)
        except Exception:
            _automation_state["last_error"] = (
                "La dernière sauvegarde automatique a échoué; consultez le journal du serveur."
            )
            LOGGER.exception("Échec de la sauvegarde scolaire automatique.")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    migrate()
    enabled, interval_hours, _retention = _automation_settings()
    if enabled:
        try:
            await asyncio.to_thread(_create_automatic_backup)
        except Exception:
            LOGGER.exception("Échec de la sauvegarde initiale des données scolaires.")
            raise
        task = asyncio.create_task(_automatic_backup_loop(interval_hours))
    else:
        task = None
    try:
        yield
    finally:
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def migrate() -> None:
    with _connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            connection.executescript(
                """
                CREATE TABLE users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    permissions_json TEXT NOT NULL,
                    class_scope_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE app_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    module TEXT NOT NULL,
                    record_id TEXT,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX audit_log_created_at_idx ON audit_log(created_at);
                PRAGMA user_version = 1;
                """
            )
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version < 2:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "cycle_scope_json" not in columns:
                connection.execute("ALTER TABLE users ADD COLUMN cycle_scope_json TEXT")
            connection.execute("PRAGMA user_version = 2")
        if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None:
            salt, digest = _hash_password(DEFAULT_PASSWORD)
            connection.execute(
                "INSERT INTO users (id, username, password_hash, role, active, permissions_json, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (
                    str(uuid.uuid4()),
                    DEFAULT_USERNAME,
                    f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}",
                    "proviseur",
                    json.dumps(_permissions("proviseur")),
                    _timestamp(),
                ),
            )
        for row in connection.execute(
            "SELECT id, role, permissions_json FROM users"
        ).fetchall():
            stored_permissions = json.loads(row["permissions_json"])
            role_permissions = _permissions(row["role"])
            updated_permissions = copy.deepcopy(stored_permissions)
            for module, defaults in role_permissions.items():
                updated_permissions.setdefault(module, copy.deepcopy(defaults))
                for action, default in defaults.items():
                    updated_permissions[module].setdefault(action, default)
            if updated_permissions != stored_permissions:
                connection.execute(
                    "UPDATE users SET permissions_json = ? WHERE id = ?",
                    (json.dumps(updated_permissions), row["id"]),
                )
        state_row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
        if state_row:
            state = json.loads(state_row["state_json"])
            original_state = copy.deepcopy(state)
            school = state.setdefault("school", {})
            school.setdefault("year", _current_school_year())
            school.setdefault("academicCycles", ACADEMIC_CYCLES)
            school.setdefault("conductMaximum", 18)
            school.setdefault("promotionThreshold", 10)
            school.setdefault("whatsappMode", "test")
            school.setdefault("countryCallingCode", "223")
            if not isinstance(school.get("academicClosures"), list):
                school["academicClosures"] = []
            state.setdefault("incidents", [])
            state.setdefault("attendance", [])
            for school_class in state.setdefault("classes", []):
                school_class.setdefault(
                    "academicLevel",
                    LEGACY_CLASS_LEVELS.get(school_class.get("name"), ""),
                )
                school_class["academicLevel"] = ACADEMIC_LEVEL_ALIASES.get(
                    school_class["academicLevel"], school_class["academicLevel"]
                )
            for student in state.setdefault("students", []):
                student.setdefault("enrollmentYear", school["year"])
                if not isinstance(student.get("enrollmentHistory"), list):
                    student["enrollmentHistory"] = [
                        {
                            "schoolYear": student["enrollmentYear"],
                            "className": student.get("className", ""),
                            "decision": "inscrit",
                        }
                    ]
            for assessment in state.setdefault("assessments", []):
                assessment.setdefault("schoolYear", school["year"])
            assessments = {
                item.get("id"): item for item in state["assessments"] if item.get("id")
            }
            for grade in state.setdefault("grades", []):
                assessment = assessments.get(grade.get("assessmentId"), {})
                grade.setdefault(
                    "schoolYear", assessment.get("schoolYear") or school["year"]
                )
            if state != original_state:
                connection.execute(
                    "UPDATE app_state SET state_json = ?, updated_at = ? WHERE id = 1",
                    (json.dumps(state, ensure_ascii=False), _timestamp()),
                )


def _current_school_year() -> str:
    now = _now()
    start = now.year if now.month >= 8 else now.year - 1
    return f"{start} — {start + 1}"


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return salt.hex(), digest.hex()


def _password_matches(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
    except ValueError, TypeError:
        return False
    return hmac.compare_digest(actual, expected)


def _row_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "active": bool(row["active"]),
        "permissions": json.loads(row["permissions_json"]),
        "class_scope": json.loads(row["class_scope_json"])
        if row["class_scope_json"]
        else None,
        "cycle_scope": json.loads(row["cycle_scope_json"])
        if row["cycle_scope_json"]
        else None,
        "created_at": row["created_at"],
    }


def _get_user(user_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return _row_user(row) if row else None


def _user_from_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _connect() as connection:
        row = connection.execute(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
            "WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
            (token_hash, _timestamp()),
        ).fetchone()
    if not row or not row["active"]:
        return None
    return _row_user(row)


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    if host == "testclient":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _authorize(user: dict[str, Any], module: str, action: str) -> None:
    if user["role"] == "proviseur":
        return
    if user["permissions"].get(module, {}).get(action) is not True:
        raise HTTPException(
            status_code=403, detail="Cette action n'est pas autorisée pour ce compte."
        )


def _require_user(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "school_user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Connectez-vous pour continuer.")
    return user


def _audit(
    connection: sqlite3.Connection,
    user: dict[str, Any],
    action: str,
    module: str,
    summary: str,
    record_id: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO audit_log (user_id, username, action, module, record_id, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user["id"],
            user["username"],
            action,
            module,
            record_id,
            summary,
            _timestamp(),
        ),
    )


def _scope_class(key: str, record: dict[str, Any], state: dict[str, Any]) -> str | None:
    if key in ("students", "assessments"):
        return record.get("className")
    if key == "incidents":
        student = next(
            (
                item
                for item in state.get("students", [])
                if item.get("id") == record.get("studentId")
            ),
            None,
        )
        return student.get("className") if student else None
    if key == "classes":
        return record.get("name")
    if key == "attendance":
        if record.get("className"):
            return record["className"]
        student = next(
            (
                item
                for item in state.get("students", [])
                if item.get("id") == record.get("studentId")
            ),
            None,
        )
        return student.get("className") if student else None
    if key == "grades":
        assessment = next(
            (
                item
                for item in state.get("assessments", [])
                if item.get("id") == record.get("assessmentId")
            ),
            None,
        )
        if assessment:
            return assessment.get("className")
        student = next(
            (
                item
                for item in state.get("students", [])
                if item.get("id") == record.get("studentId")
            ),
            None,
        )
        return student.get("className") if student else None
    if key == "payments":
        student = next(
            (
                item
                for item in state.get("students", [])
                if item.get("id") == record.get("studentId")
            ),
            None,
        )
        return student.get("className") if student else None
    return None


def _teacher_classes(record: dict[str, Any], state: dict[str, Any]) -> list[str]:
    teacher_name = f"{record.get('firstName', '')} {record.get('lastName', '')}".strip()
    return [
        school_class["name"]
        for school_class in state.get("classes", [])
        if school_class.get("name")
        and (
            school_class.get("teacherId") == record.get("id")
            or school_class.get("teacher") == teacher_name
        )
    ]


def _can_access_record(
    user: dict[str, Any], key: str, record: dict[str, Any], state: dict[str, Any]
) -> bool:
    if key == "teachers" and (user.get("class_scope") or user.get("cycle_scope")):
        return any(
            _can_access_record(user, "classes", {"name": class_name}, state)
            for class_name in _teacher_classes(record, state)
        )
    class_name = _scope_class(key, record, state)
    class_scope = user.get("class_scope")
    if class_scope and class_name is not None and class_name not in class_scope:
        return False
    cycle_scope = user.get("cycle_scope")
    if cycle_scope and class_name is not None:
        school_class = next(
            (
                item
                for item in state.get("classes", [])
                if item.get("name") == class_name
            ),
            None,
        )
        level = _class_level(school_class or {})
        cycle = (
            ACADEMIC_LEVELS.get(level, (school_class.get("level"),))[0]
            if school_class
            else None
        )
        if not cycle or cycle not in cycle_scope:
            return False
    return True


def _filter_state(state: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(state)
    for key, module in COLLECTION_MODULES.items():
        if (
            not user["permissions"].get(module, {}).get("read", False)
            and user["role"] != "proviseur"
        ):
            result[key] = []
            continue
        scope = user.get("class_scope") or user.get("cycle_scope")
        if scope and key in (
            "students",
            "classes",
            "teachers",
            "assessments",
            "grades",
            "payments",
            "incidents",
            "attendance",
        ):
            result[key] = [
                item
                for item in result.get(key, [])
                if _can_access_record(user, key, item, state)
            ]
    return result


def _validate_state_update(
    user: dict[str, Any], old: dict[str, Any], incoming: dict[str, Any]
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    merged = copy.deepcopy(old)
    changes: list[tuple[str, str, str]] = []
    for key, module in COLLECTION_MODULES.items():
        if module not in user["permissions"] and user["role"] != "proviseur":
            continue
        if (
            not user["permissions"].get(module, {}).get("read", False)
            and user["role"] != "proviseur"
        ):
            continue
        before = {item["id"]: item for item in old.get(key, []) if item.get("id")}
        after = {item["id"]: item for item in incoming.get(key, []) if item.get("id")}
        scoped_records = []
        for record_id in before.keys() | after.keys():
            previous, current = before.get(record_id), after.get(record_id)
            if previous == current:
                continue
            if key == "incidents":
                if previous and previous.get("validated"):
                    raise HTTPException(
                        status_code=403,
                        detail="Un incident valide est conserve dans l'historique et ne peut plus etre modifie ou supprime.",
                    )
                if previous is None and current:
                    if current.get("validated"):
                        raise HTTPException(
                            status_code=422,
                            detail="Un incident doit être créé en attente, puis validé séparément.",
                        )
                    current["validated"] = False
                    current["author"] = user["username"]
                    current["validatedBy"] = ""
                    current["validatedAt"] = ""
                    student_exists = any(
                        item.get("id") == current.get("studentId")
                        for item in incoming.get("students", [])
                    )
                    if (
                        not student_exists
                        or not str(current.get("reason", "")).strip()
                        or not re.fullmatch(
                            r"\d{4}-\d{2}-\d{2}", str(current.get("date", ""))
                        )
                        or isinstance(current.get("pointsRemoved"), bool)
                        or not isinstance(current.get("pointsRemoved"), int | float)
                        or not math.isfinite(current["pointsRemoved"])
                        or current["pointsRemoved"] <= 0
                    ):
                        raise HTTPException(
                            status_code=422,
                            detail="Un élève, une date, un motif et un retrait positif sont requis pour signaler un incident.",
                        )
            if key == "attendance" and current is not None:
                if previous is not None:
                    for field in (
                        "studentId",
                        "date",
                        "className",
                        "schoolYear",
                        "recordedBy",
                    ):
                        if field in previous and current.get(field) != previous[field]:
                            raise HTTPException(
                                status_code=422,
                                detail="L'élève, la date, la classe et l'auteur d'une présence sont immuables.",
                            )
                student = next(
                    (
                        item
                        for item in (incoming if current else old).get("students", [])
                        if item.get("id") == current.get("studentId")
                    ),
                    None,
                )
                if (
                    not student
                    or not _is_iso_date(current.get("date"))
                    or current.get("status")
                    not in {"Présent", "Absent", "Retard", "Excusé"}
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="Une présence doit relier un élève, une date et un statut valides.",
                    )
                if previous is None:
                    current["recordedBy"] = user["username"]
                    current["className"] = student.get("className", "")
                    current["schoolYear"] = incoming.get("school", {}).get("year", "")
                else:
                    current["className"] = previous.get(
                        "className", student.get("className", "")
                    )
                    current["schoolYear"] = previous.get(
                        "schoolYear", old.get("school", {}).get("year", "")
                    )
                    current["recordedBy"] = previous.get(
                        "recordedBy", "Historique antérieur"
                    )
            assignment_only = (
                key == "classes"
                and previous is not None
                and current is not None
                and {
                    name
                    for name in previous.keys() | current.keys()
                    if previous.get(name) != current.get(name)
                }
                <= {"teacherId", "teacher"}
            )
            if previous is not None and not _can_access_record(
                user, key, previous, old
            ):
                if current is not None:
                    raise HTTPException(
                        status_code=403,
                        detail="Ce dossier est hors du périmètre de classes autorisé.",
                    )
                continue
            action = (
                "create"
                if previous is None
                else "delete"
                if current is None
                else "update"
            )
            audit_module = "teachers" if assignment_only else module
            if assignment_only:
                _authorize(user, "teachers", "update")
                teacher_id = current.get("teacherId")
                if teacher_id and not any(
                    teacher.get("id") == teacher_id
                    for teacher in incoming.get("teachers", [])
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="Le professeur affecté à cette classe n'existe pas.",
                    )
            else:
                _authorize(user, module, action)
            if key == "students":
                if action in {"create", "delete"}:
                    _authorize(user, "enrollments", action)
                elif action == "update":
                    changed_fields = {
                        name
                        for name in previous.keys() | current.keys()
                        if previous.get(name) != current.get(name)
                    }
                    if changed_fields & {
                        "className",
                        "enrollmentDate",
                        "enrollmentYear",
                        "enrollmentHistory",
                    }:
                        _authorize(user, "enrollments", "update")
            candidate = current or previous
            if not _can_access_record(
                user, key, candidate, incoming if current else old
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Ce dossier est hors du périmètre de classes autorisé.",
                )
            changes.append((audit_module, action, record_id))
        scope = user.get("class_scope") or user.get("cycle_scope")
        if scope:
            retained = [
                item
                for item in old.get(key, [])
                if not _can_access_record(user, key, item, old)
            ]
            scoped_records = [
                item
                for item in incoming.get(key, [])
                if _can_access_record(user, key, item, incoming)
            ]
            merged[key] = retained + scoped_records
        else:
            merged[key] = copy.deepcopy(incoming.get(key, []))
    if "school" in incoming and incoming["school"] != old.get("school"):
        old_school = old.get("school", {})
        new_school = incoming["school"]
        changed_school_fields = {
            key
            for key in old_school.keys() | new_school.keys()
            if old_school.get(key) != new_school.get(key)
        }
        if changed_school_fields & {"curriculum", "academicCycles"}:
            _authorize(user, "curriculum", "update")
        if changed_school_fields - {"curriculum", "academicCycles"}:
            _authorize(user, "settings", "update")
        conduct_maximum = new_school.get("conductMaximum", 18)
        if conduct_maximum not in (18, 20):
            raise HTTPException(
                status_code=422,
                detail="La note maximale de conduite doit être de 18 ou 20 points.",
            )
        merged["school"] = copy.deepcopy(new_school)
        changes.append(("settings", "update", "school"))
    attendance_keys = [
        (item.get("studentId"), item.get("date"))
        for item in merged.get("attendance", [])
    ]
    if len(attendance_keys) != len(set(attendance_keys)):
        raise HTTPException(
            status_code=409,
            detail="Une seule présence par élève et par date peut être enregistrée.",
        )
    return merged, changes


class LoginPayload(BaseModel):
    username: str
    password: str = Field(repr=False)


class StatePayload(BaseModel):
    state: dict[str, Any]


class BackupPayload(BaseModel):
    format: str
    state: dict[str, Any]


class AccountPayload(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=8, repr=False)
    role: str = "user"
    class_scope: list[str] | None = None
    cycle_scope: list[str] | None = None


class CredentialsPayload(BaseModel):
    current_password: str = Field(repr=False)
    username: str
    new_password: str | None = Field(default=None, repr=False)


class AnnualDecision(BaseModel):
    student_id: str
    decision: str
    target_class_id: str | None = None


class CloseAcademicYearPayload(BaseModel):
    school_year: str
    decisions: list[AnnualDecision]


class RosterLinkPayload(BaseModel):
    class_id: str
    school_year: str


def _school_year_start(value: str) -> int | None:
    match = re.fullmatch(r"\s*(\d{4})\s*[—-]\s*(\d{4})\s*", value)
    if not match or int(match.group(2)) != int(match.group(1)) + 1:
        return None
    return int(match.group(1))


def _next_school_year(value: str) -> str:
    start = _school_year_start(value)
    if start is None:
        raise HTTPException(status_code=422, detail="Format d'année scolaire invalide.")
    return f"{start + 1} — {start + 2}"


def _class_level(school_class: dict[str, Any]) -> str:
    level = school_class.get("academicLevel") or LEGACY_CLASS_LEVELS.get(
        school_class.get("name"), ""
    )
    return ACADEMIC_LEVEL_ALIASES.get(level, level)


def _student_year_enrollment(
    student: dict[str, Any], school_year: str
) -> dict[str, Any] | None:
    history = student.get("enrollmentHistory", [])
    if isinstance(history, list):
        for enrollment in history:
            if enrollment.get("schoolYear") == school_year:
                return enrollment
    if student.get("enrollmentYear") == school_year:
        return {"schoolYear": school_year, "className": student.get("className")}
    if not student.get("enrollmentHistory") and not student.get("enrollmentYear"):
        return {"schoolYear": school_year, "className": student.get("className")}
    return None


def _annual_average(
    state: dict[str, Any], student_id: str, school_year: str
) -> float | None:
    assessments = {
        item.get("id"): item for item in state.get("assessments", []) if item.get("id")
    }
    points = weight = 0.0
    for grade in state.get("grades", []):
        if grade.get("studentId") != student_id:
            continue
        assessment = assessments.get(grade.get("assessmentId"), {})
        grade_year = (
            grade.get("schoolYear")
            or assessment.get("schoolYear")
            or state.get("school", {}).get("year")
        )
        if grade_year != school_year:
            continue
        score = grade.get("score")
        if score is None:
            continue
        coefficient = grade.get("coefficient", 1)
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not 0 <= score <= 20
        ):
            raise HTTPException(
                status_code=422,
                detail="Une note annuelle est invalide; les notes doivent être comprises entre 0 et 20.",
            )
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, int | float)
            or not math.isfinite(coefficient)
            or coefficient <= 0
        ):
            raise HTTPException(
                status_code=422,
                detail="Un coefficient annuel est invalide; utilisez une valeur positive.",
            )
        points += score * coefficient
        weight += coefficient
    return points / weight if weight else None


def _analytics_for_state(
    state: dict[str, Any],
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    students = state.get("students", [])
    payments = state.get("payments", [])
    expenses = state.get("expenses", [])
    grades = state.get("grades", [])
    assessments = {
        item.get("id"): item for item in state.get("assessments", []) if item.get("id")
    }
    school_year = state.get("school", {}).get("year", "")
    conduct_maximum = state.get("school", {}).get("conductMaximum", 18)
    if conduct_maximum not in (18, 20):
        conduct_maximum = 18

    def in_period(item: dict[str, Any]) -> bool:
        entry_date = item.get("date")
        if not isinstance(entry_date, str):
            return False
        try:
            parsed_date = date.fromisoformat(entry_date)
        except ValueError:
            return False
        return (start_date is None or parsed_date >= start_date) and (
            end_date is None or parsed_date <= end_date
        )

    def amount(item: dict[str, Any]) -> float:
        value = item.get("amount", 0)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return 0.0
        return float(value) if math.isfinite(value) else 0.0

    period_payments = [item for item in payments if in_period(item)]
    period_expenses = [item for item in expenses if in_period(item)]
    paid_by_student: dict[str, float] = {}
    for payment in payments:
        student_id = payment.get("studentId")
        if student_id:
            paid_by_student[student_id] = paid_by_student.get(student_id, 0.0) + amount(
                payment
            )

    deducted_by_student: dict[str, float] = {}
    for incident in state.get("incidents", []):
        if incident.get("validated") and incident.get("studentId"):
            deducted_by_student[incident["studentId"]] = deducted_by_student.get(
                incident["studentId"], 0.0
            ) + max(0.0, amount({"amount": incident.get("pointsRemoved", 0)}))

    averages: dict[str, dict[str, float | None]] = {}
    subject_averages: dict[str, dict[str, float | None]] = {}
    for student in students:
        student_id = student.get("id")
        if not student_id:
            continue
        student_grades: dict[tuple[str, str], list[dict[str, Any]]] = {}
        student_subject_grades: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for grade in grades:
            if grade.get("studentId") != student_id or grade.get("score") is None:
                continue
            assessment = assessments.get(grade.get("assessmentId"), {})
            year = (
                grade.get("schoolYear") or assessment.get("schoolYear") or school_year
            )
            term = grade.get("term") or ""
            student_grades.setdefault((year, term), []).append(grade)
            subject = grade.get("subject") or ""
            student_subject_grades.setdefault((year, term, subject), []).append(grade)

        for year in {key[0] for key in student_grades} | {school_year}:
            annual_value = _annual_average(state, student_id, year)
            averages[f"{student_id}|{year}|annual"] = {"average": annual_value}
            for grade_year, term in student_grades:
                if grade_year != year:
                    continue
                term_grades = student_grades[(grade_year, term)]
                points = sum(
                    float(grade["score"]) * float(grade.get("coefficient", 1))
                    for grade in term_grades
                )
                weight = sum(
                    float(grade.get("coefficient", 1)) for grade in term_grades
                )
                averages[f"{student_id}|{year}|{term}"] = {
                    "average": points / weight if weight else None
                }
            for grade_year, term, subject in student_subject_grades:
                if grade_year != year:
                    continue
                subject_grades = student_subject_grades[(grade_year, term, subject)]
                points = sum(
                    float(grade["score"]) * float(grade.get("coefficient", 1))
                    for grade in subject_grades
                )
                weight = sum(
                    float(grade.get("coefficient", 1)) for grade in subject_grades
                )
                subject_averages[f"{student_id}|{year}|{term}|{subject}"] = {
                    "average": points / weight if weight else None
                }
            annual_subjects = {
                subject
                for grade_year, _term, subject in student_subject_grades
                if grade_year == year
            }
            for subject in annual_subjects:
                subject_grades = [
                    grade
                    for (
                        grade_year,
                        _term,
                        grade_subject,
                    ), term_grades in student_subject_grades.items()
                    if grade_year == year and grade_subject == subject
                    for grade in term_grades
                ]
                points = sum(
                    float(grade["score"]) * float(grade.get("coefficient", 1))
                    for grade in subject_grades
                )
                weight = sum(
                    float(grade.get("coefficient", 1)) for grade in subject_grades
                )
                subject_averages[f"{student_id}|{year}|annual|{subject}"] = {
                    "average": points / weight if weight else None
                }

    student_financials: dict[str, dict[str, float]] = {}
    class_sizes: dict[str, int] = {}
    for student in students:
        student_id = student.get("id")
        if not student_id:
            continue
        paid = paid_by_student.get(student_id, 0.0)
        fee = amount({"amount": student.get("annualFee", 0)})
        student_financials[student_id] = {
            "totalPaid": paid,
            "outstanding": max(0.0, fee - paid),
            "conductBalance": max(
                0.0,
                float(conduct_maximum) - deducted_by_student.get(student_id, 0.0),
            ),
        }
        enrollment_years = {school_year}
        history = student.get("enrollmentHistory")
        if isinstance(history, list):
            enrollment_years.update(
                item.get("schoolYear")
                for item in history
                if isinstance(item, dict) and item.get("schoolYear")
            )
        if student.get("enrollmentYear"):
            enrollment_years.add(student["enrollmentYear"])
        for year in enrollment_years:
            enrollment = _student_year_enrollment(student, year)
            if not enrollment or (year == school_year and not student.get("active")):
                continue
            class_name = enrollment.get("className")
            if class_name:
                key = f"{year}|{class_name}"
                class_sizes[key] = class_sizes.get(key, 0) + 1

    active_students = [student for student in students if student.get("active")]
    income = sum(amount(item) for item in period_payments)
    spending = sum(amount(item) for item in period_expenses)
    lifetime_income = sum(amount(item) for item in payments)
    lifetime_spending = sum(amount(item) for item in expenses)
    total_outstanding = sum(
        student_financials.get(student.get("id"), {}).get("outstanding", 0.0)
        for student in active_students
    )
    return {
        "dashboard": {
            "activeStudents": len(active_students),
            "activeClasses": sum(
                bool(item.get("active")) for item in state.get("classes", [])
            ),
            "income": lifetime_income,
            "expenses": lifetime_spending,
            "cashBalance": lifetime_income - lifetime_spending,
            "outstanding": total_outstanding,
            "studentsWithOutstanding": sum(
                student_financials.get(student.get("id"), {}).get("outstanding", 0) > 0
                for student in active_students
            ),
            "paidStudents": sum(
                student_financials.get(student.get("id"), {}).get("outstanding", 0) == 0
                for student in active_students
            ),
            "paymentCount": len(payments),
            "expenseCount": len(expenses),
        },
        "studentFinancials": student_financials,
        "averages": averages,
        "subjectAverages": subject_averages,
        "classSizes": class_sizes,
        "period": {
            "income": income,
            "expenses": spending,
            "cashBalance": income - spending,
            "paymentCount": len(period_payments),
            "expenseCount": len(period_expenses),
        },
    }


def _pdf_literal(value: str) -> bytes:
    encoded = value.encode("cp1252", errors="replace")
    escaped = bytearray(b"(")
    for character in encoded:
        if character in (ord("("), ord(")"), ord("\\")):
            escaped.extend(b"\\" + bytes([character]))
        elif character < 32 or character > 126:
            escaped.extend(f"\\{character:03o}".encode("ascii"))
        else:
            escaped.append(character)
    escaped.append(ord(")"))
    return bytes(escaped)


def _build_simple_pdf(lines: list[str]) -> bytes:
    import textwrap

    wrapped = [
        part for line in lines for part in (textwrap.wrap(line, width=92) or [""])
    ]
    page_lines = [wrapped[index : index + 48] for index in range(0, len(wrapped), 48)]
    if not page_lines:
        page_lines = [[]]

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    page_references = []
    for page in page_lines:
        page_object_id = len(objects) + 1
        content_object_id = page_object_id + 1
        page_references.append(f"{page_object_id} 0 R")
        content = bytearray(b"BT\n/F1 10 Tf\n50 790 Td\n14 TL\n")
        for line in page:
            content.extend(_pdf_literal(line))
            content.extend(b" Tj\nT*\n")
        content.extend(b"ET\n")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + bytes(content)
            + b"endstream"
        )
    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(page_references)}] /Count {len(page_lines)} >>"
    ).encode("ascii")

    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(document)


def _phone_for_whatsapp(phone: str, country_code: str = "223") -> str:
    if not re.fullmatch(r"[+()\-\s\d.]+", phone.strip()):
        raise HTTPException(
            status_code=422,
            detail="Le contact WhatsApp contient des caractères invalides.",
        )
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 8:
        if digits.startswith("0") or (
            country_code == "223" and digits[0] not in "6789"
        ):
            raise HTTPException(
                status_code=422,
                detail="Le numéro local ne correspond pas à un contact téléphonique valide.",
            )
        digits = f"{country_code}{digits}"
    if not 8 <= len(digits) <= 15 or digits.startswith("0"):
        raise HTTPException(
            status_code=422,
            detail="Le numéro WhatsApp doit être au format international ou local valide.",
        )
    return digits


def _audit_action(
    user: dict[str, Any],
    action: str,
    module: str,
    summary: str,
    record_id: str | None = None,
) -> None:
    with _connect() as connection:
        _audit(connection, user, action, module, summary, record_id)


app = FastAPI(
    title="Gestion scolaire",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)


@app.middleware("http")
async def local_and_session_middleware(request: Request, call_next):
    migrate()
    if request.url.path.startswith("/api/"):
        if not _is_loopback(request):
            return JSONResponse(
                {"detail": "Cette application est réservée à la machine locale."},
                status_code=403,
            )
        origin = request.headers.get("origin")
        if origin:
            origin_host = urlsplit(origin).hostname or ""
            try:
                if (
                    not ipaddress.ip_address(origin_host).is_loopback
                    and origin_host != "localhost"
                ):
                    return JSONResponse(
                        {"detail": "Origine non autorisée."}, status_code=403
                    )
            except ValueError:
                if origin_host != "localhost":
                    return JSONResponse(
                        {"detail": "Origine non autorisée."}, status_code=403
                    )
    request.state.school_user = _user_from_token(request.cookies.get(SESSION_COOKIE))
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/styles.css", include_in_schema=False)
def styles():
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def script():
    return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(
        STATIC_DIR / "assets" / "school-mark.ico", media_type="image/vnd.microsoft.icon"
    )


@app.get("/api/bootstrap")
def bootstrap(request: Request):
    with _connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
    user = getattr(request.state, "school_user", None)
    return {
        "requires_login": user is None,
        "first_run": count == 1,
        "state": json.loads(row["state_json"]) if row and user else None,
        "user": user,
    }


@app.post("/api/login")
def login(payload: LoginPayload, response: Response):
    migrate()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (payload.username.strip(),),
        ).fetchone()
        if (
            not row
            or not row["active"]
            or not _password_matches(payload.password, row["password_hash"])
        ):
            raise HTTPException(
                status_code=401, detail="Nom d'utilisateur ou mot de passe incorrect."
            )
        token = secrets.token_urlsafe(32)
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (
                hashlib.sha256(token.encode("utf-8")).hexdigest(),
                row["id"],
                (_now() + SESSION_DURATION).isoformat(timespec="seconds"),
            ),
        )
        user = _row_user(row)
        _audit(connection, user, "connexion", "accounts", "Connexion réussie.")
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(SESSION_DURATION.total_seconds()),
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )
    return {"user": user}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    user = getattr(request.state, "school_user", None)
    if token:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with _connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (token_hash,)
            )
            if user:
                _audit(connection, user, "déconnexion", "accounts", "Déconnexion.")
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    return _require_user(request)


@app.get("/api/state")
def get_state(request: Request):
    user = _require_user(request)
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
    state = json.loads(row["state_json"]) if row else None
    visible_state = _filter_state(state, user) if state else None
    return {
        "state": visible_state,
        "analytics": _analytics_for_state(visible_state) if visible_state else None,
    }


@app.get("/api/analytics")
def get_analytics(
    request: Request,
    from_date: str | None = None,
    to_date: str | None = None,
):
    user = _require_user(request)
    _authorize(user, "dashboard", "read")
    try:
        if from_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", from_date):
            raise ValueError("date must be ISO formatted")
        if to_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", to_date):
            raise ValueError("date must be ISO formatted")
        start_date = date.fromisoformat(from_date) if from_date else None
        end_date = date.fromisoformat(to_date) if to_date else None
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail="La période doit utiliser le format AAAA-MM-JJ."
        ) from error
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="La date de début doit précéder ou égaler la date de fin.",
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
    state = json.loads(row["state_json"]) if row else None
    if not state:
        return {"analytics": None}
    return {
        "analytics": _analytics_for_state(
            _filter_state(state, user), start_date=start_date, end_date=end_date
        )
    }


@app.get("/api/reports/financial-summary.pdf")
def financial_summary_pdf(
    request: Request,
    from_date: str | None = None,
    to_date: str | None = None,
):
    user = _require_user(request)
    _authorize(user, "accounting", "read")
    try:
        start_date = date.fromisoformat(from_date) if from_date else None
        end_date = date.fromisoformat(to_date) if to_date else None
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail="La période doit utiliser le format AAAA-MM-JJ."
        ) from error
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="La date de début doit précéder ou égaler la date de fin.",
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Aucune donnée scolaire.")
    state = _filter_state(json.loads(row["state_json"]), user)
    analytics = _analytics_for_state(state, start_date=start_date, end_date=end_date)
    period = analytics["period"]
    dashboard = analytics["dashboard"]
    school = state.get("school", {})
    period_end = to_date or "aujourd'hui"
    period_label = f"{from_date or 'début'} au {period_end}"

    def money_label(value: Any) -> str:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            value = 0
        return f"{value:,.0f}".replace(",", " ") + " FCFA"

    def matches_period(item: dict[str, Any]) -> bool:
        entry_date = item.get("date")
        if not isinstance(entry_date, str):
            return False
        try:
            parsed_date = date.fromisoformat(entry_date)
        except ValueError:
            return False
        return (start_date is None or parsed_date >= start_date) and (
            end_date is None or parsed_date <= end_date
        )

    lines = [
        school.get("name", "Établissement scolaire"),
        "RAPPORT FINANCIER",
        f"Période : {period_label}",
        f"Généré le : {_timestamp()}",
        "",
        "SITUATION DE L'ÉTABLISSEMENT",
        f"Année scolaire : {school.get('year', '—')}",
        f"Élèves actifs : {dashboard['activeStudents']}",
        f"Classes actives : {dashboard['activeClasses']}",
        "",
        "SYNTHÈSE DE LA PÉRIODE",
        f"Recettes : {money_label(period['income'])} ({period['paymentCount']} paiements)",
        f"Dépenses : {money_label(period['expenses'])} ({period['expenseCount']} écritures)",
        f"Solde net : {money_label(period['cashBalance'])}",
        f"Crédits actuels des familles : {money_label(dashboard['outstanding'])}",
        "",
        "RECETTES",
    ]
    students_by_id = {
        item.get("id"): item for item in state.get("students", []) if item.get("id")
    }
    payments = [item for item in state.get("payments", []) if matches_period(item)]
    expenses = [item for item in state.get("expenses", []) if matches_period(item)]
    lines.extend(
        f"{item.get('date', '—')} · "
        f"{students_by_id.get(item.get('studentId'), {}).get('firstName', '')} "
        f"{students_by_id.get(item.get('studentId'), {}).get('lastName', '')} · "
        f"{item.get('method', 'Paiement')} · {money_label(item.get('amount', 0))}"
        for item in payments
    )
    if not payments:
        lines.append("Aucun paiement sur cette période.")
    lines.extend(["", "DÉPENSES"])
    lines.extend(
        f"{item.get('date', '—')} · {item.get('label', item.get('category', 'Dépense'))} · "
        f"{item.get('category', 'Autre')} · {money_label(item.get('amount', 0))}"
        for item in expenses
    )
    if not expenses:
        lines.append("Aucune dépense sur cette période.")

    filename = "rapport-financier-scolaire.pdf"
    return Response(
        content=_build_simple_pdf(lines),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/backup")
def export_backup(request: Request):
    user = _require_user(request)
    _authorize(user, "settings", "read")
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json, updated_at FROM app_state WHERE id = 1"
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Aucune donnée à sauvegarder.")
    return {
        "format": "school-management-backup-v1",
        "created_at": row["updated_at"],
        "state": json.loads(row["state_json"]),
    }


@app.get("/api/automation")
def get_automation_status(request: Request):
    user = _require_user(request)
    _authorize(user, "settings", "read")
    enabled, interval_hours, retention = _automation_settings()
    directory = _backup_directory()
    backups = (
        sorted(
            directory.glob("school-management-*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if directory.exists()
        else []
    )
    latest = None
    if backups:
        latest = {
            "filename": backups[0].name,
            "created_at": datetime.fromtimestamp(
                backups[0].stat().st_mtime, tz=UTC
            ).isoformat(timespec="seconds"),
            "size_bytes": backups[0].stat().st_size,
        }
    return {
        "automatic_backup_enabled": enabled,
        "interval_hours": interval_hours,
        "retention": retention,
        "backup_count": len(backups),
        "latest_backup": latest,
        "last_error": _automation_state.get("last_error"),
    }


@app.post("/api/automation/backup")
def create_backup_now(request: Request):
    user = _require_user(request)
    _authorize(user, "settings", "update")
    try:
        backup = _create_automatic_backup()
    except OSError as error:
        LOGGER.exception("Échec de la sauvegarde scolaire manuelle.")
        raise HTTPException(
            status_code=500,
            detail="La sauvegarde locale a échoué; vérifiez l'espace disque et les droits du dossier.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if backup is None:
        raise HTTPException(
            status_code=404, detail="Aucune donnée scolaire à sauvegarder."
        )
    _audit_action(
        user,
        "sauvegarde",
        "settings",
        "Création d'une sauvegarde locale des données scolaires.",
    )
    return {"backup": backup}


@app.post("/api/backup")
def restore_backup(payload: BackupPayload, request: Request):
    user = _require_user(request)
    _authorize(user, "settings", "update")
    if payload.format != "school-management-backup-v1":
        raise HTTPException(status_code=422, detail="Format de sauvegarde non reconnu.")
    if "users" in payload.state:
        raise HTTPException(
            status_code=422,
            detail="Une sauvegarde scolaire ne doit pas contenir de comptes.",
        )
    required_collections = (
        "students",
        "classes",
        "teachers",
        "payments",
        "expenses",
        "grades",
        "assessments",
        "incidents",
        "attendance",
    )
    if not isinstance(payload.state.get("school"), dict) or any(
        not isinstance(payload.state.get(key), list) for key in required_collections
    ):
        raise HTTPException(
            status_code=422,
            detail="La sauvegarde doit contenir l'etablissement et les registres scolaires.",
        )
    _validate_state_shape(payload.state, require_all=True)
    encoded = json.dumps(payload.state, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
        raise HTTPException(
            status_code=413, detail="La sauvegarde dépasse la limite de 64 Mo."
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
        old = json.loads(row["state_json"]) if row else {}
        merged, changes = _validate_state_update(user, old, payload.state)
        connection.execute(
            "INSERT INTO app_state (id, state_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at",
            (json.dumps(merged, ensure_ascii=False), _timestamp()),
        )
        for module, action, record_id in changes:
            _audit(connection, user, action, module, f"{action} · {module}", record_id)
        _audit(
            connection,
            user,
            "restauration",
            "settings",
            "Restauration d'une sauvegarde scolaire; comptes et mots de passe conservés.",
        )
    return {"ok": True, "updated_at": _timestamp()}


@app.put("/api/state")
def put_state(payload: StatePayload, request: Request):
    user = _require_user(request)
    _validate_state_shape(payload.state)
    encoded = json.dumps(payload.state, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
        raise HTTPException(
            status_code=413, detail="La sauvegarde dépasse la limite de 64 Mo."
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
        old = json.loads(row["state_json"]) if row else {}
        merged, changes = _validate_state_update(user, old, payload.state)
        connection.execute(
            "INSERT INTO app_state (id, state_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at",
            (json.dumps(merged, ensure_ascii=False), _timestamp()),
        )
        for module, action, record_id in changes:
            _audit(connection, user, action, module, f"{action} · {module}", record_id)
    return {"ok": True, "updated_at": _timestamp()}


@app.post("/api/discipline/incidents/{incident_id}/validate")
def validate_disciplinary_incident(incident_id: str, request: Request):
    user = _require_user(request)
    _authorize(user, "discipline", "update")
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Aucun dossier scolaire.")
        state = json.loads(row["state_json"])
        incident = next(
            (
                item
                for item in state.get("incidents", [])
                if item.get("id") == incident_id
            ),
            None,
        )
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident introuvable.")
        if incident.get("validated"):
            raise HTTPException(status_code=409, detail="Cet incident est déjà validé.")
        if not any(
            item.get("id") == incident.get("studentId")
            for item in state.get("students", [])
        ):
            raise HTTPException(
                status_code=422, detail="L'eleve associe a cet incident n'existe pas."
            )
        if not _can_access_record(user, "incidents", incident, state):
            raise HTTPException(
                status_code=403,
                detail="Cet incident est hors du périmètre de classes autorisé.",
            )
        reason = str(incident.get("reason", "")).strip()
        date = incident.get("date", "")
        points = incident.get("pointsRemoved")
        if not reason or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
            raise HTTPException(
                status_code=422,
                detail="Un motif et une date d'incident valides sont requis.",
            )
        try:
            datetime.strptime(str(date), "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=422, detail="La date de l'incident est invalide."
            ) from None
        if (
            isinstance(points, bool)
            or not isinstance(points, int | float)
            or not math.isfinite(points)
            or points <= 0
        ):
            raise HTTPException(
                status_code=422,
                detail="Saisissez le nombre de points a retirer; aucun bareme automatique n'est applique.",
            )
        incident["validated"] = True
        incident["validatedBy"] = user["username"]
        incident["validatedAt"] = _timestamp()
        connection.execute(
            "UPDATE app_state SET state_json = ?, updated_at = ? WHERE id = 1",
            (json.dumps(state, ensure_ascii=False), _timestamp()),
        )
        _audit(
            connection,
            user,
            "validation",
            "discipline",
            f"Retrait de {points:g} point(s) validé · {reason[:120]}",
            incident_id,
        )
    return {"incident": incident}


@app.post("/api/academic/close-year")
def close_academic_year(payload: CloseAcademicYearPayload, request: Request):
    user = _require_user(request)
    _authorize(user, "students", "update")
    _authorize(user, "classes", "read")
    _authorize(user, "grades", "read")
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
        if not row:
            raise HTTPException(
                status_code=409, detail="Aucune année scolaire à clôturer."
            )
        state = json.loads(row["state_json"])
        school = state.get("school", {})
        if payload.school_year != school.get("year"):
            raise HTTPException(
                status_code=409,
                detail="Seule l'année scolaire active peut être clôturée.",
            )
        closures = school.setdefault("academicClosures", [])
        if any(item.get("schoolYear") == payload.school_year for item in closures):
            raise HTTPException(
                status_code=409, detail="Cette année scolaire est déjà clôturée."
            )
        threshold = school.get("promotionThreshold", 10)
        if not isinstance(threshold, int | float) or not 0 <= threshold <= 20:
            raise HTTPException(
                status_code=422, detail="Le seuil de passage doit être entre 0 et 20."
            )
        roster = [
            student
            for student in state.get("students", [])
            if student.get("active", True)
            and _student_year_enrollment(student, payload.school_year)
        ]
        decisions_by_student = {item.student_id: item for item in payload.decisions}
        roster_ids = {student.get("id") for student in roster}
        if (
            len(decisions_by_student) != len(payload.decisions)
            or set(decisions_by_student) != roster_ids
        ):
            raise HTTPException(
                status_code=422,
                detail="La décision doit être renseignée une seule fois pour chaque élève actif de l'année.",
            )
        next_year = _next_school_year(payload.school_year)
        classes = state.get("classes", [])
        changed_students = []
        for student in roster:
            decision = decisions_by_student[student["id"]]
            average = _annual_average(state, student["id"], payload.school_year)
            if average is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Aucune moyenne annuelle calculable pour {student.get('firstName', '')} {student.get('lastName', '')}.",
                )
            current_enrollment = _student_year_enrollment(student, payload.school_year)
            current_class_name = current_enrollment.get("className")
            current_class = next(
                (item for item in classes if item.get("name") == current_class_name),
                None,
            )
            if current_class is None:
                raise HTTPException(
                    status_code=422, detail="Classe annuelle introuvable."
                )
            if not _can_access_record(user, "students", student, state):
                raise HTTPException(
                    status_code=403,
                    detail="Un élève est hors du périmètre de classes autorisé.",
                )
            level = _class_level(current_class)
            if level not in ACADEMIC_LEVELS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Le niveau de la classe {current_class_name} doit être configuré avant la clôture.",
                )
            if decision.decision not in {"promote", "repeat", "graduate"}:
                raise HTTPException(
                    status_code=422, detail="Décision de fin d'année inconnue."
                )
            expected_decision = (
                ("graduate" if level in TERMINAL_ACADEMIC_LEVELS else "promote")
                if average >= threshold
                else "repeat"
            )
            if decision.decision != expected_decision:
                raise HTTPException(
                    status_code=422,
                    detail=f"La moyenne de {student.get('firstName', '')} {student.get('lastName', '')} impose la décision « {expected_decision} » avec le seuil de {threshold}/20.",
                )
            if decision.decision == "graduate":
                if level not in TERMINAL_ACADEMIC_LEVELS or decision.target_class_id:
                    raise HTTPException(
                        status_code=422,
                        detail="La fin de cycle est réservée aux niveaux de Terminale.",
                    )
                destination_name = current_class_name
            elif decision.decision == "repeat":
                if decision.target_class_id:
                    raise HTTPException(
                        status_code=422,
                        detail="Un redoublement conserve la classe actuelle.",
                    )
                destination_name = current_class_name
            else:
                allowed_levels = NEXT_ACADEMIC_LEVELS.get(level, ())
                destination = next(
                    (
                        item
                        for item in classes
                        if item.get("id") == decision.target_class_id
                    ),
                    None,
                )
                if (
                    level in TERMINAL_ACADEMIC_LEVELS
                    or not destination
                    or destination.get("active") is False
                    or not _can_access_record(user, "classes", destination, state)
                    or _class_level(destination) not in allowed_levels
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Choisissez une classe active du niveau suivant ({', '.join(allowed_levels) or 'fin de cycle'}).",
                    )
                destination_name = destination["name"]
            history = student.setdefault("enrollmentHistory", [])
            completed_enrollment = next(
                (
                    item
                    for item in history
                    if item.get("schoolYear") == payload.school_year
                ),
                None,
            )
            if completed_enrollment is None:
                completed_enrollment = {
                    "schoolYear": payload.school_year,
                    "className": current_class_name,
                }
                history.append(completed_enrollment)
            completed_enrollment.update(
                {
                    "className": current_class_name,
                    "decision": decision.decision,
                    "annualAverage": round(average, 2),
                }
            )
            if decision.decision != "graduate":
                student["className"] = destination_name
                student["enrollmentYear"] = next_year
            if decision.decision != "graduate" and not any(
                item.get("schoolYear") == next_year for item in history
            ):
                history.append(
                    {
                        "schoolYear": next_year,
                        "className": destination_name,
                        "decision": "diplômé"
                        if decision.decision == "graduate"
                        else decision.decision,
                        "annualAverage": round(average, 2),
                    }
                )
            if decision.decision == "graduate":
                student["active"] = False
                student["graduationYear"] = payload.school_year
            changed_students.append(
                {
                    "studentId": student["id"],
                    "decision": decision.decision,
                    "average": round(average, 2),
                    "className": destination_name,
                }
            )
        for assessment in state.get("assessments", []):
            assessment.setdefault("schoolYear", payload.school_year)
        for grade in state.get("grades", []):
            grade.setdefault("schoolYear", payload.school_year)
        closure = {
            "schoolYear": payload.school_year,
            "nextSchoolYear": next_year,
            "promotionThreshold": threshold,
            "completedAt": _timestamp(),
            "results": changed_students,
        }
        closures.append(closure)
        school["year"] = next_year
        connection.execute(
            "UPDATE app_state SET state_json = ?, updated_at = ? WHERE id = 1",
            (json.dumps(state, ensure_ascii=False), _timestamp()),
        )
        _audit(
            connection,
            user,
            "clôture annuelle",
            "students",
            f"Clôture {payload.school_year} · {len(changed_students)} décisions; nouvelle année {next_year}.",
            payload.school_year,
        )
    return {"ok": True, "closure": closure, "school_year": next_year}


@app.post("/api/whatsapp/roster")
def prepare_whatsapp_roster(payload: RosterLinkPayload, request: Request):
    user = _require_user(request)
    _authorize(user, "teachers", "read")
    _authorize(user, "students", "read")
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM app_state WHERE id = 1"
        ).fetchone()
        state = json.loads(row["state_json"]) if row else {}
        school = state.get("school", {})
        school_class = next(
            (
                item
                for item in state.get("classes", [])
                if item.get("id") == payload.class_id
            ),
            None,
        )
        if not school_class or not _can_access_record(
            user, "classes", school_class, state
        ):
            raise HTTPException(
                status_code=404, detail="Classe introuvable ou hors périmètre."
            )
        teacher = next(
            (
                item
                for item in state.get("teachers", [])
                if item.get("id") == school_class.get("teacherId")
                and item.get("status", "Actif") == "Actif"
            ),
            None,
        )
        if not teacher:
            raise HTTPException(
                status_code=422,
                detail="Aucun professeur actif n'est affecté à cette classe.",
            )
        phone = _phone_for_whatsapp(
            str(teacher.get("phone") or ""),
            str(school.get("countryCallingCode") or "223"),
        )
        roster = []
        for student in state.get("students", []):
            enrollment = _student_year_enrollment(student, payload.school_year)
            if (
                (
                    student.get("active", True)
                    or payload.school_year != school.get("year")
                )
                and enrollment
                and enrollment.get("className") == school_class.get("name")
            ):
                roster.append(student)
        roster.sort(
            key=lambda item: (item.get("lastName", ""), item.get("firstName", ""))
        )
        if not roster:
            raise HTTPException(
                status_code=422,
                detail="Aucun élève actif dans cette classe pour cette année.",
            )
        lines = [
            f"Liste des élèves · {school.get('name', 'Établissement')}",
            f"Classe : {school_class['name']} · Année scolaire : {payload.school_year}",
            f"Professeur : {teacher.get('firstName', '')} {teacher.get('lastName', '')}".strip(),
            "",
        ]
        lines.extend(
            f"{index}. {item.get('firstName', '')} {item.get('lastName', '')}".strip()
            for index, item in enumerate(roster, 1)
        )
        message = "\n".join(lines)
        if len(message) > 3500:
            raise HTTPException(
                status_code=422,
                detail="La liste dépasse la taille pratique d'un message WhatsApp; imprimez-la ou partagez-la en plusieurs parties.",
            )
        mode = school.get("whatsappMode", "test")
        if mode not in {"test", "link"}:
            mode = "test"
        url = f"https://wa.me/{phone}?text={quote(message)}" if mode == "link" else None
        _audit(
            connection,
            user,
            "préparation WhatsApp · test"
            if mode == "test"
            else "préparation lien WhatsApp",
            "teachers",
            f"Liste de {len(roster)} élèves préparée pour {school_class['name']} · {payload.school_year}; aucun envoi effectué.",
            school_class["id"],
        )
    return {
        "mode": mode,
        "url": url,
        "message": message,
        "student_count": len(roster),
        "class_name": school_class["name"],
        "school_year": payload.school_year,
        "teacher_name": f"{teacher.get('firstName', '')} {teacher.get('lastName', '')}".strip(),
    }


@app.get("/api/roles")
def roles(request: Request):
    _require_user(request)
    return {
        "roles": [
            {"id": role, "name": name, "permissions": _permissions(role)}
            for role, name in (
                ("proviseur", "Proviseur"),
                ("admin", "Administrateur"),
                ("censeur", "Censeur"),
                ("comptable", "Comptable"),
                ("econome", "Économe"),
                ("surveillant", "Surveillant"),
                ("directeur_primaire", "Directeur Crèche & Primaire"),
                ("directeur_secondaire", "Directeur Secondaire & Lycée"),
                ("enseignant", "Enseignant"),
                ("accueil", "Agent d'accueil"),
                ("user", "Rôle personnalisé"),
            )
        ]
    }


@app.get("/api/accounts")
def list_accounts(request: Request):
    user = _require_user(request)
    _authorize(user, "accounts", "read")
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM users ORDER BY username COLLATE NOCASE"
        ).fetchall()
    return {"users": [_row_user(row) for row in rows]}


@app.post("/api/accounts", status_code=201)
def create_account(payload: AccountPayload, request: Request):
    user = _require_user(request)
    _authorize(user, "accounts", "create")
    if payload.role not in {
        "admin",
        "censeur",
        "comptable",
        "econome",
        "surveillant",
        "directeur_primaire",
        "directeur_secondaire",
        "enseignant",
        "accueil",
        "user",
    }:
        raise HTTPException(
            status_code=403,
            detail="Seul un Proviseur peut attribuer le rôle Proviseur.",
        )
    if user["role"] != "proviseur" and payload.role != "user":
        raise HTTPException(
            status_code=403, detail="Seul un Proviseur peut attribuer ce rôle."
        )
    username = payload.username.strip()
    if not username:
        raise HTTPException(
            status_code=422, detail="Le nom d'utilisateur est obligatoire."
        )
    if payload.role == "enseignant" and not payload.class_scope:
        raise HTTPException(
            status_code=422, detail="Attribuez au moins une classe à l'enseignant."
        )
    cycle_scope = payload.cycle_scope
    if payload.role in DIRECTOR_CYCLE_SCOPES:
        expected_scope = DIRECTOR_CYCLE_SCOPES[payload.role]
        if cycle_scope is not None and cycle_scope != expected_scope:
            raise HTTPException(
                status_code=422,
                detail="Le périmètre de cycle ne correspond pas au rôle de direction.",
            )
        cycle_scope = expected_scope
    elif cycle_scope is not None and (
        not isinstance(cycle_scope, list)
        or any(cycle not in ACADEMIC_CYCLES_NAMES for cycle in cycle_scope)
    ):
        raise HTTPException(
            status_code=422, detail="Le périmètre contient un cycle inconnu."
        )
    salt, digest = _hash_password(payload.password)
    user_id = str(uuid.uuid4())
    try:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO users (id, username, password_hash, role, active, permissions_json, class_scope_json, cycle_scope_json, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (
                    user_id,
                    username,
                    f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}",
                    payload.role,
                    json.dumps(_permissions(payload.role)),
                    json.dumps(payload.class_scope) if payload.class_scope else None,
                    json.dumps(cycle_scope) if cycle_scope else None,
                    _timestamp(),
                ),
            )
            _audit(
                connection, user, "create", "accounts", "Création d'un compte.", user_id
            )
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409, detail="Ce nom d'utilisateur est déjà utilisé."
        ) from None
    return _row_user(row)


@app.patch("/api/accounts/{user_id}")
def update_account(user_id: str, payload: dict[str, Any], request: Request):
    user = _require_user(request)
    _authorize(user, "accounts", "update")
    allowed = {
        "username",
        "password",
        "role",
        "permissions",
        "class_scope",
        "cycle_scope",
    }
    if payload.keys() - allowed:
        raise HTTPException(status_code=422, detail="Champ de compte non modifiable.")
    if "active" in payload:
        raise HTTPException(
            status_code=403,
            detail="Utilisez l'action d'activation réservée au Proviseur.",
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Compte introuvable.")
        target = _row_user(row)
        if target["role"] == "proviseur" and user["role"] != "proviseur":
            raise HTTPException(
                status_code=403,
                detail="Seul un Proviseur peut modifier un compte Proviseur.",
            )
        if "role" in payload:
            if user["role"] != "proviseur":
                raise HTTPException(
                    status_code=403, detail="Seul un Proviseur peut changer un rôle."
                )
            if payload["role"] not in {
                "proviseur",
                "admin",
                "censeur",
                "comptable",
                "econome",
                "surveillant",
                "directeur_primaire",
                "directeur_secondaire",
                "enseignant",
                "accueil",
                "user",
            }:
                raise HTTPException(status_code=422, detail="Rôle inconnu.")
            if user_id == user["id"] and payload["role"] != "proviseur":
                raise HTTPException(
                    status_code=403,
                    detail="Vous ne pouvez pas retirer votre propre rôle de Proviseur.",
                )
        if "permissions" in payload:
            if user["role"] not in {"proviseur", "admin"}:
                raise HTTPException(
                    status_code=403,
                    detail="Seul un administrateur peut attribuer des privilèges.",
                )
            if not isinstance(payload["permissions"], dict):
                raise HTTPException(
                    status_code=422, detail="La matrice des permissions est invalide."
                )
            if user["role"] == "admin":
                for module, actions in payload["permissions"].items():
                    if not isinstance(actions, dict):
                        raise HTTPException(
                            status_code=422,
                            detail="La matrice contient une permission invalide.",
                        )
                    for action, allowed in actions.items():
                        if allowed and not user["permissions"].get(module, {}).get(
                            action, False
                        ):
                            raise HTTPException(
                                status_code=403,
                                detail="Un administrateur ne peut pas accorder un privilège qu'il ne possède pas.",
                            )
            for module, actions in payload["permissions"].items():
                if module not in MODULES or not isinstance(actions, dict):
                    raise HTTPException(
                        status_code=422,
                        detail="La matrice contient un module invalide.",
                    )
                for action, allowed in actions.items():
                    if action not in (*CRUD, "activate") or not isinstance(
                        allowed, bool
                    ):
                        raise HTTPException(
                            status_code=422,
                            detail="La matrice contient une permission invalide.",
                        )
        if user_id == user["id"] and user["role"] != "proviseur":
            raise HTTPException(
                status_code=403, detail="Vous ne pouvez pas modifier vos propres accès."
            )
        username = str(payload.get("username", target["username"])).strip()
        role = payload.get("role", target["role"])
        permissions = payload.get(
            "permissions",
            _permissions(role) if role != target["role"] else target["permissions"],
        )
        scope = payload.get("class_scope", target["class_scope"])
        cycle_scope = payload.get(
            "cycle_scope",
            DIRECTOR_CYCLE_SCOPES.get(role, target["cycle_scope"]),
        )
        if role == "enseignant" and not scope:
            raise HTTPException(
                status_code=422, detail="Attribuez au moins une classe à l'enseignant."
            )
        if (
            "class_scope" in payload
            and scope is not None
            and not isinstance(scope, list)
        ):
            raise HTTPException(
                status_code=422, detail="Le périmètre doit être une liste de classes."
            )
        if role in DIRECTOR_CYCLE_SCOPES:
            if cycle_scope != DIRECTOR_CYCLE_SCOPES[role]:
                raise HTTPException(
                    status_code=422,
                    detail="Le périmètre de cycle ne correspond pas au rôle de direction.",
                )
        elif cycle_scope is not None and (
            not isinstance(cycle_scope, list)
            or any(cycle not in ACADEMIC_CYCLES_NAMES for cycle in cycle_scope)
        ):
            raise HTTPException(
                status_code=422, detail="Le périmètre contient un cycle inconnu."
            )
        if not username:
            raise HTTPException(
                status_code=422, detail="Le nom d'utilisateur est obligatoire."
            )
        password = payload.get("password")
        password_hash = row["password_hash"]
        if password is not None:
            if not isinstance(password, str) or len(password) < 8:
                raise HTTPException(
                    status_code=422,
                    detail="Le mot de passe doit contenir au moins 8 caractères.",
                )
            salt, digest = _hash_password(password)
            password_hash = f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}"
        try:
            connection.execute(
                "UPDATE users SET username = ?, role = ?, permissions_json = ?, class_scope_json = ?, cycle_scope_json = ?, password_hash = ? WHERE id = ?",
                (
                    username,
                    role,
                    json.dumps(permissions),
                    json.dumps(scope) if scope else None,
                    json.dumps(cycle_scope) if cycle_scope else None,
                    password_hash,
                    user_id,
                ),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=409, detail="Ce nom d'utilisateur est déjà utilisé."
            ) from None
        _audit(
            connection,
            user,
            "update",
            "accounts",
            "Mise à jour des accès d'un compte.",
            user_id,
        )
        updated = connection.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return _row_user(updated)


@app.patch("/api/accounts/{user_id}/activation")
def set_account_active(user_id: str, payload: dict[str, bool], request: Request):
    user = _require_user(request)
    if not user["permissions"].get("accounts", {}).get("activate", False):
        raise HTTPException(
            status_code=403,
            detail="Votre compte ne peut pas activer ou désactiver des utilisateurs.",
        )
    if user_id == user["id"]:
        raise HTTPException(
            status_code=422, detail="Vous ne pouvez pas désactiver votre propre compte."
        )
    if set(payload) != {"active"}:
        raise HTTPException(
            status_code=422, detail="Le statut du compte est obligatoire."
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT username, role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Compte introuvable.")
        if row["role"] == "proviseur" and user["role"] != "proviseur":
            raise HTTPException(
                status_code=403,
                detail="Seul un Proviseur peut modifier un autre compte Proviseur.",
            )
        connection.execute(
            "UPDATE users SET active = ? WHERE id = ?",
            (int(payload["active"]), user_id),
        )
        if not payload["active"]:
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        _audit(
            connection,
            user,
            "activation" if payload["active"] else "désactivation",
            "accounts",
            "Modification du statut d'un compte.",
            user_id,
        )
    return {"ok": True, "active": payload["active"]}


@app.delete("/api/accounts/{user_id}", status_code=204)
def delete_account(user_id: str, request: Request):
    user = _require_user(request)
    _authorize(user, "accounts", "delete")
    if user_id == user["id"]:
        raise HTTPException(
            status_code=422, detail="Vous ne pouvez pas supprimer votre propre compte."
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Compte introuvable.")
        if row["role"] == "proviseur":
            raise HTTPException(
                status_code=403,
                detail="Les comptes Proviseur ne peuvent pas être supprimés.",
            )
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        _audit(
            connection, user, "delete", "accounts", "Suppression d'un compte.", user_id
        )
    return Response(status_code=204)


@app.put("/api/credentials")
def update_credentials(payload: CredentialsPayload, request: Request):
    user = _require_user(request)
    if user["role"] not in {"proviseur", "admin"}:
        raise HTTPException(
            status_code=403,
            detail="Seul un administrateur ou le Proviseur peut modifier ces identifiants.",
        )
    if payload.new_password is not None and len(payload.new_password) < 8:
        raise HTTPException(
            status_code=422,
            detail="Le mot de passe doit contenir au moins 8 caractères.",
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?", (user["id"],)
        ).fetchone()
        if not row or not _password_matches(
            payload.current_password, row["password_hash"]
        ):
            raise HTTPException(
                status_code=403, detail="Le mot de passe actuel est incorrect."
            )
        updates = [payload.username.strip()]
        sql = "UPDATE users SET username = ?"
        if payload.new_password:
            salt, digest = _hash_password(payload.new_password)
            sql += ", password_hash = ?"
            updates.append(f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}")
        sql += " WHERE id = ?"
        updates.append(user["id"])
        try:
            connection.execute(sql, updates)
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=409, detail="Ce nom d'utilisateur est déjà utilisé."
            ) from None
        _audit(
            connection,
            user,
            "update_credentials",
            "accounts",
            "Mise à jour des identifiants administrateur.",
            user["id"],
        )
        updated = connection.execute(
            "SELECT * FROM users WHERE id = ?", (user["id"],)
        ).fetchone()
    return _row_user(updated)


@app.get("/api/audit")
def audit_log(request: Request, limit: int = 200):
    user = _require_user(request)
    _authorize(user, "audit", "read")
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, user_id, username, action, module, record_id, summary, created_at "
            "FROM audit_log ORDER BY id DESC LIMIT ?",
            (min(max(limit, 1), 500),),
        ).fetchall()
    return {"entries": [dict(row) for row in rows]}


@app.post("/api/reset-demo")
def reset_demo(request: Request):
    user = _require_user(request)
    if user["role"] != "proviseur":
        raise HTTPException(
            status_code=403,
            detail="Seul le Proviseur peut réinitialiser les données de démonstration.",
        )
    with _connect() as connection:
        connection.execute("DELETE FROM app_state WHERE id = 1")
        _audit(
            connection,
            user,
            "reset",
            "settings",
            "Réinitialisation des données de démonstration.",
        )
    return {"ok": True}


@app.get("/api/audit/{audit_id}")
def get_audit_entry(audit_id: int, request: Request):
    user = _require_user(request)
    _authorize(user, "audit", "read")
    with _connect() as connection:
        row = connection.execute(
            "SELECT id, user_id, username, action, module, record_id, summary, created_at FROM audit_log WHERE id = ?",
            (audit_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entrée d'audit introuvable.")
    return dict(row)


def main() -> None:
    import uvicorn

    host = "127.0.0.1"
    port = int(os.environ.get("SCHOOL_MANAGEMENT_PORT", "8765"))
    if getattr(sys, "frozen", False):
        threading.Timer(1.2, webbrowser.open, args=(f"http://{host}:{port}",)).start()
    uvicorn.run(
        app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
