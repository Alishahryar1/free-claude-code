import struct

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHOOL_MANAGEMENT_DB", str(tmp_path / "school.sqlite3"))
    from free_claude_code.school_management.app import app

    with TestClient(app) as test_client:
        yield test_client


def sign_in(
    client: TestClient, username: str = "Moussa", password: str = "79408093Ms@"
):
    response = client.post(
        "/api/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["user"]


def test_initial_master_is_hashed_and_has_super_admin_permissions(client):
    user = sign_in(client)

    assert user["username"] == "Moussa"
    assert user["role"] == "proviseur"
    assert user["permissions"]["accounts"]["activate"] is True
    assert "password" not in user
    assert "password_hash" not in user
    assert client.get("/api/me").json()["id"] == user["id"]


def test_static_application_assets_are_served(client):
    page = client.get("/")
    script = client.get("/app.js")
    icon = client.get("/favicon.ico")
    styles = client.get("/styles.css")

    assert page.status_code == 200
    assert "/favicon.ico" in page.text
    assert 'id="loginForm"' in page.text
    assert script.status_code == 200
    assert "/api/bootstrap" in script.text
    assert icon.status_code == 200
    assert icon.headers["content-type"] == "image/vnd.microsoft.icon"
    reserved, icon_type, count = struct.unpack_from("<HHH", icon.content)
    assert (reserved, icon_type, count) == (0, 1, 7)
    icon_sizes = [
        struct.unpack_from("<B", icon.content, 6 + index * 16)[0] or 256
        for index in range(count)
    ]
    assert icon_sizes == [16, 24, 32, 48, 64, 128, 256]
    for index in range(count):
        image_offset = struct.unpack_from("<I", icon.content, 6 + index * 16 + 12)[0]
        assert icon.content[image_offset : image_offset + 8] == b"\x89PNG\r\n\x1a\n"
    assert styles.status_code == 200
    assert ".main-nav { display: flex; min-height: 0; flex: 1 1 0;" in styles.text
    assert "overflow-y: auto; overscroll-behavior: contain;" in styles.text
    assert ".sidebar-bottom { flex: 0 0 auto; margin-top: auto; }" in styles.text


def test_legacy_terminal_series_aliases_are_normalized():
    from free_claude_code.school_management.app import _class_level

    assert _class_level({"academicLevel": "12e TSExp"}) == "12e TSEExp"
    assert _class_level({"academicLevel": "12e TEXP"}) == "12e TSEExp"


def test_role_seeds_match_confirmed_matrix(client):
    sign_in(client)
    roles = {
        role["id"]: role["permissions"]
        for role in client.get("/api/roles").json()["roles"]
    }

    assert all(
        roles["proviseur"][module][action]
        for module in roles["proviseur"]
        for action in ("create", "read", "update", "delete")
    )
    assert roles["admin"]["audit"]["delete"] is False
    assert roles["admin"]["payments"]["delete"] is False
    assert roles["admin"]["accounts"]["activate"] is True
    assert roles["censeur"]["teachers"] == {
        "create": True,
        "read": True,
        "update": True,
        "delete": True,
    }
    assert roles["censeur"]["grades"] == {
        "create": True,
        "read": True,
        "update": True,
        "delete": True,
    }
    assert roles["censeur"]["discipline"] == {
        "create": True,
        "read": True,
        "update": True,
        "delete": True,
    }
    assert roles["comptable"]["payments"]["create"] is True
    assert roles["comptable"]["students"]["delete"] is False
    assert roles["enseignant"]["grades"]["update"] is True
    assert roles["accueil"]["students"]["create"] is True
    assert roles["accueil"]["students"]["delete"] is False
    assert roles["censeur"]["curriculum"]["update"] is True
    assert roles["censeur"]["bulletins"]["update"] is True
    assert roles["econome"]["expenses"]["create"] is True
    assert roles["econome"]["expenses"]["delete"] is False
    assert roles["surveillant"]["attendance"]["create"] is True
    assert roles["surveillant"]["students"]["create"] is False
    assert roles["directeur_primaire"]["students"]["update"] is True
    assert roles["directeur_secondaire"]["classes"]["delete"] is True


def test_class_scope_filters_records_and_rejects_out_of_scope_changes(client):
    sign_in(client)
    state = {
        "school": {"name": "École exemple"},
        "students": [
            {"id": "s-a", "firstName": "Awa", "lastName": "A", "className": "Classe A"},
            {
                "id": "s-b",
                "firstName": "Boubacar",
                "lastName": "B",
                "className": "Classe B",
            },
        ],
        "classes": [
            {"id": "c-a", "name": "Classe A"},
            {"id": "c-b", "name": "Classe B"},
        ],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    response = client.post(
        "/api/accounts",
        json={
            "username": "EnseignantA",
            "password": "Motdepasse-123",
            "role": "enseignant",
            "class_scope": ["Classe A"],
        },
    )
    assert response.status_code == 201
    client.post("/api/logout")
    sign_in(client, "EnseignantA", "Motdepasse-123")

    visible = client.get("/api/state").json()["state"]
    assert [student["id"] for student in visible["students"]] == ["s-a"]
    assert [school_class["id"] for school_class in visible["classes"]] == ["c-a"]
    changed = {
        **visible,
        "students": [
            {
                "id": "s-b",
                "firstName": "Boubacar",
                "lastName": "B",
                "className": "Classe B",
            }
        ],
    }
    assert client.put("/api/state", json={"state": changed}).status_code == 403


def test_director_cycle_scope_filters_data_and_rejects_cross_cycle_changes(client):
    sign_in(client)
    state = {
        "school": {"name": "École exemple"},
        "students": [
            {
                "id": "student-primary",
                "firstName": "Awa",
                "lastName": "A",
                "className": "Primaire 1",
            },
            {
                "id": "student-secondary",
                "firstName": "Moussa",
                "lastName": "B",
                "className": "Secondaire 7",
            },
        ],
        "classes": [
            {"id": "class-primary", "name": "Primaire 1", "academicLevel": "1re"},
            {"id": "class-secondary", "name": "Secondaire 7", "academicLevel": "7e"},
        ],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
        "attendance": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    invalid_scope = client.post(
        "/api/accounts",
        json={
            "username": "DirectionInvalide",
            "password": "Motdepasse-123",
            "role": "directeur_primaire",
            "cycle_scope": ["Secondaire", "Lycée"],
        },
    )
    assert invalid_scope.status_code == 422
    director = client.post(
        "/api/accounts",
        json={
            "username": "DirectionPrimaire",
            "password": "Motdepasse-123",
            "role": "directeur_primaire",
        },
    )
    assert director.status_code == 201
    assert director.json()["cycle_scope"] == ["Crèche", "Primaire"]
    client.post("/api/logout")
    sign_in(client, "DirectionPrimaire", "Motdepasse-123")

    visible = client.get("/api/state").json()["state"]
    assert [item["id"] for item in visible["students"]] == ["student-primary"]
    assert [item["id"] for item in visible["classes"]] == ["class-primary"]
    visible["students"].append(
        {
            "id": "student-intrusion",
            "firstName": "Intrusion",
            "lastName": "X",
            "className": "Secondaire 7",
        }
    )
    assert client.put("/api/state", json={"state": visible}).status_code == 403


def test_censeur_can_edit_curriculum_without_general_settings_access(client):
    sign_in(client)
    state = {
        "school": {
            "name": "École exemple",
            "curriculum": {"1re": [{"name": "Mathématiques", "coefficient": 1}]},
        },
        "students": [],
        "classes": [],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
        "attendance": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    account = client.post(
        "/api/accounts",
        json={"username": "Censeur", "password": "Motdepasse-123", "role": "censeur"},
    )
    assert account.status_code == 201
    client.post("/api/logout")
    sign_in(client, "Censeur", "Motdepasse-123")

    visible = client.get("/api/state").json()["state"]
    visible["school"]["curriculum"]["1re"][0]["coefficient"] = 2
    assert client.put("/api/state", json={"state": visible}).status_code == 200
    updated = client.get("/api/state").json()["state"]
    updated["school"]["name"] = "Nom non autorisé"
    assert client.put("/api/state", json={"state": updated}).status_code == 403


def test_attendance_requires_valid_student_date_and_status(client):
    sign_in(client)
    state = {
        "school": {"name": "École exemple"},
        "students": [
            {
                "id": "student-1",
                "firstName": "Awa",
                "lastName": "A",
                "className": "Classe A",
            }
        ],
        "classes": [{"id": "class-1", "name": "Classe A", "academicLevel": "1re"}],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
        "attendance": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    state["attendance"] = [
        {
            "id": "attendance-1",
            "studentId": "student-1",
            "date": "2026-09-01",
            "status": "Présent",
            "observation": "",
        }
    ]
    assert client.put("/api/state", json={"state": state}).status_code == 200
    saved_attendance = client.get("/api/state").json()["state"]["attendance"][0]
    assert saved_attendance["recordedBy"] == "Moussa"
    assert saved_attendance["className"] == "Classe A"
    state = client.get("/api/state").json()["state"]

    reassigned = {
        **state,
        "attendance": [{**state["attendance"][0], "studentId": "student-other"}],
    }
    assert client.put("/api/state", json={"state": reassigned}).status_code == 422
    invalid = {
        **state,
        "attendance": [{**state["attendance"][0], "date": "2026-99-40"}],
    }
    assert client.put("/api/state", json={"state": invalid}).status_code == 422
    duplicate = {
        **state,
        "attendance": [
            *state["attendance"],
            {
                "id": "attendance-2",
                "studentId": "student-1",
                "date": "2026-09-01",
                "status": "Absent",
            },
        ],
    }
    assert client.put("/api/state", json={"state": duplicate}).status_code == 409


def test_local_backup_excludes_accounts_and_restores_school_data(client):
    sign_in(client)
    state = {
        "school": {"name": "École avant sauvegarde"},
        "students": [
            {
                "id": "student-photo",
                "firstName": "Awa",
                "lastName": "Traoré",
                "photo": "data:image/png;base64,iVBORw0KGgo=",
            }
        ],
        "classes": [],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
        "attendance": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    exported = client.get("/api/backup")
    assert exported.status_code == 200
    backup = exported.json()
    assert backup["format"] == "school-management-backup-v1"
    assert "users" not in backup["state"]
    assert "password_hash" not in exported.text
    assert (
        backup["state"]["students"][0]["photo"] == "data:image/png;base64,iVBORw0KGgo="
    )

    backup["state"]["school"]["name"] = "École restaurée"
    assert client.post("/api/backup", json=backup).status_code == 200
    restored_state = client.get("/api/state").json()["state"]
    assert restored_state["school"]["name"] == "École restaurée"
    assert (
        restored_state["students"][0]["photo"] == "data:image/png;base64,iVBORw0KGgo="
    )
    assert client.get("/api/me").json()["username"] == "Moussa"
    invalid = {**backup, "state": {"school": {"name": "Invalide"}, "students": []}}
    assert client.post("/api/backup", json=invalid).status_code == 422


def test_python_automation_creates_rotating_local_backups(
    client, tmp_path, monkeypatch
):
    sign_in(client)
    monkeypatch.setenv("SCHOOL_MANAGEMENT_BACKUP_RETENTION", "1")
    state = {
        "school": {"name": "École automatisée"},
        "students": [],
        "classes": [],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
        "attendance": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200

    first = client.post("/api/automation/backup")
    assert first.status_code == 200
    assert first.json()["backup"]["filename"].endswith(".json")
    backup_dir = tmp_path / "school-management-backups"
    first_backup = backup_dir / first.json()["backup"]["filename"]
    saved = first_backup.read_text(encoding="utf-8")
    assert "École automatisée" in saved
    assert "password_hash" not in saved
    assert "79408093Ms@" not in saved

    state["school"]["name"] = "École mise à jour"
    assert client.put("/api/state", json={"state": state}).status_code == 200
    second = client.post("/api/automation/backup")
    assert second.status_code == 200
    assert first_backup.exists() is False
    assert len(list(backup_dir.glob("school-management-*.json"))) == 1
    status = client.get("/api/automation")
    assert status.status_code == 200
    assert status.json()["backup_count"] == 1
    assert status.json()["retention"] == 1


def test_admin_cannot_delete_critical_history_or_activate_accounts(client):
    sign_in(client)
    state = {
        "school": {"name": "École exemple"},
        "students": [],
        "classes": [],
        "teachers": [],
        "payments": [{"id": "payment-1", "amount": 500}],
        "expenses": [],
        "grades": [],
        "assessments": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    created = client.post(
        "/api/accounts",
        json={"username": "Admin", "password": "Motdepasse-123", "role": "admin"},
    )
    assert created.status_code == 201
    admin_id = created.json()["id"]
    target = client.post(
        "/api/accounts",
        json={"username": "Accueil", "password": "Motdepasse-456", "role": "accueil"},
    )
    assert target.status_code == 201
    target_id = target.json()["id"]
    client.post("/api/logout")
    sign_in(client, "Admin", "Motdepasse-123")

    account = client.get("/api/accounts").json()["users"]
    reception = next(item for item in account if item["username"] == "Accueil")
    permissions = {key: dict(value) for key, value in reception["permissions"].items()}
    permissions["audit"]["delete"] = True
    assert (
        client.patch(
            f"/api/accounts/{target_id}", json={"permissions": permissions}
        ).status_code
        == 403
    )
    activation = client.patch(
        f"/api/accounts/{target_id}/activation", json={"active": False}
    )
    assert activation.status_code == 200
    current_state = client.get("/api/state").json()["state"]
    current_state["payments"] = []
    assert client.put("/api/state", json={"state": current_state}).status_code == 403
    assert admin_id != target_id


def test_master_credentials_can_be_changed_without_exposing_password(client):
    sign_in(client)
    response = client.put(
        "/api/credentials",
        json={
            "current_password": "79408093Ms@",
            "username": "Direction",
            "new_password": "Nouveau-mot-de-passe-123",
        },
    )

    assert response.status_code == 200
    assert "password_hash" not in response.json()
    assert client.post("/api/logout").status_code == 200
    assert (
        client.post(
            "/api/login",
            json={"username": "Moussa", "password": "79408093Ms@"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/login",
            json={"username": "Direction", "password": "Nouveau-mot-de-passe-123"},
        ).status_code
        == 200
    )


def test_annual_closure_calculates_weighted_average_and_preserves_enrollment_history(
    client,
):
    sign_in(client)
    year = "2026 — 2027"
    state = {
        "school": {
            "name": "École exemple",
            "year": year,
            "promotionThreshold": 10,
            "academicClosures": [],
        },
        "students": [
            {
                "id": "s-pass",
                "firstName": "Awa",
                "lastName": "Passage",
                "className": "5e A",
                "active": True,
                "enrollmentYear": year,
                "enrollmentHistory": [
                    {"schoolYear": year, "className": "5e A", "decision": "inscrit"}
                ],
            },
            {
                "id": "s-repeat",
                "firstName": "Boubacar",
                "lastName": "Redoublement",
                "className": "5e A",
                "active": True,
                "enrollmentYear": year,
                "enrollmentHistory": [
                    {"schoolYear": year, "className": "5e A", "decision": "inscrit"}
                ],
            },
            {
                "id": "s-terminal",
                "firstName": "Mariam",
                "lastName": "Diplôme",
                "className": "Terminale TLL",
                "active": True,
                "enrollmentYear": year,
                "enrollmentHistory": [
                    {
                        "schoolYear": year,
                        "className": "Terminale TLL",
                        "decision": "inscrit",
                    }
                ],
            },
        ],
        "classes": [
            {"id": "c-5", "name": "5e A", "academicLevel": "5e", "active": True},
            {"id": "c-6", "name": "6e A", "academicLevel": "6e", "active": True},
            {
                "id": "c-tll",
                "name": "Terminale TLL",
                "academicLevel": "12e TLL",
                "active": True,
            },
        ],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "assessments": [],
        "grades": [
            {
                "id": "g-pass-a",
                "studentId": "s-pass",
                "score": 16,
                "coefficient": 2,
                "schoolYear": year,
            },
            {
                "id": "g-pass-b",
                "studentId": "s-pass",
                "score": 10,
                "coefficient": 1,
                "schoolYear": year,
            },
            {
                "id": "g-repeat",
                "studentId": "s-repeat",
                "score": 8,
                "coefficient": 1,
                "schoolYear": year,
            },
            {
                "id": "g-terminal",
                "studentId": "s-terminal",
                "score": 15,
                "coefficient": 1,
                "schoolYear": year,
            },
        ],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200

    invalid = client.post(
        "/api/academic/close-year",
        json={
            "school_year": year,
            "decisions": [
                {
                    "student_id": "s-pass",
                    "decision": "promote",
                    "target_class_id": "c-6",
                },
                {
                    "student_id": "s-repeat",
                    "decision": "promote",
                    "target_class_id": "c-6",
                },
                {"student_id": "s-terminal", "decision": "graduate"},
            ],
        },
    )
    assert invalid.status_code == 422

    response = client.post(
        "/api/academic/close-year",
        json={
            "school_year": year,
            "decisions": [
                {
                    "student_id": "s-pass",
                    "decision": "promote",
                    "target_class_id": "c-6",
                },
                {"student_id": "s-repeat", "decision": "repeat"},
                {"student_id": "s-terminal", "decision": "graduate"},
            ],
        },
    )

    assert response.status_code == 200
    current = client.get("/api/state").json()["state"]
    promoted = next(item for item in current["students"] if item["id"] == "s-pass")
    repeated = next(item for item in current["students"] if item["id"] == "s-repeat")
    graduate = next(item for item in current["students"] if item["id"] == "s-terminal")
    assert current["school"]["year"] == "2027 — 2028"
    assert promoted["className"] == "6e A"
    assert promoted["enrollmentHistory"][0]["annualAverage"] == 14
    assert promoted["enrollmentHistory"][0]["className"] == "5e A"
    assert repeated["className"] == "5e A"
    assert repeated["enrollmentHistory"][-1]["schoolYear"] == "2027 — 2028"
    assert graduate["active"] is False
    assert graduate["graduationYear"] == year
    assert (
        client.post(
            "/api/academic/close-year",
            json={"school_year": year, "decisions": []},
        ).status_code
        == 409
    )


def test_whatsapp_roster_uses_validated_contact_and_explicit_test_or_link_mode(client):
    sign_in(client)
    year = "2026 — 2027"
    state = {
        "school": {
            "name": "École exemple",
            "year": year,
            "whatsappMode": "test",
            "countryCallingCode": "223",
        },
        "students": [
            {
                "id": "student-1",
                "firstName": "Awa",
                "lastName": "Traoré",
                "className": "Classe A",
                "active": True,
                "enrollmentYear": year,
                "enrollmentHistory": [{"schoolYear": year, "className": "Classe A"}],
            }
        ],
        "classes": [
            {
                "id": "class-a",
                "name": "Classe A",
                "teacherId": "teacher-a",
                "active": True,
            }
        ],
        "teachers": [
            {
                "id": "teacher-a",
                "firstName": "Moussa",
                "lastName": "Diop",
                "phone": "70001122",
                "status": "Actif",
            }
        ],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    payload = {"class_id": "class-a", "school_year": year}
    test_response = client.post("/api/whatsapp/roster", json=payload)
    assert test_response.status_code == 200
    assert test_response.json()["mode"] == "test"
    assert test_response.json()["url"] is None
    assert "Awa Traoré" in test_response.json()["message"]

    state["school"]["whatsappMode"] = "link"
    assert client.put("/api/state", json={"state": state}).status_code == 200
    link_response = client.post("/api/whatsapp/roster", json=payload)
    assert link_response.status_code == 200
    assert link_response.json()["url"].startswith("https://wa.me/22370001122?text=")
    assert "aucun envoi effectué" in client.get("/api/audit").text.lower()

    state["teachers"][0]["phone"] = "contact-invalide"
    assert client.put("/api/state", json={"state": state}).status_code == 200
    assert client.post("/api/whatsapp/roster", json=payload).status_code == 422


def test_censeur_can_assign_teacher_to_class_without_class_management_permission(
    client,
):
    sign_in(client)
    state = {
        "school": {"name": "École exemple", "year": "2026 — 2027"},
        "students": [],
        "classes": [{"id": "class-a", "name": "Classe A", "active": True}],
        "teachers": [
            {
                "id": "teacher-a",
                "firstName": "Awa",
                "lastName": "Diop",
                "phone": "70001122",
                "status": "Actif",
            }
        ],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    state = client.get("/api/state").json()["state"]
    account = client.post(
        "/api/accounts",
        json={
            "username": "CenseurTest",
            "password": "Motdepasse-123",
            "role": "censeur",
        },
    )
    assert account.status_code == 201
    client.post("/api/logout")
    sign_in(client, "CenseurTest", "Motdepasse-123")
    current = client.get("/api/state").json()["state"]
    current["classes"][0]["teacherId"] = "teacher-a"
    current["classes"][0]["teacher"] = "Awa Diop"

    response = client.put("/api/state", json={"state": current})

    assert response.status_code == 200
    assert (
        client.get("/api/state").json()["state"]["classes"][0]["teacherId"]
        == "teacher-a"
    )


def test_disciplinary_deduction_requires_separate_validation_and_is_audited(client):
    sign_in(client)
    state = {
        "school": {
            "name": "École exemple",
            "year": "2026 — 2027",
            "conductMaximum": 18,
        },
        "students": [
            {
                "id": "student-conduct",
                "firstName": "Awa",
                "lastName": "Traoré",
                "className": "Classe A",
                "active": True,
                "enrollmentYear": "2026 — 2027",
                "enrollmentHistory": [
                    {"schoolYear": "2026 — 2027", "className": "Classe A"}
                ],
            }
        ],
        "classes": [{"id": "class-a", "name": "Classe A", "active": True}],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200
    state["incidents"] = [
        {
            "id": "incident-1",
            "studentId": "student-conduct",
            "date": "2027-03-14",
            "reason": "Incident saisi par le personnel",
            "pointsRemoved": 4,
            "observation": "Observation de suivi",
            "validated": False,
            "author": "Nom falsifié par le client",
        }
    ]
    assert client.put("/api/state", json={"state": state}).status_code == 200
    incident = client.get("/api/state").json()["state"]["incidents"][0]
    assert incident["validated"] is False
    assert incident["author"] == "Moussa"

    validation = client.post("/api/discipline/incidents/incident-1/validate")

    assert validation.status_code == 200
    assert validation.json()["incident"]["validated"] is True
    assert validation.json()["incident"]["validatedBy"] == "Moussa"
    assert (
        client.post("/api/discipline/incidents/incident-1/validate").status_code == 409
    )
    audit = client.get("/api/audit").json()["entries"]
    assert any(
        entry["module"] == "discipline" and entry["action"] == "validation"
        for entry in audit
    )


def test_invalid_conduct_capital_and_directly_validated_incident_are_rejected(client):
    sign_in(client)
    state = {
        "school": {"name": "École exemple", "conductMaximum": 17},
        "students": [
            {"id": "s-1", "firstName": "Awa", "lastName": "A", "className": "A"}
        ],
        "classes": [{"id": "c-1", "name": "A"}],
        "teachers": [],
        "payments": [],
        "expenses": [],
        "grades": [],
        "assessments": [],
        "incidents": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 422

    state["school"]["conductMaximum"] = 18
    state["incidents"] = [
        {
            "id": "incident-spoof",
            "studentId": "s-1",
            "date": "2027-03-14",
            "reason": "Tentative de contournement",
            "pointsRemoved": 1,
            "validated": True,
        }
    ]
    assert client.put("/api/state", json={"state": state}).status_code == 422


def test_python_analytics_are_authoritative_for_school_totals_and_averages(client):
    sign_in(client)
    year = "2026 — 2027"
    state = {
        "school": {"name": "École exemple", "year": year, "conductMaximum": 18},
        "students": [
            {
                "id": "s-1",
                "firstName": "Awa",
                "lastName": "A",
                "className": "Classe A",
                "enrollmentYear": year,
                "annualFee": 500,
                "active": True,
            }
        ],
        "classes": [{"id": "c-1", "name": "Classe A", "active": True}],
        "payments": [
            {"id": "p-1", "studentId": "s-1", "date": "2027-01-15", "amount": 100},
            {"id": "p-2", "studentId": "s-1", "date": "2027-02-15", "amount": 150},
        ],
        "expenses": [{"id": "e-1", "date": "2027-02-20", "amount": 50}],
        "grades": [
            {
                "id": "g-1",
                "studentId": "s-1",
                "schoolYear": year,
                "term": "1er trimestre",
                "subject": "Mathématiques",
                "score": 10,
                "coefficient": 1,
            },
            {
                "id": "g-2",
                "studentId": "s-1",
                "schoolYear": year,
                "term": "1er trimestre",
                "subject": "Mathématiques",
                "score": 20,
                "coefficient": 3,
            },
        ],
        "assessments": [],
        "incidents": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200

    response = client.get("/api/state").json()
    analytics = response["analytics"]
    assert analytics["dashboard"]["income"] == 250
    assert analytics["dashboard"]["expenses"] == 50
    assert analytics["dashboard"]["cashBalance"] == 200
    assert analytics["studentFinancials"]["s-1"]["outstanding"] == 250
    assert analytics["classSizes"][f"{year}|Classe A"] == 1
    assert analytics["averages"][f"s-1|{year}|annual"]["average"] == 17.5
    assert analytics["averages"][f"s-1|{year}|1er trimestre"]["average"] == 17.5
    assert (
        analytics["subjectAverages"][f"s-1|{year}|annual|Mathématiques"]["average"]
        == 17.5
    )

    period = client.get(
        "/api/analytics",
        params={"from_date": "2027-02-01", "to_date": "2027-02-28"},
    )
    assert period.status_code == 200
    assert period.json()["analytics"]["period"] == {
        "income": 150,
        "expenses": 50,
        "cashBalance": 100,
        "paymentCount": 1,
        "expenseCount": 1,
    }
    assert period.json()["analytics"]["dashboard"]["income"] == 250


def test_python_generates_periodic_financial_pdf_report(client):
    sign_in(client)
    state = {
        "school": {"name": "École des tests", "year": "2026 — 2027"},
        "students": [
            {
                "id": "student-1",
                "firstName": "Awa",
                "lastName": "Traoré",
                "active": True,
            }
        ],
        "classes": [{"id": "class-1", "name": "1re A", "active": True}],
        "teachers": [],
        "payments": [
            {
                "id": "payment-1",
                "studentId": "student-1",
                "date": "2027-01-10",
                "amount": 12500,
                "method": "Espèces",
            }
        ],
        "expenses": [
            {
                "id": "expense-1",
                "date": "2027-01-12",
                "amount": 2500,
                "label": "Achat de craies",
                "category": "Fournitures",
            }
        ],
        "grades": [],
        "assessments": [],
        "incidents": [],
        "attendance": [],
    }
    assert client.put("/api/state", json={"state": state}).status_code == 200

    response = client.get(
        "/api/reports/financial-summary.pdf",
        params={"from_date": "2027-01-01", "to_date": "2027-01-31"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-1.4")
    assert b"RAPPORT FINANCIER" in response.content
    assert b"FCFA" in response.content
    xref_offset = int(
        response.content.split(b"startxref\n", maxsplit=1)[1].splitlines()[0]
    )
    assert response.content[xref_offset : xref_offset + 4] == b"xref"
    assert (
        client.get(
            "/api/reports/financial-summary.pdf",
            params={"from_date": "2027-02-01", "to_date": "2027-01-01"},
        ).status_code
        == 422
    )
