import warnings

from fastapi.testclient import TestClient

from free_claude_code.core.version import package_version
from tests.api.support import create_test_app


def test_fastapi_and_openapi_report_installed_package_version() -> None:
    app = create_test_app()

    assert app.version == package_version()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Duplicate Operation ID",
            category=UserWarning,
        )
        response = TestClient(app).get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == package_version()
