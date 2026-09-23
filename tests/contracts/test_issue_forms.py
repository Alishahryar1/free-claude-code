from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("filename", "label"),
    [
        ("bug-report.yml", "bug"),
        ("feature-request.yml", "enhancement"),
    ],
)
def test_issue_forms_classify_with_labels_not_title_prefixes(
    filename: str,
    label: str,
) -> None:
    form = Path(".github/ISSUE_TEMPLATE", filename).read_text(encoding="utf-8")

    assert f"  - {label}" in form
    assert not any(line.startswith("title:") for line in form.splitlines())
