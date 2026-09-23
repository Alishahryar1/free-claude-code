import json
import re
import shutil
import subprocess
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest

BUG_FORM = Path(".github/ISSUE_TEMPLATE/bug-report.yml")
WORKFLOW = Path(".github/workflows/validate-bug-report-version.yml")


def _workflow_pattern(name: str) -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(rf'const {name} = ("(?:\\.|[^"\\])*");', workflow)
    assert match is not None
    return json.loads(match.group(1))


def _reported_version(value: str) -> str | None:
    matches = re.findall(_workflow_pattern("versionPattern"), value)
    return matches[0] if len(matches) == 1 else None


def _javascript_function(name: str) -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        rf"^            const {name} = .*?^            }};",
        workflow,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    return textwrap.dedent(match.group(0))


def _workflow_script() -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = "          script: |\n"
    _, separator, script = workflow.partition(marker)
    assert separator == marker
    return textwrap.dedent(script)


def _run_javascript(script: str) -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the GitHub workflow contract")
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_bug_form_requests_a_contained_version_or_none() -> None:
    form = BUG_FORM.read_text(encoding="utf-8")

    assert "Run `fcc-server --version`" in form
    assert "include one version" in form
    assert "`number.number.number` format" in form
    assert "enter `None`" in form
    assert 'placeholder: "The version is 1.22.333, or None"' in form
    assert "not installed" not in form


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.0.0", "0.0.0"),
        ("1.22.333", "1.22.333"),
        ("The version is 1.22.333", "1.22.333"),
        ("free-claude-code 4.11.4", "4.11.4"),
        ("v123.45.678", "123.45.678"),
        ("Version 4.6.1.", "4.6.1"),
    ],
)
def test_version_pattern_extracts_a_contained_version(
    value: str,
    expected: str,
) -> None:
    assert _reported_version(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "latest",
        "4.6",
        "4.6.1.2",
        ".4.6.1",
        "4.6.x",
        "none",
        "free-claude-code",
        "free-claude-code 4.6",
        "the version is 4.6.1.2",
        "upgraded from 4.6.1 to 4.11.4",
        "build4.6.1",
        "4.6.1-beta",
        "4.6.1+build",
        "4.6.1.x",
    ],
)
def test_version_pattern_rejects_ambiguous_values(value: str) -> None:
    assert _reported_version(value) is None


def test_none_remains_an_exact_escape_hatch() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'fieldValue === "None"' in workflow
    assert _reported_version("None") is None
    assert _reported_version("The version is None") is None


def test_numeric_version_comparison_uses_all_three_components() -> None:
    function = _javascript_function("isOlderVersion")
    cases = [
        ["4.9.99", "4.10.0"],
        ["4.10.0", "4.10.0"],
        ["4.10.1", "4.10.0"],
        ["9007199254740993.0.0", "9007199254740994.0.0"],
    ]
    script = (
        f"{function}\n"
        f"const cases = {json.dumps(cases)};\n"
        "process.stdout.write(JSON.stringify("
        "cases.map(([reported, latest]) => isOlderVersion(reported, latest))));"
    )

    assert _run_javascript(script) == [True, False, False, True]


def test_field_pattern_extracts_the_issue_form_value() -> None:
    body = """### FCC version

4.6.1

### CLI

Claude Code (fcc-claude)
"""

    match = re.search(_workflow_pattern("fieldPattern"), body, flags=re.MULTILINE)

    assert match is not None
    assert match.group(1) == "4.6.1"


def test_workflow_owns_one_idempotent_triage_state() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "types: [opened, edited]" in workflow
    assert "issues: write" in workflow
    assert "needs-fcc-version" in workflow
    assert "<!-- fcc-version-validator -->" in workflow
    assert "github.rest.issues.createLabel" in workflow
    assert "github.rest.issues.addLabels" in workflow
    assert "github.rest.issues.removeLabel" in workflow
    assert "comments.find" in workflow


def test_workflow_reads_and_compares_the_default_branch_version() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    expected = tomllib.loads(pyproject)["project"]["version"]
    project_pattern = _workflow_pattern("projectVersionPattern")
    function = _javascript_function("projectVersionFromToml")
    scoped_project = (
        "[tool.before]\nversion = \"99.0.0\"\n\n[project]\nversion = '1.2.3'\n"
    )
    commented_project = (
        '[project]\nname = "demo"\nversion = "2.3.4" # current release\n'
    )
    missing_project_version = (
        '[project]\nname = "demo"\n\n[[tool.items]]\nversion = "99.0.0"\n'
    )
    script = (
        f"const projectVersionPattern = {json.dumps(project_pattern)};\n"
        f"{function}\n"
        "process.stdout.write(JSON.stringify(["
        f"projectVersionFromToml({json.dumps(pyproject)}),"
        f"projectVersionFromToml({json.dumps(scoped_project)}),"
        f"projectVersionFromToml({json.dumps(commented_project)}),"
        f"projectVersionFromToml({json.dumps(missing_project_version)})"
        "]));"
    )

    assert _run_javascript(script) == [expected, "1.2.3", "2.3.4", None]
    assert "contents: read" in workflow
    assert "github.rest.repos.getContent" in workflow
    assert 'path: "pyproject.toml"' in workflow
    assert "context.payload.repository.default_branch" in workflow
    assert 'split(".").map((part) => BigInt(part))' in workflow
    assert "isOlderVersion(reportedVersion, latestVersion)" in workflow


def test_outdated_version_comment_is_reconciled_across_edits() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    latest = "17.23.456"
    source = f"return (async () => {{\n{_workflow_script()}\n}})();"
    harness = r"""
const run = new Function("github", "context", __SOURCE__);
const latestVersion = __LATEST__;
const comments = [];
const calls = [];
const liveIssue = { number: 7, labels: [], body: "" };
const record = (name, args) => calls.push({ name, args });
const github = {
  paginate: async () => comments,
  rest: {
    issues: {
      get: async (args) => {
        record("getIssue", args);
        return { data: liveIssue };
      },
      getLabel: async (args) => record("getLabel", args),
      createLabel: async (args) => record("createLabel", args),
      addLabels: async (args) => {
        record("addLabels", args);
        liveIssue.labels.push(...args.labels.map((name) => ({ name })));
      },
      removeLabel: async (args) => {
        record("removeLabel", args);
        liveIssue.labels = liveIssue.labels.filter((label) => label.name !== args.name);
      },
      createComment: async (args) => {
        record("createComment", args);
        comments.push({
          id: 100 + calls.filter((call) => call.name === "createComment").length,
          user: { login: "github-actions[bot]" },
          body: args.body,
        });
      },
      updateComment: async (args) => {
        record("updateComment", args);
        comments.find((comment) => comment.id === args.comment_id).body = args.body;
      },
      deleteComment: async (args) => {
        record("deleteComment", args);
        const index = comments.findIndex((comment) => comment.id === args.comment_id);
        if (index !== -1) comments.splice(index, 1);
      },
    },
    repos: {
      getContent: async (args) => {
        record("getContent", args);
        const content = `[tool.before]\nversion = "99.0.0"\n\n[project]\nversion = '${latestVersion}' # current release\n`;
        return {
          data: {
            type: "file",
            encoding: "base64",
            content: Buffer.from(content).toString("base64"),
          },
        };
      },
    },
  },
};
const context = {
  repo: { owner: "owner", repo: "repo" },
  payload: {
    issue: { number: 7, labels: [], body: "stale event snapshot" },
    repository: {
      default_branch: "main",
      html_url: "https://github.com/owner/repo",
    },
  },
};
const bodyFor = (value) => `### FCC version\n\n${value}\n\n### CLI\n\nClaude Code`;

liveIssue.body = bodyFor("latest");
await run(github, context);
liveIssue.body = bodyFor("The version is 17.23.454");
await run(github, context);
await run(github, context);
liveIssue.body = bodyFor("free-claude-code 17.23.455");
await run(github, context);
liveIssue.body = bodyFor(latestVersion);
await run(github, context);
comments.push({
  id: 102,
  user: { login: "github-actions[bot]" },
  body: "<!-- fcc-version-outdated -->\nstale",
});
liveIssue.labels = [{ name: "needs-fcc-version" }];
liveIssue.body = bodyFor("None");
await run(github, context);

process.stdout.write(JSON.stringify({ calls, comments }));
"""
    result = _run_javascript(
        harness.replace("__SOURCE__", json.dumps(source)).replace(
            "__LATEST__", json.dumps(latest)
        )
    )
    calls = result["calls"]
    names = [call["name"] for call in calls]
    content_reads = [call for call in calls if call["name"] == "getContent"]

    assert names.count("createComment") == 2
    assert names.count("updateComment") == 1
    assert names.count("deleteComment") == 3
    assert names.count("getContent") == 4
    assert names.count("getIssue") == 6
    assert names.count("getLabel") == 1
    assert names.count("addLabels") == 1
    assert names.count("removeLabel") == 2
    assert "createLabel" not in names
    assert all(call["args"]["path"] == "pyproject.toml" for call in content_reads)
    assert all(call["args"]["ref"] == "main" for call in content_reads)
    assert "`17.23.454`" in next(
        call["args"]["body"]
        for call in calls
        if call["name"] == "createComment"
        and "fcc-version-outdated" in call["args"]["body"]
    )
    assert "`17.23.455`" in next(
        call["args"]["body"] for call in calls if call["name"] == "updateComment"
    )
    assert f"`{latest}`" in next(
        call["args"]["body"] for call in calls if call["name"] == "updateComment"
    )
    assert result["comments"] == []
    assert "cancel-in-progress: false" in workflow
    assert "github.rest.issues.update({" not in workflow
    assert 'state: "closed"' not in workflow
