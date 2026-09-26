import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

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


def test_bug_forms_require_description_and_post_install_diagnostics() -> None:
    post = yaml.safe_load(BUG_FORM.read_text(encoding="utf-8"))
    pre = yaml.safe_load(
        BUG_FORM.with_name("installation-problem.yml").read_text(encoding="utf-8")
    )
    assert len(post["body"]) == 2
    assert all(field["validations"]["required"] for field in post["body"])
    assert all(field["type"] == "textarea" for field in post["body"])
    assert post["body"][1]["attributes"]["render"] == "json"
    assert "fcc-doctor" in post["body"][1]["attributes"]["description"]
    assert len(pre["body"]) == 1
    assert pre["body"][0]["validations"]["required"]
    assert "render" not in pre["body"][0]["attributes"]


@pytest.mark.parametrize(
    "body, valid, version",
    [
        ('### FCC doctor output\n\n```json\n{"version":"1.2.3"}\n```', True, "1.2.3"),
        (
            '### FCC doctor output\r\n\r\n```json\r\n{"version":"1.2.3"}\r\n```',
            True,
            "1.2.3",
        ),
        ('### FCC doctor output\n\n{"version":"0+unknown"}', True, None),
        (
            '### FCC doctor output\n\n{"version":"6.2.67.dev1+g123"}',
            True,
            "6.2.67.dev1+g123",
        ),
        (
            '### Describe the issue\n\nversion 9.9.9\n\n### FCC doctor output\n\n{"version":"1.2.3"}',
            True,
            "1.2.3",
        ),
        ("### FCC doctor output\n\n{invalid}", False, None),
        ('### FCC doctor output\n\n{"version":123}', False, None),
        ('### FCC doctor output\n\n{"version":"hello 1.2.3"}', False, None),
        ("### FCC doctor output\n\n{}\n\n### FCC version\n\n1.2.3", False, None),
        ("### Installation issue\n\nCannot install", True, None),
        ("unstructured issue 1.2.3", False, None),
    ],
)
def test_report_parser_uses_the_correct_form_section(body, valid, version):
    patterns = "\n".join(
        f"const {name} = {json.dumps(_workflow_pattern(name))};"
        for name in ("fieldPattern", "versionPattern")
    )
    script = patterns + "\n" + _javascript_function("parseReport")
    script += (
        f"\nprocess.stdout.write(JSON.stringify(parseReport({json.dumps(body)})));"
    )
    result = _run_javascript(script)
    assert result["valid"] is valid
    assert result.get("reportedVersion") == version


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


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v1.2.3", "1.2.3"),
        ("v10.0.0", "10.0.0"),
        ("1.2.3", None),
        ("v1.2.3rc1", None),
        ("v01.2.3", None),
        ("v1.2.3.4", None),
    ],
)
def test_release_tags_are_exact_stable_versions(tag, expected):
    pattern = _workflow_pattern("releaseVersionPattern")
    match = re.fullmatch(pattern, tag)
    assert (match[1] if match else None) == expected


@pytest.mark.parametrize("form", ["legacy", "doctor"])
def test_outdated_version_comment_is_reconciled_across_edits(form) -> None:
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
      getLatestRelease: async (args) => {
        record("getLatestRelease", args);
        return { data: { tag_name: "v" + latestVersion, draft: false, prerelease: false } };
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
const bodyFor = __BODY_FOR__;

liveIssue.body = bodyFor("latest");
await run(github, context);
liveIssue.body = bodyFor("17.23.454");
await run(github, context);
await run(github, context);
liveIssue.body = bodyFor("17.23.455");
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
        harness.replace("__SOURCE__", json.dumps(source))
        .replace("__LATEST__", json.dumps(latest))
        .replace(
            "__BODY_FOR__",
            (
                "(value) => `### FCC version\\n\\n${value}\\n\\n### CLI\\n\\nClaude Code`"
                if form == "legacy"
                else '(value) => value === "None" ? "### Installation issue\\n\\nCannot install" : `### FCC doctor output\\n\\n${JSON.stringify({version: value})}`'
            ),
        )
    )
    calls = result["calls"]
    names = [call["name"] for call in calls]

    assert names.count("createComment") == 2
    assert names.count("updateComment") == 1
    assert names.count("deleteComment") == 3
    assert names.count("getLatestRelease") == 4
    assert names.count("getIssue") == 6
    assert names.count("getLabel") == 1
    assert names.count("addLabels") == 1
    assert names.count("removeLabel") == 2
    assert "createLabel" not in names
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


@pytest.mark.parametrize("status,should_fail", [(404, False), (500, True)])
def test_release_lookup_distinguishes_no_release_from_failure(status, should_fail):
    source = f"return (async () => {{\n{_workflow_script()}\n}})();"
    harness = """
const run = new Function("github", "context", __SOURCE__);
const deleted = [];
const github = {
  paginate: async () => [{id: 42, user: {login: "github-actions[bot]"}, body: "<!-- fcc-version-outdated -->"}],
  rest: {
    issues: {
      get: async () => ({data: {labels: [], body: "### FCC version\\n\\n1.2.3"}}),
      deleteComment: async ({comment_id}) => deleted.push(comment_id),
    },
    repos: {getLatestRelease: async () => {throw {status: __STATUS__};}},
  },
};
let failed = false;
try {await run(github, {repo: {owner: "o", repo: "r"}, payload: {issue: {number: 1}}});}
catch {failed = true;}
process.stdout.write(JSON.stringify({failed, deleted}));
"""
    result = _run_javascript(
        harness.replace("__SOURCE__", json.dumps(source)).replace(
            "__STATUS__", str(status)
        )
    )
    assert result["failed"] is should_fail
    assert result["deleted"] == ([] if should_fail else [42])


@pytest.mark.parametrize(
    "value",
    [
        "6.2.67.dev1+g2f30121c0",
        "6.2.67.dev0+g2f30121c0.d20260925",
        "6.2.67.dev1",
        "6.2.66+d20260925",
    ],
)
def test_development_versions_are_accepted_without_stable_release_comparison(value):
    assert _reported_version(f"free-claude-code {value}") == value
    source = f"return (async () => {{\n{_workflow_script()}\n}})();"
    harness = """
const run = new Function("github", "context", __SOURCE__);
const calls = [];
const github = {
  paginate: async () => [
    {id: 1, user: {login: "github-actions[bot]"}, body: "<!-- fcc-version-validator -->"},
    {id: 2, user: {login: "github-actions[bot]"}, body: "<!-- fcc-version-outdated -->"},
  ],
  rest: {
    issues: {
      get: async () => ({data: {labels: [{name: "needs-fcc-version"}], body: "### FCC version\\n\\nfree-claude-code " + __VERSION__}}),
      removeLabel: async () => calls.push("removeLabel"),
      deleteComment: async ({comment_id}) => calls.push(comment_id),
    },
    repos: {getLatestRelease: async () => {throw new Error("Development builds must not be compared as stable releases");}},
  },
};
await run(github, {repo: {owner: "o", repo: "r"}, payload: {issue: {number: 1}}});
process.stdout.write(JSON.stringify(calls));
"""
    result = _run_javascript(
        harness.replace("__SOURCE__", json.dumps(source)).replace(
            "__VERSION__", json.dumps(value)
        )
    )
    assert result == ["removeLabel", 1, 2]


@pytest.mark.parametrize(
    "value",
    [
        "6.2.67.dev",
        "6.2.67.dev1+gwrong",
        "6.2.67.dev1+g123.extra",
        "6.2.67.dev1+g123.d2026",
        "6.2.67.dev1+g123 and 6.2.66",
    ],
)
def test_invalid_or_ambiguous_development_versions_are_rejected(value):
    assert _reported_version(value) is None
