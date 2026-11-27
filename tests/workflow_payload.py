"""Typed sections shared by tests of the repository's GitHub Actions workflows."""

from pathlib import Path
from typing import TypedDict

import yaml
from pydantic import TypeAdapter

ActionStep = TypedDict(
    "ActionStep",
    {
        "id": str,
        "name": str,
        "uses": str,
        "run": str,
        "if": str,
        "with": dict[str, str | int | bool],
        "env": dict[str, str],
    },
    total=False,
)


class Matrix(TypedDict, total=False):
    include: list[dict[str, str]]


Strategy = TypedDict(
    "Strategy",
    {"matrix": Matrix | str, "fail-fast": bool, "max-parallel": int},
    total=False,
)


WorkflowJob = TypedDict(
    "WorkflowJob",
    {
        "name": str,
        "if": str,
        "runs-on": str,
        "environment": str,
        "continue-on-error": str | bool,
        "permissions": dict[str, str],
        "outputs": dict[str, str],
        "needs": str | list[str],
        "env": dict[str, str],
        "strategy": Strategy,
        "steps": list[ActionStep],
    },
    total=False,
)


class PullRequestTrigger(TypedDict, total=False):
    types: list[str]


class Workflow(TypedDict, total=False):
    jobs: dict[str, WorkflowJob]
    env: dict[str, str]
    concurrency: dict[str, str | bool]


class CompositeRun(TypedDict):
    steps: list[ActionStep]


class CompositeAction(TypedDict, total=False):
    runs: CompositeRun
    inputs: dict[str, dict[str, str | bool]]


def _read_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_workflow(path: Path) -> Workflow:
    value = _read_yaml(path)
    assert isinstance(value, dict)
    # PyYAML reads the YAML 1.1 spelling "on" as a boolean key.
    config = {key: item for key, item in value.items() if isinstance(key, str)}
    return TypeAdapter(Workflow).validate_python(config, strict=True)


def read_workflow_triggers(path: Path) -> dict[str, PullRequestTrigger | None]:
    value = _read_yaml(path)
    assert isinstance(value, dict)
    triggers: object = value.get(True)
    return TypeAdapter(dict[str, PullRequestTrigger | None]).validate_python(
        triggers, strict=True
    )


def read_action(path: Path) -> CompositeAction:
    return TypeAdapter(CompositeAction).validate_python(_read_yaml(path), strict=True)
