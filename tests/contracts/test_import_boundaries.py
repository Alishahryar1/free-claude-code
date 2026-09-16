"""Declarative static import contracts; dynamic imports are deliberately unscanned."""

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "src" / "free_claude_code"
_PACKAGE_NAME = "free_claude_code"

ALLOWED_PACKAGE_DEPENDENCIES: dict[str, set[str]] = {
    "config": {"core"},
    "core": set(),
    "application": {"config", "core"},
    "messaging": {"core"},
    "providers": {"application", "config", "core"},
    "api": {"application", "config", "core"},
    "harnesses": {"application", "config", "core"},
    "cli": {"application", "config", "core", "harnesses"},
    "runtime": {
        "api",
        "application",
        "cli",
        "config",
        "core",
        "harnesses",
        "messaging",
        "providers",
    },
}

IMPORT_EXCEPTIONS: dict[tuple[str, str], str] = {
    (
        "free_claude_code.cli.commands",
        "free_claude_code.runtime.bootstrap",
    ): (
        "Owner: installed server command. "
        "Reason: the command delegates construction to the process composition root."
    ),
}

FACADE_ONLY_BOUNDARIES = {
    "free_claude_code.core.openai_responses",
    "free_claude_code.messaging.trees",
    "free_claude_code.providers.openai_chat",
}

OPTIONAL_IMPORT_OWNERS = {
    "librosa": "free_claude_code.messaging.transcription",
    "torch": "free_claude_code.messaging.transcription",
    "transformers": "free_claude_code.messaging.transcription",
    "riva": "free_claude_code.providers.nvidia_nim.voice",
}


@dataclass(frozen=True, slots=True)
class ImportRecord:
    importer: str
    imported: str
    path: str
    line: int
    inside_function: bool

    def describe(self) -> str:
        return f"{self.path}:{self.line}: {self.importer} -> {self.imported}"


class _ImportVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        importer: str,
        importer_is_package: bool,
        modules: set[str],
        path: str,
    ) -> None:
        self._importer = importer
        self._importer_is_package = importer_is_package
        self._modules = modules
        self._path = path
        self._function_depth = 0
        self.records: list[ImportRecord] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolve_import_from_base(
            importer=self._importer,
            importer_is_package=self._importer_is_package,
            level=node.level,
            module=node.module,
        )
        if base is None:
            return
        for alias in node.names:
            candidate = f"{base}.{alias.name}"
            imported = candidate if candidate in self._modules else base
            self._record(imported, node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def _record(self, imported: str, line: int) -> None:
        self.records.append(
            ImportRecord(
                importer=self._importer,
                imported=imported,
                path=self._path,
                line=line,
                inside_function=self._function_depth > 0,
            )
        )


def test_api_configuration_access_goes_through_runtime() -> None:
    storage_modules = {
        "free_claude_code.config.loader",
        "free_claude_code.config.env_files",
        "free_claude_code.config.env_migrations",
    }
    offenders = [
        record.describe()
        for record in _scan_imports(_PACKAGE_ROOT)
        if record.importer.startswith("free_claude_code.api.")
        and record.imported in storage_modules
    ]
    assert not offenders, (
        "API config I/O belongs behind the async runtime port:\n" + "\n".join(offenders)
    )


def test_request_recovery_policy_has_no_client_or_transport_binding() -> None:
    owners = {
        f"{_PACKAGE_NAME}.providers.request_recovery",
        f"{_PACKAGE_NAME}.providers.endpoint",
    }
    forbidden = (
        "openai",
        "httpx",
        "httpx2",
        f"{_PACKAGE_NAME}.providers.openai_client",
        f"{_PACKAGE_NAME}.providers.openai_chat",
        f"{_PACKAGE_NAME}.providers.openai_responses",
        f"{_PACKAGE_NAME}.providers.anthropic_messages",
    )
    offenders = [
        record.describe()
        for record in _scan_imports(_PACKAGE_ROOT)
        if record.importer in owners
        and any(
            record.imported == prefix or record.imported.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    ]
    assert not offenders, (
        "Request recovery policy depends on an adapter:\n" + "\n".join(offenders)
    )


def test_shared_harness_setup_has_no_terminal_owner() -> None:
    former_paths = {
        "cli/environment.py",
        "cli/claude_env.py",
        "cli/vscode.py",
        "cli/codex_integration.py",
        "cli/config_file.py",
        "cli/launchers/resources.py",
        "cli/launchers/model_catalog.py",
        "cli/launchers/codex_model_catalog.py",
    }
    former_modules = {
        f"{_PACKAGE_NAME}.{path.removesuffix('.py').replace('/', '.')}"
        for path in former_paths
    }
    callers = {
        f"{_PACKAGE_NAME}.runtime.codex_app_server",
        f"{_PACKAGE_NAME}.runtime.codex_catalog",
    }
    records = _scan_imports(_PACKAGE_ROOT, known_modules=former_modules)
    offenders = [
        record.describe()
        for record in records
        if record.imported in former_modules
        or (
            record.importer in callers
            and record.imported.startswith(f"{_PACKAGE_NAME}.cli.launchers")
        )
    ]
    remaining = sorted(path for path in former_paths if (_PACKAGE_ROOT / path).exists())
    assert not offenders and not remaining, "\n".join(offenders + remaining)


def test_native_model_catalog_has_no_http_formatter_or_decoder_dependency() -> None:
    callers = {
        f"{_PACKAGE_NAME}.runtime.codex_app_server",
        f"{_PACKAGE_NAME}.runtime.codex_catalog",
        f"{_PACKAGE_NAME}.harnesses.codex_model_catalog",
    }
    offenders = [
        record.describe()
        for record in _scan_imports(_PACKAGE_ROOT)
        if record.importer in callers
        and record.imported.startswith(
            (f"{_PACKAGE_NAME}.api.", f"{_PACKAGE_NAME}.cli.launchers.")
        )
    ]
    assert not offenders, (
        "Native catalog policy belongs outside HTTP adapters:\n" + "\n".join(offenders)
    )
    assert not (_PACKAGE_ROOT / "harnesses/model_catalog.py").exists()


def test_package_dependencies_follow_declarative_policy() -> None:
    modules = _module_paths(_PACKAGE_ROOT)
    records = _scan_imports(_PACKAGE_ROOT)
    actual_packages = _ownership_roots(set(modules), _PACKAGE_NAME)

    assert set(ALLOWED_PACKAGE_DEPENDENCIES) == actual_packages
    assert all(
        (_PACKAGE_ROOT / package / "__init__.py").is_file()
        for package in actual_packages
    )
    assert all(
        dependency in actual_packages and dependency != package
        for package, dependencies in ALLOWED_PACKAGE_DEPENDENCIES.items()
        for dependency in dependencies
    )

    observed_dependencies: set[tuple[str, str]] = set()
    observed_exceptions: set[tuple[str, str]] = set()
    offenders: list[str] = []
    for record in records:
        source_package = _top_level_package(record.importer)
        target_package = _top_level_package(record.imported)
        if source_package is None or target_package is None:
            continue
        if source_package == target_package:
            continue
        exact_edge = (record.importer, record.imported)
        if exact_edge in IMPORT_EXCEPTIONS:
            observed_exceptions.add(exact_edge)
            continue
        if target_package in ALLOWED_PACKAGE_DEPENDENCIES[source_package]:
            observed_dependencies.add((source_package, target_package))
            continue
        offenders.append(record.describe())

    declared_dependencies = {
        (package, dependency)
        for package, dependencies in ALLOWED_PACKAGE_DEPENDENCIES.items()
        for dependency in dependencies
    }
    assert sorted(offenders) == []
    assert declared_dependencies - observed_dependencies == set()
    assert set(IMPORT_EXCEPTIONS) - observed_exceptions == set()
    assert all(
        "Owner:" in reason and "Reason:" in reason
        for reason in IMPORT_EXCEPTIONS.values()
    )
    for source, target in IMPORT_EXCEPTIONS:
        source_package = _top_level_package(source)
        target_package = _top_level_package(target)
        assert source_package is not None
        assert target_package is not None
        assert target_package not in ALLOWED_PACKAGE_DEPENDENCIES[source_package]
    expected_package_modules = {
        _PACKAGE_NAME,
        *(f"{_PACKAGE_NAME}.{package}" for package in actual_packages),
    }
    assert set(modules) >= expected_package_modules


def test_first_party_imports_use_the_installable_namespace() -> None:
    offenders = _legacy_first_party_import_offenders(
        _scan_imports(_PACKAGE_ROOT),
        set(ALLOWED_PACKAGE_DEPENDENCIES),
    )

    assert offenders == []


def test_openai_chat_collaborators_have_explicit_ownership_boundaries() -> None:
    provider_root = _PACKAGE_ROOT / "providers" / "openai_chat"

    assert _provider_backchannel_offenders(provider_root) == []


def test_google_reasoning_wire_fields_have_one_owner() -> None:
    owner = _PACKAGE_ROOT / "providers" / "google_openai" / "reasoning.py"
    roots = [
        _PACKAGE_ROOT / "providers" / "gemini",
        _PACKAGE_ROOT / "providers" / "google_openai",
        _PACKAGE_ROOT / "providers" / "vertex",
    ]
    owned_fields = {
        "include_thoughts",
        "reasoning_effort",
        "thinking_budget",
        "thinking_config",
    }
    offenders: list[str] = []

    for root in roots:
        for path in root.rglob("*.py"):
            if path == owner:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}: {node.value}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and node.value in owned_fields
            )

    assert sorted(offenders) == []


def test_cli_local_http_transport_has_one_owner() -> None:
    cli_root = _PACKAGE_ROOT / "cli"
    owner = cli_root / "local_http.py"
    owned_urllib_names = {"ProxyHandler", "build_opener", "urlopen"}
    offenders: list[str] = []

    for path in cli_root.rglob("*.py"):
        if path == owner:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "urllib.request":
                offenders.extend(
                    f"{relative_path}:{node.lineno}: urllib.request.{alias.name}"
                    for alias in node.names
                    if alias.name in owned_urllib_names
                )
    assert sorted(offenders) == []


def test_child_proxy_environment_has_one_owner() -> None:
    owner = _PACKAGE_ROOT / "harnesses" / "environment.py"
    offenders: list[str] = []
    for root in (_PACKAGE_ROOT / "cli", _PACKAGE_ROOT / "harnesses"):
        for path in root.rglob("*.py"):
            if path == owner:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}: {node.value}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and node.value in {"NO_PROXY", "no_proxy"}
            )
    assert sorted(offenders) == []


def test_provider_backchannel_detector_reports_untyped_private_access(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "openai_chat"
    _write_module(
        provider_root / "sample" / "runner.py",
        "from typing import Any\n"
        "\n"
        "class Runner:\n"
        "    def __init__(self, provider: object | Any) -> None:\n"
        "        self._provider = provider\n"
        "\n"
        "    def run(self) -> object:\n"
        "        return self._provider._send()\n",
    )

    assert _provider_backchannel_offenders(provider_root) == [
        "sample/runner.py:4: untyped provider collaborator",
        "sample/runner.py:8: private provider member _send outside provider.py",
    ]


def test_legacy_first_party_import_detector_rejects_bare_owner_names() -> None:
    record = ImportRecord(
        importer="free_claude_code.api.routes",
        imported="core.anthropic",
        path="free_claude_code/api/routes.py",
        line=7,
        inside_function=False,
    )

    assert _legacy_first_party_import_offenders([record], {"api", "core"}) == [
        "free_claude_code/api/routes.py:7: "
        "free_claude_code.api.routes -> core.anthropic"
    ]


def test_ownership_root_discovery_includes_modules_and_namespace_directories(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "sample"
    _write_module(package_root / "__init__.py")
    _write_module(package_root / "declared" / "__init__.py")
    _write_module(package_root / "namespace" / "module.py")
    _write_module(package_root / "root_module.py")

    roots = _ownership_roots(set(_module_paths(package_root)), "sample")

    assert roots == {"declared", "namespace", "root_module"}


def test_descendants_do_not_import_ancestor_package_facades() -> None:
    modules = _module_paths(_PACKAGE_ROOT)
    packages = {
        module for module, path in modules.items() if path.name == "__init__.py"
    }
    offenders = _ancestor_facade_offenders(_scan_imports(_PACKAGE_ROOT), packages)

    assert offenders == []


def test_static_first_party_import_graph_is_acyclic() -> None:
    modules = set(_module_paths(_PACKAGE_ROOT))
    graph = {module: set() for module in modules}
    for record in _scan_imports(_PACKAGE_ROOT):
        if record.imported in modules:
            graph[record.importer].add(record.imported)

    assert _cyclic_components(graph) == []


def test_cycle_detector_reports_exact_strongly_connected_components() -> None:
    graph = {
        "package.a": {"package.b"},
        "package.b": {"package.a"},
        "package.c": set(),
        "package.self": {"package.self"},
    }

    assert _cyclic_components(graph) == [
        ("package.a", "package.b"),
        ("package.self",),
    ]


def test_external_consumers_use_owned_package_facades() -> None:
    offenders: list[str] = []
    for record in _scan_imports(_PACKAGE_ROOT):
        for facade in FACADE_ONLY_BOUNDARIES:
            if record.importer == facade or record.importer.startswith(f"{facade}."):
                continue
            if record.imported.startswith(f"{facade}."):
                offenders.append(record.describe())

    assert sorted(offenders) == []


def test_import_scanner_resolves_absolute_relative_and_lazy_imports(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "sample"
    _write_module(package_root / "__init__.py")
    _write_module(package_root / "alpha" / "__init__.py", "PUBLIC = object()\n")
    _write_module(package_root / "alpha" / "sibling.py")
    _write_module(package_root / "beta" / "__init__.py")
    _write_module(package_root / "beta" / "absolute.py")
    _write_module(package_root / "beta" / "relative.py")
    _write_module(package_root / "beta" / "lazy.py")
    _write_module(
        package_root / "alpha" / "consumer.py",
        "\n".join(
            (
                "import sample.beta.absolute",
                "from ..beta import relative",
                "from . import sibling",
                "from . import PUBLIC",
                "def load():",
                "    import sample.beta.lazy",
                "",
            )
        ),
    )

    records = _scan_imports(package_root)
    resolved = {
        (record.imported, record.inside_function)
        for record in records
        if record.importer == "sample.alpha.consumer"
    }

    assert resolved == {
        ("sample.alpha", False),
        ("sample.alpha.sibling", False),
        ("sample.beta.absolute", False),
        ("sample.beta.relative", False),
        ("sample.beta.lazy", True),
    }


def test_import_scanner_distinguishes_facade_symbols_from_sibling_modules(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "sample"
    _write_module(package_root / "__init__.py")
    _write_module(package_root / "owner" / "__init__.py", "PUBLIC = object()\n")
    _write_module(package_root / "owner" / "sibling.py")
    _write_module(
        package_root / "owner" / "consumer.py",
        "from . import PUBLIC, sibling\n",
    )
    modules = _module_paths(package_root)
    packages = {
        module for module, path in modules.items() if path.name == "__init__.py"
    }

    offenders = _ancestor_facade_offenders(_scan_imports(package_root), packages)

    assert offenders == [
        "sample/owner/consumer.py:1: sample.owner.consumer -> sample.owner"
    ]


def test_anthropic_request_boundaries_use_the_protocol_model() -> None:
    """Known Messages fields must not cross core/provider boundaries by duck typing."""
    roots = [
        _PACKAGE_ROOT / "core" / "anthropic",
        _PACKAGE_ROOT / "providers",
    ]
    request_names = {"request", "request_data"}
    protocol_fields = {
        "extra_body",
        "max_tokens",
        "messages",
        "model",
        "stop_sequences",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
    offenders: list[str] = []

    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(_REPO_ROOT).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    arguments = [
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ]
                    for argument in arguments:
                        if argument.arg.lstrip("_") not in request_names:
                            continue
                        annotation_names = (
                            {
                                child.id
                                for child in ast.walk(argument.annotation)
                                if isinstance(child, ast.Name)
                            }
                            if argument.annotation is not None
                            else set()
                        )
                        if argument.annotation is None or annotation_names & {
                            "Any",
                            "Mapping",
                        }:
                            offenders.append(
                                f"{relative}:{argument.lineno}: "
                                f"{node.name}({argument.arg}) is not concrete"
                            )
                if not isinstance(node, ast.Call) or not (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id.lstrip("_") in request_names
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in protocol_fields
                ):
                    continue
                offenders.append(
                    f"{relative}:{node.lineno}: "
                    f"getattr({node.args[0].id}, {node.args[1].value!r})"
                )

    assert sorted(offenders) == []


def test_core_does_not_import_provider_transport_sdks() -> None:
    forbidden_roots = {"aiohttp", "httpx", "openai"}
    offenders = [
        record.describe()
        for record in _scan_imports(_PACKAGE_ROOT)
        if (
            record.importer == "free_claude_code.core"
            or record.importer.startswith("free_claude_code.core.")
        )
        and record.imported.split(".", 1)[0] in forbidden_roots
    ]

    assert sorted(offenders) == []


def test_web_tool_workflow_has_no_http_adapter_owner() -> None:
    former_owner = "free_claude_code.api.web_tools"
    application_owner = "free_claude_code.application.web_tools"
    forbidden_clients = {"aiohttp", "httpx", "requests", "socket"}
    offenders = [
        record.describe()
        for record in _scan_imports(_PACKAGE_ROOT)
        if record.imported.startswith(former_owner)
        or (
            record.importer.startswith(application_owner)
            and record.imported.split(".", 1)[0] in forbidden_clients
        )
    ]
    remaining = list((_PACKAGE_ROOT / "api" / "web_tools").glob("*.py"))
    assert not offenders and not remaining, "\n".join(
        [*offenders, *(str(path.relative_to(_REPO_ROOT)) for path in remaining)]
    )


def test_providers_do_not_own_wire_error_type_literals() -> None:
    wire_types = {
        "api_error",
        "authentication_error",
        "billing_error",
        "invalid_request_error",
        "not_found_error",
        "overloaded_error",
        "permission_error",
        "rate_limit_error",
        "request_too_large",
        "timeout_error",
    }
    offenders: list[str] = []
    for path in (_PACKAGE_ROOT / "providers").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}: {node.value}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value in wire_types
        )

    assert sorted(offenders) == []


def test_optional_dependencies_have_one_lazy_owner() -> None:
    seen: set[str] = set()
    offenders: list[str] = []
    for record in _scan_imports(_PACKAGE_ROOT):
        dependency = record.imported.split(".", 1)[0]
        owner = OPTIONAL_IMPORT_OWNERS.get(dependency)
        if owner is None:
            continue
        seen.add(dependency)
        if record.importer != owner or not record.inside_function:
            offenders.append(record.describe())

    assert seen == set(OPTIONAL_IMPORT_OWNERS)
    assert sorted(offenders) == []


def test_runtime_imports_without_optional_voice_dependencies() -> None:
    blocked = sorted(OPTIONAL_IMPORT_OWNERS)
    script = "\n".join(
        (
            "import importlib.abc",
            "import sys",
            f"BLOCKED = {blocked!r}",
            "class Blocker(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname.split('.', 1)[0] in BLOCKED:",
            "            raise ModuleNotFoundError(fullname)",
            "        return None",
            "sys.meta_path.insert(0, Blocker())",
            "import free_claude_code.runtime.bootstrap",
            "import free_claude_code.api.app",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_supported_messaging_facade_is_explicit() -> None:
    import free_claude_code.messaging as facade
    from free_claude_code.messaging.managed_protocols import (
        ManagedClaudeSessionManagerProtocol,
        ManagedClaudeSessionProtocol,
    )
    from free_claude_code.messaging.models import IncomingMessage, MessageScope
    from free_claude_code.messaging.platforms.ports import OutboundMessenger

    expected = {
        "IncomingMessage": IncomingMessage,
        "ManagedClaudeSessionManagerProtocol": ManagedClaudeSessionManagerProtocol,
        "ManagedClaudeSessionProtocol": ManagedClaudeSessionProtocol,
        "MessageScope": MessageScope,
        "OutboundMessenger": OutboundMessenger,
    }

    assert set(facade.__all__) == set(expected)
    assert all(getattr(facade, name) is value for name, value in expected.items())


def test_message_tree_mutability_stays_behind_its_facade() -> None:
    import free_claude_code.messaging.trees as facade

    for internal_owner in {
        "MessageNode",
        "MessageTree",
        "TreeQueueProcessor",
        "TreeRepository",
    }:
        assert internal_owner not in facade.__all__
        assert not hasattr(facade, internal_owner)


def _module_paths(package_root: Path) -> dict[str, Path]:
    return {
        _module_name(package_root, path): path for path in package_root.rglob("*.py")
    }


def _module_name(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root)
    module_parts = (
        relative.parent.parts
        if path.name == "__init__.py"
        else relative.with_suffix("").parts
    )
    return ".".join((package_root.name, *module_parts))


def _scan_imports(
    package_root: Path, *, known_modules: set[str] | None = None
) -> list[ImportRecord]:
    module_paths = _module_paths(package_root)
    modules = set(module_paths) | (known_modules or set())
    records: list[ImportRecord] = []
    for importer, path in sorted(module_paths.items()):
        visitor = _ImportVisitor(
            importer=importer,
            importer_is_package=path.name == "__init__.py",
            modules=modules,
            path=path.relative_to(package_root.parent).as_posix(),
        )
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        records.extend(visitor.records)
    return records


def _resolve_import_from_base(
    *,
    importer: str,
    importer_is_package: bool,
    level: int,
    module: str | None,
) -> str | None:
    if level == 0:
        return module
    package_parts = importer.split(".")
    if not importer_is_package:
        package_parts.pop()
    parents_to_remove = level - 1
    if parents_to_remove > len(package_parts):
        return None
    if parents_to_remove:
        del package_parts[-parents_to_remove:]
    if module is not None:
        package_parts.extend(module.split("."))
    return ".".join(package_parts) or None


def _top_level_package(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != _PACKAGE_NAME:
        return None
    return parts[1]


def _ownership_roots(modules: set[str], package_name: str) -> set[str]:
    roots: set[str] = set()
    for module in modules:
        parts = module.split(".")
        if len(parts) >= 2 and parts[0] == package_name:
            roots.add(parts[1])
    return roots


def _legacy_first_party_import_offenders(
    records: list[ImportRecord],
    owner_names: set[str],
) -> list[str]:
    offenders = [
        record.describe()
        for record in records
        if record.imported.split(".", 1)[0] in owner_names
    ]
    return sorted(offenders)


def _provider_backchannel_offenders(provider_root: Path) -> list[str]:
    offenders: list[str] = []
    for path in sorted(provider_root.rglob("*.py")):
        relative_path = path.relative_to(provider_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                arguments = (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                offenders.extend(
                    f"{relative_path}:{argument.lineno}: untyped provider collaborator"
                    for argument in arguments
                    if argument.arg == "provider"
                    and _annotation_is_any(argument.annotation)
                )
            if (
                path.name != "provider.py"
                and isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and _is_provider_reference(node.value)
            ):
                offenders.append(
                    f"{relative_path}:{node.lineno}: private provider member "
                    f"{node.attr} outside provider.py"
                )
    return sorted(offenders)


def _annotation_is_any(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return any(
        (isinstance(node, ast.Name) and node.id == "Any")
        or (isinstance(node, ast.Attribute) and node.attr == "Any")
        for node in ast.walk(annotation)
    )


def _is_provider_reference(expression: ast.expr) -> bool:
    return (isinstance(expression, ast.Name) and expression.id == "provider") or (
        isinstance(expression, ast.Attribute)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "self"
        and expression.attr == "_provider"
    )


def _ancestor_facade_offenders(
    records: list[ImportRecord],
    packages: set[str],
) -> list[str]:
    offenders = [
        record.describe()
        for record in records
        if record.imported in packages
        and record.importer.startswith(f"{record.imported}.")
    ]
    return sorted(offenders)


def _cyclic_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for successor in sorted(graph[node]):
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])

        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in graph[node]:
            components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components)


def _write_module(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
