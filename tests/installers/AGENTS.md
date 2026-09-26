# Installer test rules

These rules apply to installer, uninstaller, and harness lifecycle tests here.

- Test behavior through the real script functions. Use focused function scenarios
  for command discovery, version validation, process checks, and component errors.
  Do not run the entire installer for every variation of a component rule.
- Keep full workflows for fresh installation, upgrades, cross-harness interactions,
  and abort ordering between installation phases. Preserve those assertions when
  making tests faster. Prepare unrelated prerequisites as already installed.
- Keep coverage for both Windows PowerShell and PowerShell 7, plus the applicable
  POSIX paths. Do not skip coverage or move it out of CI to hide slow tests.
- Stub downloads and external installers. Never install real harnesses, use live
  providers, or modify the user's installed tools and configuration in these tests.
- Give every scenario private files, configuration, PATH, and cache storage. Use
  the shared `powershell_module_paths` fixture for Windows subprocesses, including
  child installers. Inherited host modules can make missing-command discovery scan
  or load unrelated software. `-NoProfile` alone does not isolate module discovery.
- Disable disk module-analysis caching with `PSModuleAnalysisCachePath=NUL` for
  Windows test processes, including fixture bootstrap and nested children. Test
  inheritance and command discovery without waiting for background cache writes.
- Preserve process isolation between scenarios. Do not share mutable shell state
  or user configuration to save startup time. Clean up processes you launch.
- Diagnose slowness with per-test durations and timings inside subprocesses. Do not
  increase timeouts or add sleeps without identifying the operation being delayed.
- Use `scripts/ci.ps1` for local verification. `PYTEST_ADDOPTS` can select focused
  cases and enable duration reporting. Run the full relevant suite before pushing.
- Reuse pytest's capture handler for logs. Do not create a logging cycle between
  Python logging and Loguru while adding diagnostics.
