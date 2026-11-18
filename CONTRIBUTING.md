# Contributing

Thanks for helping improve Free Claude Code. Keep changes focused, test the behavior you change, and preserve the public Claude Code and Codex workflows.

## Before Opening A Pull Request

- Standalone documentation PRs are reserved for the repository owner.
- Include relevant README and other documentation updates with production code changes. These accompanying updates do not require a separate issue.
- Do not open Docker integration pull requests.
- For bugs, include every model mapping, the active model when the failure occurred, the complete error, and reproducible steps.
- Add focused tests for behavior changes and relevant edge cases.

## Pull Request Titles

Changes to `assets/`, `scripts/`, `src/`, `pyproject.toml`, or `uv.lock` require a title starting with `patch: `, `minor: `, or `major: ` followed by a description. Use exactly one space after the colon.

PRs without changes to those paths must not use a release prefix. Do not edit version numbers. Publishing calculates the version after merge, creates its Git tag, and publishes to PyPI and GitHub Releases.

## Windows release signing

The post-merge workflow builds the x64 ZIP from the exact staged FCC wheel and the locked dependencies. It preserves valid upstream signatures, signs the unsigned executables and libraries, and checks the ZIP's GitHub build attestation before publishing the release.

Before enabling this release path, configure an [Azure Artifact Signing Public Trust profile](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart) for the verified publisher. Create a GitHub Actions environment named `windows-signing`, restricted to the `main` deployment branch. Follow the [Azure OIDC setup](https://github.com/Azure/artifact-signing-action/blob/main/docs/OIDC.md) for that environment and grant the federated identity the Artifact Signing Certificate Profile Signer role on the profile. Set these environment variables in Actions.

- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- `ARTIFACT_SIGNING_ENDPOINT`, `ARTIFACT_SIGNING_ACCOUNT`, `ARTIFACT_SIGNING_PROFILE`

The first signed Windows release stays in draft after signing. On a disposable Windows 11 test system with Smart App Control set to On, authenticate `gh` and run `./scripts/windows/accept_release_candidate.ps1 -Tag <tag>`. It retrieves the signed draft ZIP and runs the release's installer. Run `fcc-doctor`, start FCC Desktop and the proxy, and exercise a harness launcher. Repeat with `-Update` over the previous release and run the candidate's uninstaller. Check the Code Integrity log for blocked components using the [Microsoft signing test procedure](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/test-your-app-with-smart-app-control).

Once the candidate passes, dispatch the **Post merge** workflow on `main` with the draft's commit as `release_commit` and the tested ZIP's digest as `accepted_windows_digest` in `sha256:<lowercase hex>` form. The publisher checks the exact ZIP, release commit, and staged wheel before releasing it. Later releases publish automatically after signing and verification. If verification rejects an uploaded ZIP, inspect the draft asset before removing it and retrying the same release commit.

If acceptance finds a bug that requires a code change, dispatch the workflow for the corrected main commit and set `supersedes_windows_candidate` to the rejected draft tag. This allocates a new version and keeps the rejected candidate in draft. Include the same rejected tag when dispatching acceptance of the corrected candidate.

The default Windows install uses the signed bundle. `-VoiceLocal` uses the uv install with the optional PyTorch dependencies.

## Development Setup

Install [uv](https://docs.astral.sh/uv/), then run directly from the checkout. uv automatically installs the Python version required by the project.

```bash
git clone https://github.com/Alishahryar1/free-claude-code.git
cd free-claude-code
uv run fcc-server
```

Use `uv run` for Python commands. Do not run the project with a global Python interpreter.

## Quality Checks

Run the complete local CI sequence before opening a pull request:

```bash
./scripts/ci.sh
```

```powershell
.\scripts\ci.ps1
```

Useful iteration flags are `--only`, `--skip`, and `--dry-run` on macOS/Linux, or `-Only`, `-Skip`, and `-DryRun` in PowerShell.

Individual repair and test commands:

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check
uv run pytest -v --tb=short
```

GitHub CI runs Ruff in check-only mode and also bans `# type: ignore`, `# ty: ignore`, and legacy annotation workarounds. Fix underlying typing and import-boundary problems instead of suppressing them.

## Project Standards

- Target Python 3.14 and rely on native lazy annotations; do not add `from __future__ import annotations`.
- Python 3.14 supports multiple exception types without parentheses, such as `except TypeError, ValueError:`.
- Keep shared Anthropic protocol behavior under `src/free_claude_code/core/anthropic/` rather than importing utilities from another provider.
- Keep provider-specific configuration in the provider that owns it.
- Remove dead compatibility code when completing migrations unless preserving a published interface is explicitly required.
