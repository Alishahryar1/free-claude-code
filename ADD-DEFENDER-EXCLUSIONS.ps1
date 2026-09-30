# Optional: cut FCC's import cost by keeping Defender out of the hot path.
#
# Why: on this machine `import fastapi` takes ~21.5s against a ~0.5s baseline.
# The cause is Defender real-time protection scanning every file an import
# touches. That slows the test suite (which is what produced 20 bogus
# TimeoutExpired failures) and also slows real FCC startup and restarts.
#
# Trade-off: these directories stop being scanned. They hold Python source and
# virtualenvs you control, so the risk is low, but it is a real reduction in
# coverage. Skip this if you would rather keep scanning everything.
#
# MUST be run from an elevated PowerShell (Run as Administrator).

$repo = "$env:USERPROFILE\OneDrive\Documents\Default Project\free-claude-code"

$paths = @(
    $repo                                                   # project + its .venv
    "$env:USERPROFILE\.local\bin"                           # uv shims
    "$env:APPDATA\uv\tools\free-claude-code"                # live install
    "$env:LOCALAPPDATA\uv\cache"                            # uv cache
    "$env:USERPROFILE\.local\share\uv"                      # uv tools/state
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        Add-MpPreference -ExclusionPath $p
        "excluded: $p"
    }
}

"--- verify (empty here means the earlier read was blocked, not that there are none) ---"
(Get-MpPreference).ExclusionPath

"--- expected effect ---"
'Re-run:  python -c "import time;t=time.time();import fastapi;print(round(time.time()-t,2))"'
"Anything near 0.5s means the exclusions took effect."

"--- to undo ---"
"Remove-MpPreference -ExclusionPath '<path>'   # repeat per path; list with (Get-MpPreference).ExclusionPath"
