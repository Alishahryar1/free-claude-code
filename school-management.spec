# PyInstaller onedir bundle for the local French school-management app.
from PyInstaller.utils.hooks import collect_submodules


hidden_imports = collect_submodules("uvicorn")

analysis = Analysis(
    ["src/free_claude_code/school_management/app.py"],
    pathex=["src"],
    binaries=[],
    datas=[("school-management", "school-management")],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="GestionScolaire",
    icon="school-management/assets/school-mark.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="GestionScolaire",
)
