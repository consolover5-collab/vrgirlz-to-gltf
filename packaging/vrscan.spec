# PyInstaller spec: the Windows onedir build dist\VRScan\VRScan.exe, windowed
# (no console). Build with packaging\build.ps1 (docs/LAUNCHER.md, "Packaged
# exe"); PyInstaller is pinned in pyproject.toml's `build` group.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
VIEWER = ROOT / "viewer"

# viewer/vendor is not in git (scripts/vendor-three.sh fetches it); without
# it the served viewer is a blank page.
for required in (VIEWER / "vendor" / "three" / "build" / "three.module.js", VIEWER / "vendor" / "webxr-input-profiles"):
    if not required.exists():
        raise SystemExit(f"{required} is missing - run scripts/vendor-three.sh first")

# What `vrscan serve` serves (`cli._viewer_dir`): the page, its modules,
# three.js and the controller models — not the viewer's tests or dev pages.
datas = [
    (str(VIEWER / "index.html"), "viewer"),
    (str(VIEWER / "src"), "viewer/src"),
    (str(VIEWER / "vendor"), "viewer/vendor"),
    *collect_data_files("UnityPy"),  # its type trees (resources/*.tpk)
    # astc_encoder, imported by UnityPy's texture decoding, picks its CPU
    # variant with archspec, which reads its CPU tables from JSON.
    *collect_data_files("archspec"),
]

a = Analysis(
    [str(ROOT / "packaging" / "vrscan_exe.py")],
    pathex=[str(ROOT / "src")],
    # UnityPy's texture decoding imports its audio converter, which loads
    # FMOD's library by path (ctypes) on import: invisible to the analysis.
    binaries=collect_dynamic_libs("fmod_toolkit"),
    datas=datas,
    excludes=["pytest", "_pytest", "trimesh"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VRScan",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="VRScan", upx=False)
