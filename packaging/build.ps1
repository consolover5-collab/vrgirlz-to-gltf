# Build dist\VRScan\VRScan.exe (PyInstaller onedir, windowed) on Windows,
# from the project root or anywhere:
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build.ps1
#
# It uses a venv of its own, .venv-build in the project folder: the project's
# .venv can be in use (a running `vrscan serve` holds its files, and
# `uv sync` fails), and a venv under %TEMP% has been eaten by Defender.
# Versions come from uv.lock (`--frozen`), PyInstaller from the `build` group.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$env:UV_PROJECT_ENVIRONMENT = ".venv-build"
uv sync --frozen --no-dev --group build
if ($LASTEXITCODE -ne 0) { throw "uv sync into .venv-build failed" }
uv run --no-sync pyinstaller packaging\vrscan.spec --noconfirm --clean --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }
Write-Host "Built: $root\dist\VRScan\VRScan.exe"
