# Local Windows cuTile environment setup.
# Does not install drivers, CUDA Toolkit system-wide, WSL2, Docker, or a runner.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PythonLauncher) {
    throw "Python Launcher 'py' was not found. Install 64-bit CPython 3.12 for the current user from https://www.python.org/downloads/windows/ , enable the Python Launcher, reopen PowerShell, and rerun this script. No system-wide driver/toolkit change is needed."
}

& py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "CPython 3.12 was not found through 'py -3.12'. Install 64-bit CPython 3.12 for the current user, then rerun."
}
$PythonBits = & py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"
if ($LASTEXITCODE -ne 0 -or "$PythonBits".Trim() -ne "64") {
    throw "This setup requires 64-bit CPython 3.12. Found bitness: $PythonBits"
}

$VenvPath = Join-Path $RepoRoot ".venv-cutile-windows"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    & py -3.12 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Creating the project virtual environment failed."
    }
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Updating pip inside .venv-cutile-windows failed."
}

& $PythonPath -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    throw "Installing project CPU/test requirements failed."
}

# CuPy's [ctk] extra supplies the CUDA runtime/compiler libraries inside this
# venv. cuTile's [tileiras] extra supplies TileIR compiler components. This
# avoids assuming that nvcc or a system-wide CUDA Toolkit is installed.
& $PythonPath -m pip install "cupy-cuda13x[ctk]" "cuda-tile[tileiras]"
if ($LASTEXITCODE -ne 0) {
    throw "Installing CuPy/cuTile package wheels failed. Keep the pip error log; do not install a system-wide Toolkit as a blind fix."
}

$LogDirectory = Join-Path $env:LOCALAPPDATA "GPU-NVIDIA\logs"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDirectory "cutile-smoke-$Stamp.log"

"DATE_LOCAL=$(Get-Date -Format o)" | Set-Content -Encoding utf8 -Path $LogPath
"VENV=$VenvPath" | Add-Content -Encoding utf8 -Path $LogPath
& $PythonPath --version 2>&1 | Add-Content -Encoding utf8 -Path $LogPath
"=== pip freeze ===" | Add-Content -Encoding utf8 -Path $LogPath
& $PythonPath -m pip freeze 2>&1 | Add-Content -Encoding utf8 -Path $LogPath
"=== genuine cuTile smoke test ===" | Tee-Object -FilePath $LogPath -Append

& $PythonPath scripts/cutile_smoke.py 2>&1 | Tee-Object -FilePath $LogPath -Append
$SmokeExitCode = $LASTEXITCODE
if ($SmokeExitCode -ne 0) {
    Write-Error "Genuine cuTile smoke test FAILED (exit $SmokeExitCode). Log: $LogPath. Do not proceed to attention tests or runner registration."
    exit $SmokeExitCode
}

Write-Host "Genuine cuTile smoke test passed. Log saved to: $LogPath"
Write-Host "Next: run the targeted cuTile attention tests with REQUIRE_CUTILE=1. Do not register a GitHub Actions runner yet."
