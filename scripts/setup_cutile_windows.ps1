# Local Windows cuTile environment setup.
# All environments, caches, temp files and logs are confined to the project root.
# Does not install drivers, system CUDA Toolkit, WSL2, Docker, Nsight Compute, or a runner.
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

$TempRoot = Join-Path $RepoRoot ".tmp"
$LogDirectory = Join-Path $TempRoot "logs"
$PipCache = Join-Path $RepoRoot ".cache\pip"
$CupyCache = Join-Path $RepoRoot ".cache\cupy"
$CudaCache = Join-Path $RepoRoot ".cache\cuda"
New-Item -ItemType Directory -Force -Path $TempRoot,$LogDirectory,$PipCache,$CupyCache,$CudaCache | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:PIP_CACHE_DIR = $PipCache
$env:CUPY_CACHE_DIR = $CupyCache
$env:CUDA_CACHE_PATH = $CudaCache

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PythonLauncher) {
    throw "Python Launcher 'py' was not found. This script will not download or install Python. No changes have been made outside the project root."
}

& py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "CPython 3.12 was not found through 'py -3.12'. This script will not download or install Python."
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
        throw "Creating the project-local virtual environment failed."
    }
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDirectory "cutile-setup-$Stamp.log"
"DATE_LOCAL=$(Get-Date -Format o)" | Set-Content -Encoding utf8 -Path $LogPath
"PROJECT_ROOT=$RepoRoot" | Add-Content -Encoding utf8 -Path $LogPath
"VENV=$VenvPath" | Add-Content -Encoding utf8 -Path $LogPath
"TEMP=$env:TEMP" | Add-Content -Encoding utf8 -Path $LogPath
"PIP_CACHE_DIR=$env:PIP_CACHE_DIR" | Add-Content -Encoding utf8 -Path $LogPath
"CUPY_CACHE_DIR=$env:CUPY_CACHE_DIR" | Add-Content -Encoding utf8 -Path $LogPath
"CUDA_CACHE_PATH=$env:CUDA_CACHE_PATH" | Add-Content -Encoding utf8 -Path $LogPath

& $PythonPath --version 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "Project venv Python failed. Log: $LogPath"
}

# No pip upgrade: the setup installs only project dependencies into this venv.
# All pip caches and temporary extraction/download files are kept under RepoRoot.
$PreviousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"

& $PythonPath -m pip install --disable-pip-version-check --cache-dir $PipCache -r requirements-dev.txt *> (Join-Path $LogDirectory "cutile-requirements-$Stamp.log")
$RequirementsExit = $LASTEXITCODE
Get-Content (Join-Path $LogDirectory "cutile-requirements-$Stamp.log") | Tee-Object -FilePath $LogPath -Append
if ($RequirementsExit -ne 0) {
    $ErrorActionPreference = $PreviousErrorPreference
    throw "Installing requirements-dev.txt inside .venv-cutile-windows failed (exit $RequirementsExit). Log: $LogPath"
}

# Match the package versions already verified by the successful local RTX 4060 run.
# The CUDA component wheels required by cuTile are installed as Python packages
# inside the project venv; no driver or system-toolkit changes are attempted.
& $PythonPath -m pip install --disable-pip-version-check --cache-dir $PipCache "cupy-cuda13x[ctk]==14.2.0" "cuda-tile[tileiras]==1.6.0" *> (Join-Path $LogDirectory "cutile-gpu-packages-$Stamp.log")
$GpuInstallExit = $LASTEXITCODE
Get-Content (Join-Path $LogDirectory "cutile-gpu-packages-$Stamp.log") | Tee-Object -FilePath $LogPath -Append
$ErrorActionPreference = $PreviousErrorPreference
if ($GpuInstallExit -ne 0) {
    throw "Installing CuPy/cuTile inside .venv-cutile-windows failed (exit $GpuInstallExit). Log: $LogPath"
}

"=== pip freeze ===" | Tee-Object -FilePath $LogPath -Append
& $PythonPath -m pip freeze 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "pip freeze failed. Log: $LogPath"
}

"=== genuine cuTile smoke test ===" | Tee-Object -FilePath $LogPath -Append
& $PythonPath scripts/cutile_smoke.py 2>&1 | Tee-Object -FilePath $LogPath -Append
$SmokeExitCode = $LASTEXITCODE
if ($SmokeExitCode -ne 0) {
    throw "Genuine cuTile smoke test FAILED (exit $SmokeExitCode). Log: $LogPath. Do not proceed to attention tests."
}

Write-Host "Genuine cuTile smoke test passed. Log saved under project root: $LogPath"
Write-Host "Next: run $PythonPath -m pytest -q tests/test_gpu_correctness.py tests/test_gpu_edge_cases.py with REQUIRE_CUTILE=1."
