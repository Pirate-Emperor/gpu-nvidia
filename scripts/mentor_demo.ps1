[CmdletBinding()]
param(
    [switch]$SkipTileSweep,
    [switch]$WithProfile
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pyCandidates = @(
    (Join-Path $repoRoot ".venv-cutile-windows\Scripts\python.exe"),
    "C:\Users\Public\GPU-cutile-project\gpu-nvidia\.venv-cutile-windows\Scripts\python.exe"
)
$py = $pyCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (!$py) { throw "Project cuTile Python not found. Checked: $($pyCandidates -join '; ')" }
$logDir = Join-Path $repoRoot ".tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$inFile = Join-Path $logDir "mentor_attention_input_$stamp.npz"
$outFile = Join-Path $logDir "mentor_attention_output_$stamp.npy"
$testLog = Join-Path $logDir "mentor_tests_$stamp.log"
$smokeLog = Join-Path $logDir "mentor_smoke_$stamp.log"
$cliLog = Join-Path $logDir "mentor_cli_$stamp.log"

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$CmdArgs
    )
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    & $Exe @CmdArgs
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code" }
}

Write-Host "REPO=$repoRoot"
Write-Host "PYTHON_EXE=$py"
Invoke-Checked "Python version" $py @("--version")
Invoke-Checked "NVIDIA GPU inventory" "nvidia-smi" @()
Invoke-Checked "Structured GPU inventory" "nvidia-smi" @("--query-gpu=name,driver_version,compute_cap,memory.total,memory.free,pci.bus_id","--format=csv")

Write-Host ""
Write-Host "=== Genuine cuTile smoke ===" -ForegroundColor Cyan
& $py (Join-Path $repoRoot "scripts\cutile_smoke.py") 2>&1 | Tee-Object -FilePath $smokeLog
$smokeExit = $LASTEXITCODE
if ($smokeExit -ne 0) { throw "cuTile smoke failed with exit code $smokeExit. Log: $smokeLog" }
if (!(Select-String -Path $smokeLog -SimpleMatch "CUTILE_SMOKE=PASS" -Quiet)) { throw "Smoke output lacks CUTILE_SMOKE=PASS. Log: $smokeLog" }

$oldRequireCutile = $env:REQUIRE_CUTILE
try {
    $env:REQUIRE_CUTILE = "1"
    Write-Host ""
    Write-Host "=== Required actual-cuTile attention and edge tests ===" -ForegroundColor Cyan
    & $py -m pytest -q -ra (Join-Path $repoRoot "tests\test_gpu_correctness.py") (Join-Path $repoRoot "tests\test_gpu_edge_cases.py") 2>&1 | Tee-Object -FilePath $testLog
    $testExit = $LASTEXITCODE
    if ($testExit -ne 0) { throw "Required GPU tests failed with exit code $testExit. Log: $testLog" }
}
finally {
    if ($null -eq $oldRequireCutile) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
    else { $env:REQUIRE_CUTILE = $oldRequireCutile }
}

Write-Host ""
Write-Host "=== Generate deterministic NPZ input ===" -ForegroundColor Cyan
$genArgs = @(
    (Join-Path $repoRoot "scripts\generate_inputs.py"),
    "--output",$inFile,
    "--batch","1",
    "--heads","1",
    "--seq","64",
    "--head-dim","32",
    "--seed","20261018"
)
& $py @genArgs
$genExit = $LASTEXITCODE
if ($genExit -ne 0) { throw "Input generation failed with exit code $genExit" }
if (!(Test-Path -LiteralPath $inFile)) { throw "Input generator did not create expected file: $inFile" }

Write-Host ""
Write-Host "=== Run course-facing cuTile CLI ===" -ForegroundColor Cyan
$cliArgs = @(
    (Join-Path $repoRoot "run_attention.py"),
    "--input",$inFile,
    "--output",$outFile,
    "--backend","cutile",
    "--tile-q","64",
    "--tile-k","64"
)
& $py @cliArgs 2>&1 | Tee-Object -FilePath $cliLog
$cliExit = $LASTEXITCODE
if ($cliExit -ne 0) { throw "cuTile CLI failed with exit code $cliExit. Log: $cliLog" }
if (!(Test-Path -LiteralPath $outFile)) { throw "CLI did not create expected output: $outFile" }

Write-Host ""
Write-Host "=== Independently verify saved output with NumPy oracle ===" -ForegroundColor Cyan
$verifyArgs = @(
    (Join-Path $repoRoot "scripts\verify_cli_output.py"),
    "--input",$inFile,
    "--output",$outFile
)
Invoke-Checked "CLI vs independent NumPy oracle" $py $verifyArgs

if (!$SkipTileSweep) {
    $benchFile = Join-Path $logDir "mentor_benchmark_$stamp.json"
    $benchCsv = Join-Path $logDir "mentor_benchmark_$stamp.csv"
    Write-Host ""
    Write-Host "=== Runtime tile comparison: five tile shapes x two workloads ===" -ForegroundColor Cyan
    Write-Host "This is a presentation-sized sweep: it is not the full three-session experiment."
    Write-Host "Each tile gets 10 warmups, 30 CUDA-event samples, and 5 launches/sample."
    Write-Host "The script records median, P90, unfused FP32 CuPy baseline, speedup, baseline drift, and first-call wall time."
    $benchArgs = @(
        (Join-Path $repoRoot "benchmarks\benchmark_attention.py"),
        "--output",$benchFile,
        "--warmup","10",
        "--iterations","30",
        "--repeats-per-sample","5",
        "--tiles","32x64","64x64","64x128","128x64","128x128",
        "--case","medium_d128",
        "--case","longer_batched_d128"
    )
    Invoke-Checked "Runtime tile sweep" $py $benchArgs
    if (!(Test-Path -LiteralPath $benchFile)) { throw "Benchmark did not create $benchFile" }
    if (!(Test-Path -LiteralPath $benchCsv)) { throw "Benchmark did not create $benchCsv" }
    Write-Host "BENCHMARK_JSON=$benchFile"
    Write-Host "BENCHMARK_CSV=$benchCsv"
    Write-Host "Use the printed BEST MEDIAN TILE PER CASE, then compare P90 and drift before drawing a conclusion."
}
else {
    Write-Host "TILE_SWEEP=SKIPPED (explicit -SkipTileSweep)"
}

if ($WithProfile) {
    $ncuCandidates = @(
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\ncu.exe"
    )
    $ncu = $ncuCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (!$ncu) {
        $ncu = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Filter "ncu.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    }
    if (!$ncu) { throw "Nsight Compute CLI ncu.exe not found" }
    Invoke-Checked "Nsight Compute version" $ncu @("--version")
    $profileBase = Join-Path $logDir "mentor_medium_d128_64x64_$stamp"
    $profileLog = Join-Path $logDir "mentor_ncu_$stamp.log"
    $profileArgs = @(
        "--set","basic",
        "--launch-skip","6",
        "--launch-count","1",
        "--force-overwrite",
        "-o",$profileBase,
        $py,
        (Join-Path $repoRoot "benchmarks\profile_fused.py"),
        "--case","medium_d128",
        "--tile-q","64",
        "--tile-k","64",
        "--warmup","5"
    )
    Write-Host ""
    Write-Host "=== Nsight Compute: attempt one warmed kernel capture ===" -ForegroundColor Cyan
    & $ncu @profileArgs 2>&1 | Tee-Object -FilePath $profileLog
    $profileExit = $LASTEXITCODE
    $profileText = Get-Content -LiteralPath $profileLog -Raw
    if ($profileText -match "ERR_NVGPUCTRPERM") {
        Write-Warning "NCU_STATUS=BLOCKED_ERR_NVGPUCTRPERM. The kernel may launch, but this session cannot collect the requested hardware counters."
        Write-Host "Enable access in NVIDIA App > System > Advanced > Developer > Manage GPU Performance Counters. This setting requires administrator approval. Alternatively, open PowerShell as Administrator for a diagnostic rerun."
        Write-Host "Saved profile log: $profileLog"
    }
    elseif ($profileExit -ne 0) {
        throw "Nsight Compute failed with exit code $profileExit. Log: $profileLog"
    }
    else {
        Write-Host "NCU_COMMAND_EXIT=0"
        Write-Host "Check the saved log and report file before claiming counters were collected."
        Write-Host "NCU_LOG=$profileLog"
        Write-Host "NCU_REPORT_BASE=$profileBase"
    }
}

Write-Host ""
Write-Host "MENTOR_DEMO_CORE=PASS" -ForegroundColor Green
Write-Host "SMOKE_LOG=$smokeLog"
Write-Host "TEST_LOG=$testLog"
Write-Host "CLI_LOG=$cliLog"
Write-Host "INPUT_NPZ=$inFile"
Write-Host "OUTPUT_NPY=$outFile"
Write-Host "Next: show the saved benchmark summary and previously captured NCU report. The default run includes the runtime tile sweep. Add -SkipTileSweep only for a quick smoke/test/CLI check; add -WithProfile for a live NCU attempt."
