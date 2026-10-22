[CmdletBinding()]
param(
    [switch]$NoPause,
    [switch]$Guided,
    [switch]$Standard,
    [switch]$Menu,
    [switch]$SkipNumericalMatrix,
    [switch]$SkipTileExperiment,
    [switch]$SkipLatencyHintExperiment
)

$runMenu = $Menu -or ($PSBoundParameters.Count -eq 0)
if ($runMenu) {
    $ErrorActionPreference = "Stop"
    $menuRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $menuPyCandidates = @(
        (Join-Path $menuRoot ".venv-cutile-windows\Scripts\python.exe"),
        "C:\Users\Public\GPU-cutile-project\gpu-nvidia\.venv-cutile-windows\Scripts\python.exe"
    )
    $menuPy = $menuPyCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (!$menuPy) { throw "Project cuTile Python not found. Checked: $($menuPyCandidates -join '; ')" }
    $menuLogDir = Join-Path $menuRoot ".tmp\logs"
    New-Item -ItemType Directory -Force -Path $menuLogDir | Out-Null
    $configPath = Join-Path $menuRoot "tests\legal_attention_configs.json"
    $registryScript = Join-Path $menuRoot "scripts\legal_config_registry.py"
    $matrixScript = Join-Path $menuRoot "scripts\mentor_numerical_matrix.py"

    function Invoke-MenuScript {
        param([string]$ScriptPath, [string[]]$Arguments = @())
        if (!(Test-Path -LiteralPath $ScriptPath)) { throw "Script not found: $ScriptPath" }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            Write-Host "The selected operation returned exit code $LASTEXITCODE. Review its output before continuing." -ForegroundColor Red
        }
    }
    function Invoke-MenuPython {
        param([string[]]$Arguments)
        & $menuPy @Arguments
        if ($LASTEXITCODE -ne 0) {
            Write-Host "The selected Python operation returned exit code $LASTEXITCODE." -ForegroundColor Red
        }
    }
    function Get-RequiredGpuTestNodes {
        $collected = & $menuPy -m pytest --collect-only -q tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Pytest collection failed: $($collected -join [Environment]::NewLine)" }
        $nodes = @(
            $collected |
            ForEach-Object { $_.ToString().Trim() } |
            Where-Object { $_ -match '^tests[/\\]test_gpu_(correctness|edge_cases)\.py::test_' }
        )
        if ($nodes.Count -ne 48) {
            throw "Expected the fixed suite to contain 48 GPU test executions; collection reported $($nodes.Count). Check the test files before presenting."
        }
        return $nodes
    }

    while ($true) {
        Write-Host ""
        Write-Host ("=" * 88) -ForegroundColor DarkGray
        Write-Host "CUTILE ATTENTION — MENTOR DEMONSTRATION MENU" -ForegroundColor Cyan
        Write-Host "Python: $menuPy"
        Write-Host "Editable numerical test list: $configPath"
        Write-Host ""
        Write-Host " 1. Standard step-by-step demo"
        Write-Host " 2. Guided kernel explanation + run each of 48 tests + each legal numerical configuration"
        Write-Host " 3. Run all 48 required GPU tests together (verbose)"
        Write-Host " 4. Select and run one of the 48 required GPU tests"
        Write-Host " 5. Run one legal numerical configuration, or all tiles for one workload"
        Write-Host " 6. Run every legal numerical configuration from the JSON list"
        Write-Host " 7. Add a workload or tile candidate to the JSON list"
        Write-Host " 8. List and validate the current JSON configuration"
        Write-Host " 9. Open the JSON configuration in Notepad"
        Write-Host "10. Run the three-session tile performance experiment (seven standard workloads)"
        Write-Host "11. Run the load-latency hint experiment"
        Write-Host "12. Open the formal submission report"
        Write-Host " 0. Exit"
        $choice = Read-Host "Select a mode"
        try {
            switch ($choice) {
                "1" {
                    Invoke-MenuScript -ScriptPath $PSCommandPath -Arguments @("-Standard")
                }
                "2" {
                    Invoke-MenuScript -ScriptPath (Join-Path $menuRoot "scripts\mentor_guided.ps1")
                }
                "3" {
                    $old = $env:REQUIRE_CUTILE
                    try {
                        $env:REQUIRE_CUTILE = "1"
                        $log = Join-Path $menuLogDir ("gpu_tests_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
                        & $menuPy -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1 | Tee-Object -FilePath $log
                        if ($LASTEXITCODE -ne 0) { Write-Host "GPU TEST SUITE FAILED. Log: $log" -ForegroundColor Red }
                        else { Write-Host "GPU_TESTS=48 PASSED; LOG=$log" -ForegroundColor Green }
                    }
                    finally {
                        if ($null -eq $old) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
                        else { $env:REQUIRE_CUTILE = $old }
                    }
                }
                "4" {
                    $nodes = Get-RequiredGpuTestNodes
                    for ($i = 0; $i -lt $nodes.Count; $i++) {
                        Write-Host ("{0,2}. {1}" -f ($i + 1), $nodes[$i])
                    }
                    $selected = Read-Host "Enter the test number (1-$($nodes.Count)), or 0 to cancel"
                    if ($selected -eq "0") { break }
                    $index = 0
                    if (![int]::TryParse($selected, [ref]$index) -or $index -lt 1 -or $index -gt $nodes.Count) {
                        Write-Host "WARNING: test selection must be a number from 1 through $($nodes.Count)." -ForegroundColor Yellow
                        break
                    }
                    $node = $nodes[$index - 1]
                    Write-Host "PURPOSE: the pytest test name and source function identify the exact behavior being checked." -ForegroundColor Gray
                    $relative = ($node -split "::")[0] -replace "/", "\"
                    $func = (($node -split "::")[1] -split "\[")[0]
                    $source = Join-Path $menuRoot $relative
                    $sourceLines = Get-Content -LiteralPath $source -Encoding utf8
                    $start = -1
                    for ($i = 0; $i -lt $sourceLines.Count; $i++) {
                        if ($sourceLines[$i] -match ("^def " + [regex]::Escape($func) + "\s*\(")) { $start = $i; break }
                    }
                    if ($start -ge 0) {
                        $end = $sourceLines.Count - 1
                        for ($i = $start + 1; $i -lt $sourceLines.Count; $i++) {
                            if ($sourceLines[$i] -match "^def ") { $end = $i - 1; break }
                        }
                        for ($i = $start; $i -le $end; $i++) { Write-Host ("{0,4}: {1}" -f ($i + 1), $sourceLines[$i]) }
                    }
                    Read-Host "Review the test source. Press Enter to execute it"
                    $old = $env:REQUIRE_CUTILE
                    try {
                        $env:REQUIRE_CUTILE = "1"
                        & $menuPy -m pytest -v -ra $node
                        if ($LASTEXITCODE -eq 0) { Write-Host "SELECTED_TEST=PASS" -ForegroundColor Green }
                        else { Write-Host "SELECTED_TEST=FAIL" -ForegroundColor Red }
                    }
                    finally {
                        if ($null -eq $old) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
                        else { $env:REQUIRE_CUTILE = $old }
                    }
                }
                "5" {
                    Invoke-MenuPython -Arguments @($registryScript, "list", "--config", $configPath)
                    $cfg = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
                    $cases = @($cfg.cases)
                    for ($i = 0; $i -lt $cases.Count; $i++) {
                        Write-Host ("{0,2}. {1}  (B={2}, H={3}, N={4}, D={5})" -f ($i + 1),$cases[$i].name,$cases[$i].B,$cases[$i].H,$cases[$i].N,$cases[$i].D)
                    }
                    $selected = Read-Host "Choose a workload number, or 0 to cancel"
                    if ($selected -eq "0") { break }
                    $caseIndex = 0
                    if (![int]::TryParse($selected, [ref]$caseIndex) -or $caseIndex -lt 1 -or $caseIndex -gt $cases.Count) {
                        Write-Host "WARNING: invalid workload number; try again from the menu." -ForegroundColor Yellow
                        break
                    }
                    $case = $cases[$caseIndex - 1]
                    $mode = Read-Host "Enter 1 for one tile configuration, 2 for every legal tile for this workload"
                    $out = Join-Path $menuLogDir ("one_legal_test_{0}_{1}.json" -f $case.name,(Get-Date -Format "yyyyMMdd_HHmmss"))
                    if ($mode -eq "2") {
                        Invoke-MenuPython -Arguments @($matrixScript, "--config", $configPath, "--case", $case.name, "--output", $out)
                    }
                    elseif ($mode -eq "1") {
                        $tiles = @($cfg.tile_candidates | Where-Object { ($case.N % $_.tile_q) -eq 0 -and ($case.N % $_.tile_k) -eq 0 })
                        for ($i = 0; $i -lt $tiles.Count; $i++) { Write-Host ("{0,2}. {1}x{2}" -f ($i + 1),$tiles[$i].tile_q,$tiles[$i].tile_k) }
                        $tileChoice = Read-Host "Choose tile number, or 0 to cancel"
                        if ($tileChoice -eq "0") { break }
                        $tileIndex = 0
                        if (![int]::TryParse($tileChoice, [ref]$tileIndex) -or $tileIndex -lt 1 -or $tileIndex -gt $tiles.Count) {
                            Write-Host "WARNING: invalid tile selection." -ForegroundColor Yellow
                            break
                        }
                        $tile = $tiles[$tileIndex - 1]
                        Invoke-MenuPython -Arguments @($matrixScript, "--config", $configPath, "--case", $case.name, "--tile-q", "$($tile.tile_q)", "--tile-k", "$($tile.tile_k)", "--output", $out)
                    }
                    else { Write-Host "WARNING: choose 1 or 2." -ForegroundColor Yellow }
                }
                "6" {
                    $configValidation = & $menuPy $registryScript validate --config $configPath
                    if ($LASTEXITCODE -ne 0) { Write-Host ($configValidation -join [Environment]::NewLine) -ForegroundColor Red; break }
                    $out = Join-Path $menuLogDir ("all_legal_configurations_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
                    Invoke-MenuPython -Arguments @($matrixScript, "--config", $configPath, "--output", $out)
                }
                "7" {
                    Invoke-MenuPython -Arguments @($registryScript, "interactive", "--config", $configPath)
                }
                "8" {
                    Invoke-MenuPython -Arguments @($registryScript, "validate", "--config", $configPath)
                    Invoke-MenuPython -Arguments @($registryScript, "list", "--config", $configPath)
                }
                "9" {
                    if (!(Test-Path -LiteralPath $configPath)) { throw "Configuration file not found: $configPath" }
                    Start-Process notepad.exe -ArgumentList $configPath
                    Write-Host "After editing JSON, select menu option 8 to validate it before running tests."
                }
                "10" {
                    $out = Join-Path $menuLogDir ("tile_performance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
                    Invoke-MenuPython -Arguments @((Join-Path $menuRoot "benchmarks\benchmark_attention_sessions.py"), "--output", $out, "--warmup", "10", "--iterations", "30", "--repeats-per-sample", "5")
                }
                "11" {
                    $out = Join-Path $menuLogDir ("load_latency_hints_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
                    Invoke-MenuPython -Arguments @((Join-Path $menuRoot "benchmarks\benchmark_load_latency_hints.py"), "--output", $out, "--warmup", "10", "--iterations", "30", "--repeats-per-sample", "5", "--sessions", "3")
                }
                "12" {
                    $report = Join-Path $menuRoot "docs\SUBMISSION_REPORT_2026-10-22.md"
                    if (Test-Path -LiteralPath $report) { Invoke-Item $report }
                    else { Write-Host "Report was not found at $report" -ForegroundColor Yellow }
                }
                "0" { return }
                default { Write-Host "Choose one of the displayed menu numbers." -ForegroundColor Yellow }
            }
        }
        catch {
            Write-Host "Operation stopped: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "No local source file was overwritten by this menu operation." -ForegroundColor Gray
        }
    }
}

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
$smokeLog = Join-Path $logDir "walkthrough_smoke_$stamp.log"
$coreLog = Join-Path $logDir "walkthrough_core_tests_$stamp.log"
$edgeLog = Join-Path $logDir "walkthrough_edge_tests_$stamp.log"
$generateLog = Join-Path $logDir "walkthrough_generate_$stamp.log"
$cliLog = Join-Path $logDir "walkthrough_cli_$stamp.log"
$oracleLog = Join-Path $logDir "walkthrough_oracle_$stamp.log"
$matrixLog = Join-Path $logDir "walkthrough_numerical_matrix_$stamp.log"
$tileLog = Join-Path $logDir "walkthrough_tile_experiment_$stamp.log"
$hintLog = Join-Path $logDir "walkthrough_latency_hint_experiment_$stamp.log"

function Show-Step {
    param(
        [int]$Number,
        [string]$Title,
        [string]$Why,
        [string]$Command
    )
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host ("STEP {0}: {1}" -f $Number, $Title) -ForegroundColor Cyan
    Write-Host ("WHY: {0}" -f $Why) -ForegroundColor Gray
    Write-Host ("COMMAND: {0}" -f $Command) -ForegroundColor Yellow
    if (!$NoPause) {
        [void](Read-Host "Press Enter to execute this step (Ctrl+C cancels the walkthrough)")
    }
}

function Invoke-PythonChecked {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$LogPath
    )
    Write-Host ""
    Write-Host ("--- {0} ---" -f $Label) -ForegroundColor Cyan
    & $py @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code. Log: $LogPath" }
}

function Invoke-NativeChecked {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host ("--- {0} ---" -f $Label) -ForegroundColor Cyan
    & $Exe @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code" }
}

if ($Guided) {
    $guidedScript = Join-Path $repoRoot "scripts\mentor_guided.ps1"
    if (!(Test-Path -LiteralPath $guidedScript)) { throw "Guided mode script not found: $guidedScript" }
    $guidedArgs = @()
    if ($NoPause) { $guidedArgs += "-NoPause" }
    if ($SkipTileExperiment) { $guidedArgs += "-SkipTileExperiment" }
    if ($SkipLatencyHintExperiment) { $guidedArgs += "-SkipLatencyHintExperiment" }
    & $guidedScript @guidedArgs
    return
}

Push-Location $repoRoot
try {
    Show-Step -Number 1 -Title "Environment and GPU preflight" -Why "Record which Python environment and physical GPU are being used. A green test on the wrong interpreter or without the expected GPU would not validate the intended setup." -Command "$py --version; nvidia-smi; nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv"
    Write-Host "REPO=$repoRoot"
    Write-Host "PYTHON_EXE=$py"
    Invoke-NativeChecked "Python version" $py @("--version")
    Invoke-NativeChecked "NVIDIA GPU inventory" "nvidia-smi" @()
    Invoke-NativeChecked "Structured GPU inventory" "nvidia-smi" @("--query-gpu=name,driver_version,compute_cap,memory.total,memory.free","--format=csv")

    Show-Step -Number 2 -Title "Genuine cuTile runtime check" -Why "Separate install/compiler/runtime/device problems from attention-algorithm bugs. This compiles and launches a tiny real cuTile vector-add kernel; it is not an attention speed test." -Command "$py scripts\cutile_smoke.py"
    Invoke-PythonChecked "Genuine cuTile runtime check" @((Join-Path $repoRoot "scripts\cutile_smoke.py")) $smokeLog
    if (!(Select-String -Path $smokeLog -SimpleMatch "CUTILE_SMOKE=PASS" -Quiet)) {
        throw "Runtime check did not report success. Log: $smokeLog"
    }

    $oldRequireCutile = $env:REQUIRE_CUTILE
    try {
        $env:REQUIRE_CUTILE = "1"

        Show-Step -Number 3 -Title "Core correctness suite (34 executions)" -Why "Checks each legal mandatory case/tile pair against an independent FP32 NumPy oracle, plus numerical stability, causal invariance, an exploratory tile and invalid sequence handling." -Command "$py -m pytest -v -ra tests\test_gpu_correctness.py"
        Invoke-PythonChecked "Core correctness tests" @("-m","pytest","-v","-ra",(Join-Path $repoRoot "tests\test_gpu_correctness.py")) $coreLog

        Show-Step -Number 4 -Title "Edge-case suite (14 executions)" -Why "Targets invariants that can pass ordinary random comparisons by accident: constant values, uniform-score prefix means, future-key/value isolation, invalid inputs, layout and dimension contracts." -Command "$py -m pytest -v -ra tests\test_gpu_edge_cases.py"
        Invoke-PythonChecked "Edge-case tests" @("-m","pytest","-v","-ra",(Join-Path $repoRoot "tests\test_gpu_edge_cases.py")) $edgeLog
    }
    finally {
        if ($null -eq $oldRequireCutile) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
        else { $env:REQUIRE_CUTILE = $oldRequireCutile }
    }

    $inputFile = Join-Path $logDir "walkthrough_attention_input_$stamp.npz"
    $outputFile = Join-Path $logDir "walkthrough_attention_output_$stamp.npy"

    Show-Step -Number 5 -Title "Generate deterministic course-format input" -Why "A fixed seed makes the demo reproducible and lets the output be checked against exactly the same Q/K/V tensors. The input file is saved so the CLI output can be independently audited." -Command "$py scripts\generate_inputs.py --output <saved-input.npz> --batch 1 --heads 1 --seq 64 --head-dim 32 --seed 20261018"
    Invoke-PythonChecked "Generate CLI input" @(
        (Join-Path $repoRoot "scripts\generate_inputs.py"),
        "--output",$inputFile,"--batch","1","--heads","1","--seq","64","--head-dim","32","--seed","20261018"
    ) $generateLog
    if (!(Test-Path -LiteralPath $inputFile)) { throw "Input generator did not create $inputFile" }

    Show-Step -Number 6 -Title "Run the course-facing cuTile CLI" -Why "Verifies the real user interface, not just a Python function call. It loads the NPZ input, launches cuTile with the default 64x64 tile and writes a NumPy output file." -Command "$py run_attention.py --input <saved-input.npz> --output <saved-output.npy> --backend cutile --tile-q 64 --tile-k 64"
    Invoke-PythonChecked "Course-facing cuTile CLI" @(
        (Join-Path $repoRoot "run_attention.py"),
        "--input",$inputFile,"--output",$outputFile,"--backend","cutile","--tile-q","64","--tile-k","64"
    ) $cliLog
    if (!(Test-Path -LiteralPath $outputFile)) { throw "CLI did not create $outputFile" }

    Show-Step -Number 7 -Title "Independently check CLI output against NumPy" -Why "Verifies output shape, FP16 dtype, finiteness and mixed absolute/relative tolerance against the independent CPU oracle. The kernel is not allowed to generate its own expected answer." -Command "$py scripts\verify_cli_output.py --input <saved-input.npz> --output <saved-output.npy>"
    Invoke-PythonChecked "CLI versus independent NumPy oracle" @(
        (Join-Path $repoRoot "scripts\verify_cli_output.py"),
        "--input",$inputFile,"--output",$outputFile
    ) $oracleLog

    if (!$SkipNumericalMatrix) {
        $matrixFile = Join-Path $logDir "walkthrough_legal_numerical_matrix_$stamp.json"
        Show-Step -Number 8 -Title "Run all 29 legal numerical case/tile checks" -Why "Performs a fresh output-versus-NumPy check for every supported case/tile pair. It reports each row and writes JSON/CSV. Divisibility and power-of-two restrictions define the legal matrix; an illegal configuration is not silently counted as a pass." -Command "$py scripts\mentor_numerical_matrix.py --output <matrix.json>"
        Invoke-PythonChecked "29-pair numerical matrix" @(
            (Join-Path $repoRoot "scripts\mentor_numerical_matrix.py"),
            "--output",$matrixFile
        ) $matrixLog
    }
    else {
        Write-Host "NUMERICAL_MATRIX=SKIPPED by -SkipNumericalMatrix"
    }

    if (!$SkipTileExperiment) {
        $tileFile = Join-Path $logDir "walkthrough_full_tile_sweep_$stamp.json"
        $tileArgs = @(
            (Join-Path $repoRoot "benchmarks\benchmark_attention_sessions.py"),
            "--output",$tileFile,
            "--warmup","10","--iterations","30","--repeats-per-sample","5"
        )
        Show-Step -Number 9 -Title "Run the full 7-workload, three-session tile experiment (29 legal rows)" -Why "Changes tile shape while holding each case's seeded Q/K/V fixed. Ten warmups reduce startup effects; 30 CUDA-event samples show the distribution; five launches per sample improve event timing granularity. The driver rotates/reverses tile order across three sessions, and baseline timings bracket each case's tile sweep to expose drift." -Command "$py benchmarks\benchmark_attention_sessions.py --output <aggregate.json> --warmup 10 --iterations 30 --repeats-per-sample 5"
        Invoke-PythonChecked "Three-session tile experiment" $tileArgs $tileLog
        if (!(Test-Path -LiteralPath $tileFile)) { throw "Tile experiment did not create $tileFile" }
    }
    else {
        Write-Host "TILE_EXPERIMENT=SKIPPED by -SkipTileExperiment"
    }

    if (!$SkipLatencyHintExperiment) {
        $hintFile = Join-Path $logDir "walkthrough_latency_hints_$stamp.json"
        Show-Step -Number 10 -Title "Run the DRAM load-latency hint experiment (36 trials)" -Why "Isolates one scheduling hint at a time on two fixed workloads. Each hint must pass independent NumPy tolerance before timing. Three rotated sessions help prevent one fixed execution order from deciding the result. This stage intentionally does not launch Nsight Compute." -Command "$py benchmarks\benchmark_load_latency_hints.py --output <hints.json> --warmup 10 --iterations 30 --repeats-per-sample 5 --sessions 3"
        Invoke-PythonChecked "DRAM load-latency hint experiment" @(
            (Join-Path $repoRoot "benchmarks\benchmark_load_latency_hints.py"),
            "--output",$hintFile,"--warmup","10","--iterations","30","--repeats-per-sample","5","--sessions","3"
        ) $hintLog
    }
    else {
        Write-Host "LATENCY_HINT_EXPERIMENT=SKIPPED by -SkipLatencyHintExperiment"
    }

    Write-Host ""
    Write-Host "WALKTHROUGH_CORE=PASS" -ForegroundColor Green
    Write-Host "LOG_DIR=$logDir"
    Write-Host "SMOKE_LOG=$smokeLog"
    Write-Host "CORE_TEST_LOG=$coreLog"
    Write-Host "EDGE_TEST_LOG=$edgeLog"
    Write-Host "INPUT_NPZ=$inputFile"
    Write-Host "OUTPUT_NPY=$outputFile"
    if (!$SkipNumericalMatrix) { Write-Host "NUMERICAL_MATRIX_JSON=$matrixFile" }
    if (!$SkipTileExperiment) { Write-Host "TILE_SWEEP_JSON=$tileFile" }
    if (!$SkipLatencyHintExperiment) { Write-Host "LATENCY_HINT_JSON=$hintFile" }
    Write-Host "Nsight Compute is intentionally excluded from this walkthrough."
    Write-Host "Use docs\MENTOR_STEP_BY_STEP_WALKTHROUGH.md for command rationale and recorded historical results."
}
finally {
    Pop-Location
}
