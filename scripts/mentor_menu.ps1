[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$menuRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonCandidates = @(
    (Join-Path $menuRoot '.venv-cutile-windows\Scripts\python.exe'),
    'C:\Users\Public\GPU-cutile-project\gpu-nvidia\.venv-cutile-windows\Scripts\python.exe'
)
$menuPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$menuPython) { throw "Project cuTile Python was not found. Checked: $($pythonCandidates -join '; ')" }

$logDirectory = Join-Path $menuRoot '.tmp\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$configPath = Join-Path $menuRoot 'tests\legal_attention_configs.json'
$registryScript = Join-Path $menuRoot 'scripts\legal_config_registry.py'
$matrixScript = Join-Path $menuRoot 'scripts\mentor_numerical_matrix.py'
$mentorStepScript = Join-Path $menuRoot 'scripts\mentor_step_by_step.ps1'
$mentorGuidedScript = Join-Path $menuRoot 'scripts\mentor_guided.ps1'
$mentorDemoScript = Join-Path $menuRoot 'scripts\mentor_demo.ps1'

function Get-RequiredGpuTestNodes {
    $oldRequireCutile = $env:REQUIRE_CUTILE
    try {
        $env:REQUIRE_CUTILE = '1'
        $collected = & $menuPython -m pytest --collect-only -q tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1
        $collectExit = $LASTEXITCODE
        if ($collectExit -ne 0) {
            throw "GPU pytest collection failed (exit $collectExit): $($collected -join ' | ')"
        }
        $nodes = @(
            $collected |
            ForEach-Object { $_.ToString().Trim() } |
            Where-Object { $_ -match '^tests[/\\]test_gpu_(correctness|edge_cases)\.py::test_' }
        )
        if ($nodes.Count -eq 0) { throw 'Pytest collected zero required GPU test nodes.' }
        return ,$nodes
    }
    finally {
        if ($null -eq $oldRequireCutile) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
        else { $env:REQUIRE_CUTILE = $oldRequireCutile }
    }
}

function Invoke-ChildPowerShell {
    param([string]$Label, [string]$ScriptPath, [string[]]$Arguments = @())
    if (!(Test-Path -LiteralPath $ScriptPath)) { throw "$Label script not found: $ScriptPath" }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode." }
}

Push-Location $menuRoot
try {
    $leaveMenu = $false
    while (!$leaveMenu) {
        Write-Host ''
        Write-Host '================================================================================' -ForegroundColor DarkGray
        Write-Host 'CUTILE ATTENTION - MENTOR DEMONSTRATION MENU' -ForegroundColor Cyan
        Write-Host ('Repository: ' + $menuRoot)
        Write-Host ('Python:     ' + $menuPython)
        Write-Host ''
        Write-Host ' 1. Standard ten-step walkthrough (GPU, tests, CLI, matrix, benchmarks)'
        Write-Host ' 2. Guided kernel tour with source explanations and one-by-one GPU tests'
        Write-Host ' 3. Run the complete required native cuTile GPU test suite'
        Write-Host ' 4. Select and run one collected native cuTile GPU test'
        Write-Host ' 5. Run every legal workload/tile numerical configuration'
        Write-Host ' 6. Run the three-session tile performance experiment'
        Write-Host ' 7. Run the load-latency hint experiment'
        Write-Host ' 8. Run configuration-registry unit tests (CPU)'
        Write-Host ' 9. Add a workload or tile candidate to the registry'
        Write-Host '10. Validate and list the configuration registry'
        Write-Host '11. Open the current formal submission report'
        Write-Host '12. Run the reusable end-to-end mentor demo'
        Write-Host '13. Run the mentor demo with an Nsight Compute capture attempt'
        Write-Host ' 0. Exit'
        $choice = Read-Host 'Select a mode'

        try {
            switch ($choice) {
                '1' {
                    Invoke-ChildPowerShell -Label 'Standard walkthrough' -ScriptPath $mentorStepScript -Arguments @('-Standard')
                }
                '2' {
                    Invoke-ChildPowerShell -Label 'Guided walkthrough' -ScriptPath $mentorGuidedScript
                }
                '3' {
                    $nodes = Get-RequiredGpuTestNodes
                    $oldRequireCutile = $env:REQUIRE_CUTILE
                    try {
                        $env:REQUIRE_CUTILE = '1'
                        $testLog = Join-Path $logDirectory ('gpu_tests_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
                        Write-Host ("Running {0} collected tests. Missing cuTile is treated as an error." -f $nodes.Count) -ForegroundColor Yellow
                        & $menuPython -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1 | Tee-Object -FilePath $testLog
                        $exitCode = $LASTEXITCODE
                        if ($exitCode -ne 0) { throw "GPU test suite failed with exit code $exitCode. Log: $testLog" }
                        Write-Host ("GPU_TESTS={0} PASSED; LOG={1}" -f $nodes.Count,$testLog) -ForegroundColor Green
                    }
                    finally {
                        if ($null -eq $oldRequireCutile) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
                        else { $env:REQUIRE_CUTILE = $oldRequireCutile }
                    }
                }
                '4' {
                    $nodes = Get-RequiredGpuTestNodes
                    for ($i = 0; $i -lt $nodes.Count; $i++) {
                        Write-Host (($i + 1).ToString().PadLeft(3) + '. ' + $nodes[$i])
                    }
                    $selectedText = Read-Host ("Enter a test number from 1 to {0}, or 0 to cancel" -f $nodes.Count)
                    if ($selectedText -eq '0') { break }
                    $selectedNumber = 0
                    if (![int]::TryParse($selectedText, [ref]$selectedNumber) -or $selectedNumber -lt 1 -or $selectedNumber -gt $nodes.Count) {
                        Write-Host 'Invalid test number; no test was run.' -ForegroundColor Yellow
                        break
                    }
                    $oldRequireCutile = $env:REQUIRE_CUTILE
                    try {
                        $env:REQUIRE_CUTILE = '1'
                        $testLog = Join-Path $logDirectory ('selected_gpu_test_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
                        & $menuPython -m pytest -v -ra $nodes[$selectedNumber - 1] 2>&1 | Tee-Object -FilePath $testLog
                        $exitCode = $LASTEXITCODE
                        if ($exitCode -ne 0) { throw "Selected test failed with exit code $exitCode. Log: $testLog" }
                        Write-Host ("SELECTED_TEST=PASS; NODE={0}; LOG={1}" -f $nodes[$selectedNumber - 1],$testLog) -ForegroundColor Green
                    }
                    finally {
                        if ($null -eq $oldRequireCutile) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
                        else { $env:REQUIRE_CUTILE = $oldRequireCutile }
                    }
                }
                '5' {
                    $validation = & $menuPython $registryScript validate --config $configPath 2>&1
                    $validation | ForEach-Object { Write-Host $_ }
                    if ($LASTEXITCODE -ne 0) { throw 'Configuration registry validation failed.' }
                    $matrixJson = Join-Path $logDirectory ('numerical_matrix_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
                    & $menuPython $matrixScript --config $configPath --output $matrixJson
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw "Numerical matrix failed with exit code $exitCode." }
                    if (!(Test-Path -LiteralPath $matrixJson)) { throw "Numerical matrix output is missing: $matrixJson" }
                    $summary = Get-Content -Raw -LiteralPath $matrixJson | ConvertFrom-Json
                    if ($summary.fail_count -ne 0 -or $summary.pass_count -ne $summary.legal_pair_count) {
                        throw "Numerical matrix contains failures or incomplete results: legal=$($summary.legal_pair_count), passed=$($summary.pass_count), failed=$($summary.fail_count)."
                    }
                    Write-Host ("NUMERICAL_MATRIX=PASS; CONFIGURATIONS={0}; OUTPUT={1}" -f $summary.legal_pair_count,$matrixJson) -ForegroundColor Green
                }
                '6' {
                    $benchmarkScript = Join-Path $menuRoot 'benchmarks\benchmark_attention_sessions.py'
                    $benchmarkJson = Join-Path $logDirectory ('tile_performance_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
                    & $menuPython $benchmarkScript --output $benchmarkJson --warmup 10 --iterations 30 --repeats-per-sample 5
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw "Tile performance experiment failed with exit code $exitCode." }
                    if (!(Test-Path -LiteralPath $benchmarkJson)) { throw "Tile performance output is missing: $benchmarkJson" }
                    Write-Host ('TILE_EXPERIMENT=PASS; OUTPUT=' + $benchmarkJson) -ForegroundColor Green
                }
                '7' {
                    $hintScript = Join-Path $menuRoot 'benchmarks\benchmark_load_latency_hints.py'
                    $hintJson = Join-Path $logDirectory ('latency_hints_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
                    & $menuPython $hintScript --output $hintJson --warmup 10 --iterations 30 --repeats-per-sample 5 --sessions 3
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw "Load-latency experiment failed with exit code $exitCode." }
                    if (!(Test-Path -LiteralPath $hintJson)) { throw "Load-latency output is missing: $hintJson" }
                    Write-Host ('LATENCY_HINT_EXPERIMENT=PASS; OUTPUT=' + $hintJson) -ForegroundColor Green
                }
                '8' {
                    & $menuPython -m pytest -v -ra tests\test_legal_config_registry.py
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw "Registry unit tests failed with exit code $exitCode." }
                    Write-Host 'CONFIG_REGISTRY_TESTS=PASS' -ForegroundColor Green
                }
                '9' {
                    & $menuPython $registryScript interactive --config $configPath
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw "Interactive registry editor failed with exit code $exitCode." }
                }
                '10' {
                    & $menuPython $registryScript validate --config $configPath
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw 'Configuration registry validation failed.' }
                    & $menuPython $registryScript list --config $configPath
                    $exitCode = $LASTEXITCODE
                    if ($exitCode -ne 0) { throw 'Configuration registry listing failed.' }
                }
                '11' {
                    $reportPath = Join-Path $menuRoot 'docs\SUBMISSION_REPORT_2026-11-03.md'
                    if (!(Test-Path -LiteralPath $reportPath)) { throw "Current report not found: $reportPath" }
                    Invoke-Item $reportPath
                    Write-Host ('OPENED_REPORT=' + $reportPath) -ForegroundColor Green
                }
                '12' {
                    Invoke-ChildPowerShell -Label 'Reusable mentor demo' -ScriptPath $mentorDemoScript
                }
                '13' {
                    Invoke-ChildPowerShell -Label 'Mentor demo with Nsight Compute' -ScriptPath $mentorDemoScript -Arguments @('-SkipTileSweep','-WithProfile')
                }
                '0' { $leaveMenu = $true }
                default { Write-Host 'Choose one of the displayed menu numbers.' -ForegroundColor Yellow }
            }
        }
        catch {
            Write-Host ('Operation stopped: ' + $_.Exception.Message) -ForegroundColor Red
        }
    }
}
finally {
    Pop-Location
}
