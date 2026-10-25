[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$menuRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonCandidates = @(
    (Join-Path $menuRoot '.venv-cutile-windows\Scripts\python.exe'),
    'C:\Users\Public\GPU-cutile-project\gpu-nvidia\.venv-cutile-windows\Scripts\python.exe'
)
$menuPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$menuPython) { throw 'Project cuTile Python was not found.' }

$logDirectory = Join-Path $menuRoot '.tmp\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$configPath = Join-Path $menuRoot 'tests\legal_attention_configs.json'
$registryScript = Join-Path $menuRoot 'scripts\legal_config_registry.py'
$matrixScript = Join-Path $menuRoot 'scripts\mentor_numerical_matrix.py'

while ($true) {
    Write-Host ''
    Write-Host '============================================================' -ForegroundColor DarkGray
    Write-Host 'CUTILE ATTENTION - MENTOR DEMONSTRATION MENU' -ForegroundColor Cyan
    Write-Host ('Python: ' + $menuPython)
    Write-Host ''
    Write-Host ' 1. Standard step-by-step demo'
    Write-Host ' 2. Guided kernel explanation and individual test runs'
    Write-Host ' 3. Run all 48 required GPU tests'
    Write-Host ' 4. Select and run one of the 48 GPU tests'
    Write-Host ' 5. Run all legal numerical configurations'
    Write-Host ' 6. Run the tile performance experiment'
    Write-Host ' 7. Run the load-latency hint experiment'
    Write-Host ' 8. Run configuration-registry unit tests'
    Write-Host ' 9. Add a workload or tile configuration'
    Write-Host '10. Validate and list the configuration registry'
    Write-Host '11. Open the formal submission report'
    Write-Host ' 0. Exit'
    $choice = Read-Host 'Select a mode'

    try {
        switch ($choice) {
            '1' {
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'mentor_step_by_step.ps1') -Standard
                break
            }
            '2' {
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'mentor_guided.ps1')
                break
            }
            '3' {
                $oldRequireCutile = $env:REQUIRE_CUTILE
                try {
                    $env:REQUIRE_CUTILE = '1'
                    $testLog = Join-Path $logDirectory ('gpu_tests_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
                    & $menuPython -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1 | Tee-Object -FilePath $testLog
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host ('GPU_TESTS=48 PASSED; LOG=' + $testLog) -ForegroundColor Green
                    }
                    else {
                        Write-Host ('GPU_TESTS=FAILED; LOG=' + $testLog) -ForegroundColor Red
                    }
                }
                finally {
                    if ($null -eq $oldRequireCutile) {
                        Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue
                    }
                    else {
                        $env:REQUIRE_CUTILE = $oldRequireCutile
                    }
                }
                break
            }
            '4' {
                $collected = & $menuPython -m pytest --collect-only -q tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1
                if ($LASTEXITCODE -ne 0) { throw 'Pytest collection failed; see output above.' }
                $nodes = @($collected | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ -match '^tests[/\\]test_gpu_(correctness|edge_cases)\.py::test_' })
                if ($nodes.Count -ne 48) { throw ('Expected 48 test nodes but found ' + $nodes.Count + '.') }
                for ($i = 0; $i -lt $nodes.Count; $i++) {
                    Write-Host (($i + 1).ToString().PadLeft(2) + '. ' + $nodes[$i])
                }
                $selectedText = Read-Host 'Enter a test number from 1 to 48, or 0 to cancel'
                if ($selectedText -eq '0') { break }
                $selectedNumber = 0
                if (![int]::TryParse($selectedText, [ref]$selectedNumber) -or $selectedNumber -lt 1 -or $selectedNumber -gt $nodes.Count) {
                    Write-Host 'Invalid test number.' -ForegroundColor Yellow
                    break
                }
                $oldRequireCutile = $env:REQUIRE_CUTILE
                try {
                    $env:REQUIRE_CUTILE = '1'
                    & $menuPython -m pytest -v -ra $nodes[$selectedNumber - 1]
                    if ($LASTEXITCODE -eq 0) { Write-Host 'SELECTED_TEST=PASS' -ForegroundColor Green }
                    else { Write-Host 'SELECTED_TEST=FAIL' -ForegroundColor Red }
                }
                finally {
                    if ($null -eq $oldRequireCutile) {
                        Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue
                    }
                    else {
                        $env:REQUIRE_CUTILE = $oldRequireCutile
                    }
                }
                break
            }
            '5' {
                $validation = & $menuPython $registryScript validate --config $configPath
                if ($LASTEXITCODE -ne 0) { throw 'Configuration validation failed.' }
                $matrixLog = Join-Path $logDirectory ('numerical_matrix_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
                & $menuPython $matrixScript --config $configPath --output $matrixLog
                if ($LASTEXITCODE -eq 0) { Write-Host ('NUMERICAL_MATRIX=PASS; OUTPUT=' + $matrixLog) -ForegroundColor Green }
                else { Write-Host ('NUMERICAL_MATRIX=FAIL; OUTPUT=' + $matrixLog) -ForegroundColor Red }
                break
            }
            '6' {
                $benchmarkScript = Join-Path $menuRoot 'benchmarks\benchmark_attention_sessions.py'
                $benchmarkLog = Join-Path $logDirectory ('tile_performance_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
                & $menuPython $benchmarkScript --output $benchmarkLog --warmup 10 --iterations 30 --repeats-per-sample 5
                if ($LASTEXITCODE -eq 0) { Write-Host ('TILE_EXPERIMENT=PASS; OUTPUT=' + $benchmarkLog) -ForegroundColor Green }
                else { Write-Host ('TILE_EXPERIMENT=FAIL; OUTPUT=' + $benchmarkLog) -ForegroundColor Red }
                break
            }
            '7' {
                $hintScript = Join-Path $menuRoot 'benchmarks\benchmark_load_latency_hints.py'
                $hintLog = Join-Path $logDirectory ('latency_hints_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
                & $menuPython $hintScript --output $hintLog --warmup 10 --iterations 30 --repeats-per-sample 5 --sessions 3
                if ($LASTEXITCODE -eq 0) { Write-Host ('LATENCY_HINT_EXPERIMENT=PASS; OUTPUT=' + $hintLog) -ForegroundColor Green }
                else { Write-Host ('LATENCY_HINT_EXPERIMENT=FAIL; OUTPUT=' + $hintLog) -ForegroundColor Red }
                break
            }
            '8' {
                & $menuPython -m pytest -v -ra tests\test_legal_config_registry.py
                if ($LASTEXITCODE -eq 0) { Write-Host 'CONFIG_REGISTRY_TESTS=PASS' -ForegroundColor Green }
                else { Write-Host 'CONFIG_REGISTRY_TESTS=FAIL' -ForegroundColor Red }
                break
            }
            '9' {
                & $menuPython $registryScript interactive --config $configPath
                break
            }
            '10' {
                & $menuPython $registryScript validate --config $configPath
                if ($LASTEXITCODE -eq 0) { & $menuPython $registryScript list --config $configPath }
                break
            }
            '11' {
                $reportPath = Join-Path $menuRoot 'docs\SUBMISSION_REPORT_2026-10-22.md'
                if (Test-Path -LiteralPath $reportPath) { Invoke-Item $reportPath }
                else { Write-Host ('Report not found: ' + $reportPath) -ForegroundColor Yellow }
                break
            }
            '0' { return }
            default { Write-Host 'Choose one of the displayed menu numbers.' -ForegroundColor Yellow }
        }
    }
    catch {
        Write-Host ('Operation stopped: ' + $_.Exception.Message) -ForegroundColor Red
    }
}
