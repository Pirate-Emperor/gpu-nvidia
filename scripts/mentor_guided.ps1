[CmdletBinding()]
param(
    [switch]$NoPause,
    [switch]$SkipTileExperiment,
    [switch]$SkipLatencyHintExperiment
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
$guideLog = Join-Path $logDir "guided_48_tests_$stamp.log"
$matrixLog = Join-Path $logDir "guided_29_configurations_$stamp.log"
$codePath = Join-Path $repoRoot "src\cutile_attention.py"
$coreTestPath = Join-Path $repoRoot "tests\test_gpu_correctness.py"
$edgeTestPath = Join-Path $repoRoot "tests\test_gpu_edge_cases.py"

function Pause-BeforeExecution {
    param([string]$Prompt = "Press Enter to continue")
    if (!$NoPause) { [void](Read-Host $Prompt) }
}

function Show-WalkthroughStep {
    param([string]$Title, [string]$Explanation)
    Write-Host ""
    Write-Host ("=" * 88) -ForegroundColor DarkGray
    Write-Host $Title -ForegroundColor Cyan
    Write-Host $Explanation -ForegroundColor Gray
}

function Show-CodeRange {
    param(
        [string]$Path,
        [string]$Title,
        [string]$StartPattern,
        [string]$EndPattern
    )
    $lines = Get-Content -LiteralPath $Path -Encoding utf8
    $start = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $StartPattern) { $start = $i; break }
    }
    if ($start -lt 0) { throw "Could not find code section '$Title' in $Path using pattern $StartPattern" }
    $end = $lines.Count - 1
    for ($i = $start + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $EndPattern) { $end = $i; break }
    }
    Write-Host ""
    Write-Host ("--- SOURCE: {0} ---" -f $Title) -ForegroundColor Yellow
    Write-Host ("FILE: {0}" -f (Resolve-Path $Path).Path) -ForegroundColor DarkGray
    for ($i = $start; $i -le $end; $i++) {
        Write-Host ("{0,4}: {1}" -f ($i + 1), $lines[$i])
    }
}

function Show-TestFunction {
    param([string]$NodeId)
    if ($NodeId -notmatch '^(tests[/\\].+?\.py)::(test_[A-Za-z0-9_]+)') {
        throw "Cannot parse pytest node ID: $NodeId"
    }
    $relativePath = $Matches[1] -replace '/', '\'
    $functionName = $Matches[2]
    $path = Join-Path $repoRoot $relativePath
    $lines = Get-Content -LiteralPath $path -Encoding utf8
    $start = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match ("^def " + [regex]::Escape($functionName) + "\s*\(")) {
            $start = $i
            break
        }
    }
    if ($start -lt 0) { throw "Cannot locate function $functionName in $path" }
    $end = $lines.Count - 1
    for ($i = $start + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^def ') { $end = $i - 1; break }
    }
    Write-Host ""
    Write-Host ("TEST SOURCE: {0}" -f $NodeId) -ForegroundColor Yellow
    Write-Host ("FILE: {0}" -f $path) -ForegroundColor DarkGray
    for ($i = $start; $i -le $end; $i++) {
        Write-Host ("{0,4}: {1}" -f ($i + 1), $lines[$i])
    }
    Write-Host ("PURPOSE: {0}" -f (Get-TestExplanation $functionName)) -ForegroundColor Gray
}

function Get-TestExplanation {
    param([string]$Name)
    switch -Regex ($Name) {
        "matches_independent_reference" { return "Compare this case and tile with the independent FP32 NumPy calculation; checks numerical agreement, output shape, dtype and finite values." }
        "exploratory_tile_configuration" { return "Check an additional supported asymmetric tile against the same independent numerical reference." }
        "numerical_stability" { return "Exercise small and large attention scores to check for unstable exponentials or non-finite output." }
        "causal_future_value_invariance" { return "Change values at future positions and require outputs for earlier positions to remain exactly unchanged." }
        "rejects_nondivisible_sequence" { return "Confirm an input length that does not meet the fixed-tile contract is rejected before kernel execution." }
        "constant_v_is_preserved" { return "A weighted average of identical value vectors must reproduce that constant vector." }
        "zero_logits_equal_causal_prefix_average" { return "When all query/key scores are zero, probabilities are uniform over the allowed causal prefix; output must equal its prefix average." }
        "tile_configurations_agree" { return "Compare several tile shapes on the same data to detect tile-dependent algorithm errors." }
        "future_values_cannot_change" { return "Check that changing future value vectors cannot affect earlier query outputs." }
        "future_keys_cannot_change" { return "Check that changing future key vectors cannot affect earlier query outputs." }
        "nonfinite_input_is_rejected" { return "Check that NaN and positive or negative infinity are rejected before the GPU calculation." }
        "noncontiguous_input_is_rejected" { return "Check that an unsupported memory layout is rejected instead of being interpreted incorrectly." }
        "unsupported_or_mismatched_value_shape" { return "Check that unsupported head dimensions and mismatched value dimensions are rejected." }
        "nondivisible_sequence_is_rejected_for_each_tile_dimension" { return "Check that invalid sequence lengths are rejected for each tested query/key tile arrangement." }
        default { return "Inspect the assertion in this function; it states the exact behavior this test requires." }
    }
}

function Invoke-GuidedNode {
    param([string]$NodeId, [string]$Label, [string]$LogPath)
    Show-WalkthroughStep -Title $Label -Explanation ("Exact test: " + $NodeId)
    Show-TestFunction -NodeId $NodeId
    Pause-BeforeExecution -Prompt "Read the test source and purpose. Press Enter to run this exact test."
    Write-Host ("RUNNING: {0} -m pytest -v -ra {1}" -f $py, $NodeId) -ForegroundColor Green
    & $py -m pytest -v -ra $NodeId 2>&1 | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Test failed: $NodeId (exit code $exitCode). Inspect $LogPath" }
}

function Find-NodeId {
    param([string]$Pattern)
    $node = $script:nodeIds | Where-Object { $_ -match $Pattern } | Select-Object -First 1
    if (!$node) { throw "No collected test node matched: $Pattern" }
    return $node
}

Push-Location $repoRoot
$oldRequireCutile = $env:REQUIRE_CUTILE
try {
    Write-Host "GUIDED MODE: each source section is shown before its evidence test runs." -ForegroundColor Green
    Write-Host "GPU: the native Windows RTX 4060 is required. Nsight Compute is not part of this session."
    Write-Host "Logs: $logDir"
    Write-Host "Python: $py"
    Pause-BeforeExecution -Prompt "Press Enter to check the active GPU and cuTile runtime."

    & $py --version
    if ($LASTEXITCODE -ne 0) { throw "Project Python failed to run" }
    nvidia-smi
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed" }
    & $py (Join-Path $repoRoot "scripts\cutile_smoke.py") 2>&1 | Tee-Object -FilePath (Join-Path $logDir "guided_runtime_check_$stamp.log")
    if ($LASTEXITCODE -ne 0) { throw "cuTile runtime check failed" }

    $env:REQUIRE_CUTILE = "1"
    $collectOutput = & $py -m pytest --collect-only -q tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py 2>&1
    $collectExit = $LASTEXITCODE
    if ($collectExit -ne 0) { throw "Could not collect GPU tests: $($collectOutput -join ' | ')" }
    $script:nodeIds = @(
        $collectOutput |
        ForEach-Object { $_.ToString().Trim() } |
        Where-Object { $_ -match '^tests[/\\]test_gpu_(correctness|edge_cases)\.py::test_' }
    )
    if ($script:nodeIds.Count -ne 48) {
        throw "Expected exactly 48 required GPU test executions, but pytest collected $($script:nodeIds.Count). Review the collection output before presenting."
    }
    Write-Host "COLLECTED_REQUIRED_GPU_TESTS=$($script:nodeIds.Count)" -ForegroundColor Green

    Show-WalkthroughStep "KERNEL SECTION 1 — Input and tile contract" "The Python entry path rejects unsupported dimensions, bad layouts and sequence lengths that cannot be divided into the selected tiles. These checks prevent a kernel launch with assumptions the implementation cannot satisfy."
    Show-CodeRange -Path $codePath -Title "Input/tile validation" -StartPattern '^\s*def _validate_cutile_config\(' -EndPattern '^\s*def launch_device\('
    $node = Find-NodeId 'test_nondivisible_sequence_is_rejected_for_each_tile_dimension'
    Invoke-GuidedNode -NodeId $node -Label "Validate the input/tile contract" -LogPath $guideLog

    Show-WalkthroughStep "KERNEL SECTION 2 — Specialization and launch grid" "The kernel is specialized for the chosen tile sizes and head dimensions. The grid assigns one program to each batch item, attention head and query block. The launch test below checks that this setup produces the expected output for an actual workload."
    Show-CodeRange -Path $codePath -Title "Output allocation, kernel specialization and grid launch" -StartPattern '^\s*out_gpu = cp\.empty\(' -EndPattern '^\s*return out_gpu'
    $node = Find-NodeId 'test_cutile_matches_independent_reference.*long_d128-tile64x64'
    Invoke-GuidedNode -NodeId $node -Label "Validate kernel launch and output" -LogPath $guideLog

    Show-WalkthroughStep "KERNEL SECTION 3 — Query tile and causal loop bound" "The kernel loads the query block and initializes the running softmax state. The key/value loop stops after the final tile that could contribute to this query block. Entirely future key/value tiles are therefore skipped."
    Show-CodeRange -Path $codePath -Title "Query tile, softmax state and key/value loop bound" -StartPattern '^\s*def kernel\(' -EndPattern '^\s*for kb in range\(nkt\):'
    $node = Find-NodeId 'test_cutile_causal_future_value_invariance'
    Invoke-GuidedNode -NodeId $node -Label "Validate causal separation of future values" -LogPath $guideLog

    Show-WalkthroughStep "KERNEL SECTION 4 — Key/value loading, scores and causal mask" "For each relevant key/value block, the kernel calculates scaled query-key scores. It assigns negative infinity to scores where a key is later than its query. Those positions consequently contribute zero probability."
    Show-CodeRange -Path $codePath -Title "Key/value loop, matrix multiplication, scaling and mask" -StartPattern '^\s*for kb in range\(nkt\):' -EndPattern '^\s*mt=ct\.max\(s,axis=1,keepdims=True\)'
    $node = Find-NodeId 'test_future_keys_cannot_change_earlier_query_outputs'
    Invoke-GuidedNode -NodeId $node -Label "Validate that future keys cannot affect earlier outputs" -LogPath $guideLog

    Show-WalkthroughStep "KERNEL SECTION 5 — Online softmax and weighted values" "The current score block is combined with the running maximum, normalization sum and weighted-value total. The previous totals are rescaled before the current block is added. This produces the same normalized result without saving the full sequence-by-sequence probability matrix."
    Show-CodeRange -Path $codePath -Title "Running maximum, rescaling, probabilities and P-times-V update" -StartPattern '^\s*mt=ct\.max\(s,axis=1,keepdims=True\)' -EndPattern '^\s*out=u/l'
    $node = Find-NodeId 'test_zero_logits_equal_causal_prefix_average'
    Invoke-GuidedNode -NodeId $node -Label "Validate the softmax recurrence with a known mathematical result" -LogPath $guideLog

    Show-WalkthroughStep "KERNEL SECTION 6 — Normalize and write output" "After all relevant key/value blocks have been processed, the weighted-value total is divided by the running normalization sum. The result is reshaped and written to the FP16 output tensor."
    Show-CodeRange -Path $codePath -Title "Final normalization and output store" -StartPattern '^\s*out=u/l' -EndPattern '^\s*return kernel'
    $node = Find-NodeId 'test_constant_v_is_preserved_for_all_required_dimensions'
    Invoke-GuidedNode -NodeId $node -Label "Validate the final output with constant value vectors" -LogPath $guideLog

    Show-WalkthroughStep "BASELINE REVIEW" "The comparison baseline explicitly materializes the score matrix and probability matrix. It is intentionally unfused FP32 CuPy, not FlashAttention or optimized PyTorch SDPA. The baseline is a performance reference, not the expected-answer generator used by the tests."
    Show-CodeRange -Path (Join-Path $repoRoot "src\baseline.py") -Title "Unfused FP32 CuPy baseline" -StartPattern '^\s*def launch_device\(q_gpu, k_gpu, v_gpu\):' -EndPattern '^\s*def launch\(q: np\.ndarray'
    Pause-BeforeExecution -Prompt "Read how the baseline computes scores, the causal mask, softmax and the weighted-value result. Press Enter to begin the 48 individual GPU tests."

    Show-WalkthroughStep "THE 48 REQUIRED GPU TEST EXECUTIONS" "Each test is shown with its source and purpose, then run as a separate pytest node. Parameterized cases have separate node IDs, so the result identifies the exact dimensions, tile or invalid-input variant."
    $index = 0
    foreach ($node in $script:nodeIds) {
        $index++
        Invoke-GuidedNode -NodeId $node -Label ("GPU TEST {0}/48" -f $index) -LogPath $guideLog
    }
    Write-Host "GUIDED_GPU_TESTS=48 PASSED" -ForegroundColor Green
    Write-Host "GUIDED_GPU_TEST_LOG=$guideLog"

    Show-WalkthroughStep "THE 29 LEGAL NUMERICAL CONFIGURATIONS" "These are separate numerical comparisons against the independent FP32 NumPy calculation. A case fixes B, H, N and D; a tile selects query-block and key/value-block sizes. A pair is included only when N divides evenly by both tile dimensions."
    Show-CodeRange -Path $coreTestPath -Title "Core per-case/tile comparison assertion" -StartPattern '^\s*def test_cutile_matches_independent_reference\(' -EndPattern '^\s*@pytest\.mark\.parametrize\('
    $registryPath = Join-Path $repoRoot "tests\legal_attention_configs.json"
    $registryScript = Join-Path $repoRoot "scripts\legal_config_registry.py"
    & $py $registryScript validate --config $registryPath
    if ($LASTEXITCODE -ne 0) { throw "Legal numerical configuration JSON failed validation." }
    $registry = Get-Content -Raw -LiteralPath $registryPath | ConvertFrom-Json
    $cases = @($registry.cases)
    $tiles = @($registry.tile_candidates)
    $legalPairs = @(
        foreach ($case in $cases) {
            foreach ($tile in $tiles) {
                if (($case.N % $tile.tile_q) -eq 0 -and ($case.N % $tile.tile_k) -eq 0) {
                    [pscustomobject]@{ Case = $case; Tile = $tile }
                }
            }
        }
    )
    $totalLegalPairs = $legalPairs.Count
    if ($totalLegalPairs -lt 1) { throw "The JSON list has no legal numerical configurations." }
    $pairIndex = 0
    foreach ($pair in $legalPairs) {
        $case = $pair.Case
        $tile = $pair.Tile
        $pairIndex++
        $pairLog = Join-Path $logDir ("guided_numerical_{0}_{1}x{2}_{3}.log" -f $stamp,$case.name,$tile.tile_q,$tile.tile_k)
        $pairJson = Join-Path $logDir ("guided_numerical_{0}_{1}_{2}x{3}.json" -f $stamp,$case.name,$tile.tile_q,$tile.tile_k)
        Show-WalkthroughStep -Title ("NUMERICAL CONFIGURATION {0}/{1} — {2}, tile {3}x{4}" -f $pairIndex,$totalLegalPairs,$case.name,$tile.tile_q,$tile.tile_k) -Explanation ("Shape B={0}, H={1}, N={2}, D={3}. B is batch size; H is head count; N is sequence length; D is vector width. This individual run checks this workload/tile combination against the independent FP32 NumPy calculation." -f $case.B,$case.H,$case.N,$case.D)
        Write-Host "TEST SOURCE: $coreTestPath — test_cutile_matches_independent_reference" -ForegroundColor Yellow
        Write-Host "NUMERICAL RULE: abs(actual - reference) <= 0.02 + 0.02 * abs(reference)"
        Pause-BeforeExecution -Prompt "Press Enter to run this one numerical configuration."
        & $py (Join-Path $repoRoot "scripts\mentor_numerical_matrix.py") --config $registryPath --case $case.name --tile-q $tile.tile_q --tile-k $tile.tile_k --output $pairJson 2>&1 | Tee-Object -FilePath $matrixLog -Append
        $matrixExit = $LASTEXITCODE
        if ($matrixExit -ne 0) { throw "Numerical configuration failed: $($case.name) tile $($tile.tile_q)x$($tile.tile_k). Log: $matrixLog" }
        if (!(Test-Path -LiteralPath $pairJson)) { throw "Numerical result file was not written: $pairJson" }
        $pairResult = Get-Content -Raw $pairJson | ConvertFrom-Json
        if ($pairResult.legal_pair_count -ne 1 -or $pairResult.pass_count -ne 1 -or $pairResult.fail_count -ne 0) {
            throw "Numerical output did not report exactly one passing pair: $($case.name) tile $($tile.tile_q)x$($tile.tile_k)"
        }
    }
    Write-Host ("GUIDED_NUMERICAL_MATRIX={0} PASSED" -f $totalLegalPairs) -ForegroundColor Green
    Write-Host "GUIDED_NUMERICAL_MATRIX_LOG=$matrixLog"

    if (!$SkipTileExperiment) {
        $tileFile = Join-Path $logDir "guided_tile_sweep_$stamp.json"
        Show-WalkthroughStep "THREE-SESSION TILE EXPERIMENT" "Each session measures ten warmups plus 30 CUDA-event samples per case/tile configuration; each sample averages five launches. The tile order is changed between sessions so a configuration is not always measured in the same position. The report retains all three session values and the aggregate."
        Pause-BeforeExecution -Prompt "This is the longer timing experiment. Press Enter to run all seven workloads."
        & $py (Join-Path $repoRoot "benchmarks\benchmark_attention_sessions.py") --output $tileFile --warmup 10 --iterations 30 --repeats-per-sample 5 2>&1 | Tee-Object -FilePath (Join-Path $logDir "guided_tile_sweep_$stamp.log")
        if ($LASTEXITCODE -ne 0) { throw "Three-session tile experiment failed" }
    }

    if (!$SkipLatencyHintExperiment) {
        $hintFile = Join-Path $logDir "guided_latency_hints_$stamp.json"
        Show-WalkthroughStep "LOAD-LATENCY HINT EXPERIMENT" "This changes one scheduling hint while holding each workload and tile shape fixed. Every hint must pass numerical validation before its timing is accepted. This session does not launch the profiler."
        Pause-BeforeExecution -Prompt "Press Enter to run the 36 hint/session trials."
        & $py (Join-Path $repoRoot "benchmarks\benchmark_load_latency_hints.py") --output $hintFile --warmup 10 --iterations 30 --repeats-per-sample 5 --sessions 3 2>&1 | Tee-Object -FilePath (Join-Path $logDir "guided_latency_hints_$stamp.log")
        if ($LASTEXITCODE -ne 0) { throw "Load-latency hint experiment failed" }
    }

    Write-Host ""
    Write-Host "GUIDED_WALKTHROUGH=PASS" -ForegroundColor Green
    Write-Host "GUIDED_TEST_COUNT=48"
    Write-Host ("GUIDED_NUMERICAL_CONFIGURATION_COUNT={0}" -f $totalLegalPairs)
    Write-Host "LOG_DIR=$logDir"
    Write-Host "Nsight Compute is intentionally deferred to the later profiler session."
}
finally {
    if ($null -eq $oldRequireCutile) { Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue }
    else { $env:REQUIRE_CUTILE = $oldRequireCutile }
    Pop-Location
}
