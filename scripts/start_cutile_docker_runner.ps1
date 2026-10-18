$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker/cutile-runner/compose.yml"
$WorkflowFile = "cutile-rtx4060-docker.yml"
Set-Location -LiteralPath $RepoRoot

foreach ($command in @("docker", "gh")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command '$command' was not found. Install Docker Desktop with WSL2 GPU support and GitHub CLI, then open a new PowerShell."
    }
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not running. Start Docker Desktop with the WSL2 backend, then retry."
}
& docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is required."
}
& gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run gh auth login locally; do not paste credentials into chat."
}

# Hard gate: prove Docker Desktop can pass the RTX 4060 into a Linux container.
$gpuProbe = & docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0 -or ($gpuProbe -join "`n") -notmatch "RTX 4060") {
    $gpuProbe | Write-Host
    throw "Docker GPU passthrough did not expose the RTX 4060. Do not register a runner until WSL2/Docker GPU support works."
}
$gpuProbe | Write-Host

# Request the short-lived registration token just in time; never write it to disk.
$tokenJson = & gh api repos/xemptynothing/gpu-nvidia/actions/runners/registration-token --method POST
if ($LASTEXITCODE -ne 0) {
    throw "Could not obtain a repository runner registration token. The gh-authenticated account needs repository admin permission."
}
$tokenObject = $tokenJson | ConvertFrom-Json
if (-not $tokenObject.token) {
    throw "GitHub did not return a runner registration token."
}

$env:RUNNER_TOKEN = [string]$tokenObject.token
$env:RUNNER_NAME = "cutile-rtx4060-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
$runnerName = $env:RUNNER_NAME
$workflowExit = 1
$dispatchStart = [DateTime]::UtcNow
try {
    Write-Host "Building and starting one-job, ephemeral GPU runner."
    & docker compose -f $ComposeFile up --build -d
    if ($LASTEXITCODE -ne 0) { throw "Could not start Docker runner container." }

    # Wait for this exact runner to register and become online before dispatching.
    $online = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        $runnerJson = & gh api repos/xemptynothing/gpu-nvidia/actions/runners
        if ($LASTEXITCODE -eq 0) {
            $runnerList = $runnerJson | ConvertFrom-Json
            $match = $runnerList.runners | Where-Object { $_.name -eq $runnerName -and $_.status -eq "online" } | Select-Object -First 1
            if ($match) { $online = $true; break }
        }
        Start-Sleep -Seconds 2
    }
    if (-not $online) { throw "The ephemeral runner did not become online within 180 seconds." }

    Write-Host "Runner is online. Dispatching the main-branch cuTile smoke/correctness workflow."
    $dispatchStart = [DateTime]::UtcNow
    & gh workflow run $WorkflowFile --ref main
    if ($LASTEXITCODE -ne 0) { throw "GitHub rejected the workflow dispatch." }

    # Resolve the newly dispatched run rather than accidentally watching an older run.
    $runId = $null
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $runsJson = & gh run list --workflow $WorkflowFile --branch main --limit 10 --json databaseId,event,createdAt,status
        if ($LASTEXITCODE -eq 0) {
            $runs = $runsJson | ConvertFrom-Json
            $newRun = $runs |
                Where-Object { $_.event -eq "workflow_dispatch" -and ([DateTime]::Parse($_.createdAt).ToUniversalTime() -ge $dispatchStart.AddSeconds(-10)) } |
                Sort-Object createdAt -Descending |
                Select-Object -First 1
            if ($newRun) { $runId = [string]$newRun.databaseId; break }
        }
        Start-Sleep -Seconds 2
    }
    if (-not $runId) { throw "Workflow was dispatched but its run ID could not be resolved." }

    Write-Host "Watching GitHub Actions run $runId. The runner will be removed after its one job."
    & gh run watch $runId --exit-status
    $workflowExit = $LASTEXITCODE
}
finally {
    Remove-Item Env:RUNNER_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:RUNNER_NAME -ErrorAction SilentlyContinue
    & docker compose -f $ComposeFile down --remove-orphans *> $null

    # If dispatch failed before the ephemeral job consumed the runner, remove
    # its offline registration so the repository does not accumulate stale runners.
    try {
        $remainingJson = & gh api repos/xemptynothing/gpu-nvidia/actions/runners
        if ($LASTEXITCODE -eq 0) {
            $remaining = $remainingJson | ConvertFrom-Json
            $stale = $remaining.runners | Where-Object { $_.name -eq $runnerName } | Select-Object -First 1
            if ($stale) {
                & gh api "repos/xemptynothing/gpu-nvidia/actions/runners/$($stale.id)" --method DELETE *> $null
            }
        }
    } catch {
        Write-Warning "Could not check/remove a leftover runner registration; inspect repository Settings > Actions > Runners."
    }
}
if ($workflowExit -ne 0) {
    throw "The cuTile GitHub Actions workflow failed with exit code $workflowExit."
}
Write-Host "cuTile smoke/correctness workflow completed successfully."
