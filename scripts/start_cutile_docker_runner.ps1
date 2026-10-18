$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker/cutile-runner/compose.yml"
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
try {
    Write-Host "Building and starting one-job, ephemeral GPU runner."
    Write-Host "Dispatch only the RTX 4060 Docker cuTile validation workflow on main."
    & docker compose -f $ComposeFile up --build --abort-on-container-exit --exit-code-from runner
    $runnerExit = $LASTEXITCODE
}
finally {
    Remove-Item Env:RUNNER_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:RUNNER_NAME -ErrorAction SilentlyContinue
    & docker compose -f $ComposeFile down --remove-orphans *> $null
}
if ($runnerExit -ne 0) {
    throw "Docker runner exited with code $runnerExit."
}
