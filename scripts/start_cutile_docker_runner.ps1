param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RunnerToken,

    [string]$RunnerName = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker/cutile-runner/compose.yml"
$GpuProbeImage = "nvidia/cuda:12.4.1-base-ubuntu22.04"
Set-Location -LiteralPath $RepoRoot

# This launcher accepts a registration token supplied locally by the owner.
# It does not use GitHub CLI to mint tokens or dispatch a workflow.
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Open a new PowerShell after installing Docker Desktop."
}

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & docker info *> $null
    $dockerInfoExit = $LASTEXITCODE
    & docker compose version *> $null
    $composeVersionExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($dockerInfoExit -ne 0) {
    throw "Docker Engine is not ready. Start Docker Desktop and retry."
}
if ($composeVersionExit -ne 0) {
    throw "Docker Compose is unavailable."
}

# Respect the no-unapproved-download requirement: do not pull probe images.
$ErrorActionPreference = "Continue"
try {
    & docker image inspect $GpuProbeImage *> $null
    $probeImageExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($probeImageExit -ne 0) {
    throw "GPU probe image '$GpuProbeImage' is not local. Do not auto-download it; obtain approval and run 'docker pull $GpuProbeImage' separately."
}

# Hard gate: prove the Linux container can see the physical RTX 4060.
$ErrorActionPreference = "Continue"
try {
    $gpuProbe = & docker run --rm --gpus all $GpuProbeImage nvidia-smi 2>&1
    $gpuProbeExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($gpuProbeExit -ne 0 -or ($gpuProbe -join [Environment]::NewLine) -notmatch "RTX 4060") {
    $gpuProbe | ForEach-Object { Write-Host "$_" }
    throw "Docker GPU passthrough did not expose the RTX 4060. No runner was started."
}
$gpuProbe | ForEach-Object { Write-Host "$_" }

if ([string]::IsNullOrWhiteSpace($RunnerName)) {
    $RunnerName = "cutile-rtx4060-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
}
if ($RunnerName -notmatch '^[A-Za-z0-9._-]{1,64}$') {
    throw "RunnerName must be 1-64 characters using letters, numbers, dot, underscore, or hyphen."
}

Write-Host ""
Write-Host "Starting persistent self-hosted runner '$RunnerName'."
Write-Host "The runner will NOT auto-dispatch a workflow or auto-remove itself after one job."
Write-Host "Keep the registration token private; do not paste it into chat or commit it."
$env:RUNNER_TOKEN = $RunnerToken
$env:RUNNER_NAME = $RunnerName
$launchError = $null
$ErrorActionPreference = "Continue"
try {
    & docker compose -f $ComposeFile up --build -d 2>&1 | ForEach-Object { Write-Host "$_" }
    $composeUpExit = $LASTEXITCODE
    if ($composeUpExit -ne 0) {
        $launchError = "Docker Compose failed to start the runner (exit code $composeUpExit)."
    }
} catch {
    $launchError = $_
} finally {
    Remove-Item Env:RUNNER_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:RUNNER_NAME -ErrorAction SilentlyContinue
    $ErrorActionPreference = $previousPreference
}
if ($launchError) {
    # Compose uses safe defaults for interpolation during "down", so credentials
    # need not remain set in this process to clean up a failed start.
    $ErrorActionPreference = "Continue"
    try {
        & docker compose -f $ComposeFile down --remove-orphans 2>&1 | ForEach-Object { Write-Host "$_" }
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    throw $launchError
}

Write-Host ""
Write-Host "Container started detached. Runner name: $RunnerName"
Write-Host "Check GitHub: Settings > Actions > Runners; wait for this name to show Online."
Write-Host "Then manually run the 'RTX 4060 Docker cuTile validation' workflow on main."
Write-Host ('Live logs: docker compose -f "' + $ComposeFile + '" logs -f runner')
Write-Host "When finished, remove the runner in GitHub Settings > Actions > Runners, then stop the container with:"
Write-Host ('  docker compose -f "' + $ComposeFile + '" down --remove-orphans')
Write-Host "Compose cleanup no longer requires RUNNER_TOKEN or RUNNER_NAME to be present."
