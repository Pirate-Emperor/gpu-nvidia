# Persistent Docker GitHub Actions runner — RTX 4060

This is the local Actions path for actual cuTile attention tests on the inventoried RTX 4060. The A100/R550 machine remains an ordinary-CUDA control; it is not the actual-cuTile route.

## Host prerequisites

- Docker Desktop with its WSL2 backend is running.
- GitHub CLI is not required by the launcher.
- The probe image `nvidia/cuda:12.4.1-base-ubuntu22.04` must already be present locally. The launcher does not pull it automatically.
- The user must create a repository self-hosted runner registration token locally from **Settings > Actions > Runners > New self-hosted runner** (Linux, x64). Never paste the token into chat or commit it.

## Start the runner

First pull the repository's latest `main` and confirm you are at the project root:

    git pull --ff-only origin main
    git status --short --branch

The script accepts the owner-supplied token, checks Docker Compose and RTX 4060 GPU passthrough, builds the runner image from this repository's Dockerfile, then starts a **persistent** runner. It does not request a token through GitHub CLI, dispatch a workflow, wait three minutes for GitHub API registration polling, or tear down the container automatically.

Avoid putting the token literally in the PowerShell command line, because commands may be retained in shell history. This prompt keeps the token out of the typed command history while still passing it as the script parameter:

    $secureToken = Read-Host "Paste the repository runner registration token" -AsSecureString
    $tokenPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    try {
        $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPtr)
        .\scripts\start_cutile_docker_runner.ps1 -RunnerToken $plainToken
    } finally {
        if ($tokenPtr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPtr)
        }
        Remove-Variable plainToken,secureToken,tokenPtr -ErrorAction SilentlyContinue
    }

The script generates and prints a unique runner name, starts detached, then returns to PowerShell. Check **Settings > Actions > Runners** and wait for that exact name to show **Online**. If it does not come online, inspect the container logs before retrying:

    docker ps -a --filter "name=cutile-runner-runner-1"
    docker logs --tail 120 cutile-runner-runner-1

When the runner is online, manually launch **RTX 4060 Docker cuTile validation** from the Actions tab on branch `main`. This workflow is limited to `workflow_dispatch`, read-only contents permission, and the main branch. The job installs the project test dependencies in the container, runs the genuine cuTile smoke test, then runs the attention correctness matrix with `REQUIRE_CUTILE=1`.

## Stop and remove the runner

When finished, first delete the runner under **Settings > Actions > Runners**. Then, from the repository root, stop and remove the container/network:

    docker compose -f docker/cutile-runner/compose.yml down --remove-orphans

The Compose file now uses safe defaults for `RUNNER_TOKEN` and `RUNNER_NAME` interpolation during cleanup, so stopping the container does not require retaining the registration token in the PowerShell environment.

## Isolation and security notes

The runner container runs as a non-root user, drops Linux capabilities, sets `no-new-privileges`, has process/memory limits, and has no host bind mounts, Docker socket, privileged mode, exposed ports, or persisted workspace. It has the `gpu` and `cutile-rtx4060` labels.

This is a risk reduction, **not a perfect security boundary**. Docker Desktop/WSL2 and the host kernel remain part of the trust boundary. The runner executes commands from repository workflows with the permissions available in the container. Only run reviewed code from `main`; never route fork/untrusted pull-request code or jobs containing personal secrets to this runner. The current persistent runner can accept more than one job until it is removed; do not leave it registered longer than necessary.

The registration token stays local, is short-lived, and must not be pasted into ChatGPT or logged. Compose injects it into the container environment for runner configuration; users with access to the local Docker daemon can inspect container configuration, so protect the host account and revoke/remove the runner when finished. The entrypoint unsets the token before starting the runner process.

## Files

- `docker/cutile-runner/Dockerfile` — non-root runner image.
- `docker/cutile-runner/entrypoint.sh` — persistent runner registration and execution.
- `docker/cutile-runner/compose.yml` — GPU access and container restrictions.
- `scripts/start_cutile_docker_runner.ps1` — owner-supplied token, preflight, launch.
- `.github/workflows/cutile-rtx4060-docker.yml` — manual GPU smoke/correctness workflow.
