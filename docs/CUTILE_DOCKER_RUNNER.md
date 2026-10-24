# Docker runner for the RTX 4060

This is an optional route for running actual cuTile tests in a container. The native Windows runner is the primary route. The A100/R550 host is for ordinary CUDA comparisons only.

## Before starting

- Docker Desktop and its WSL2 backend must be running.
- The image nvidia/cuda:12.4.1-base-ubuntu22.04 must already exist locally; the launcher does not pull it.
- Create a self-hosted runner registration token in GitHub Settings > Actions > Runners. Do not paste it into chat, shell history or the repository.

From the project root, update the local checkout and check its state:

~~~powershell
git pull --ff-only origin main
git status --short --branch
~~~

Read the token without placing it in the typed command:

~~~powershell
$secureToken = Read-Host "Runner registration token" -AsSecureString
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
~~~

The script checks Docker Compose and GPU passthrough, builds the image and starts a persistent runner. It does not create the registration token, dispatch a workflow or remove the container automatically.

Check Settings > Actions > Runners for the generated runner name and wait for it to show Online. If it does not, inspect:

~~~powershell
docker ps -a --filter "name=cutile-runner-runner-1"
docker logs --tail 120 cutile-runner-runner-1
~~~

When online, manually run RTX 4060 Docker cuTile validation from Actions on main.

## Stop the runner

Delete it in GitHub Settings > Actions > Runners first, then run:

~~~powershell
docker compose -f docker/cutile-runner/compose.yml down --remove-orphans
~~~

## Isolation

The container runs as a non-root user, drops capabilities, enables no-new-privileges, applies process/memory limits, and has no host bind mounts or Docker socket. It does not run privileged or expose ports.

This reduces risk; it is not a complete security boundary. Docker Desktop, WSL2 and the host kernel remain trusted parts of the setup.
