# Ephemeral Docker GitHub Actions runner — RTX 4060

This is the intended remote Actions path for the local RTX 4060. It is separate from the existing A100/R550 runner, which remains an ordinary-CUDA control.

## Host prerequisites

The observed Windows host currently has **no WSL2 and no Docker**, so the runner cannot start yet. Install/start Docker Desktop using its WSL2 backend and install GitHub CLI (gh). A Windows reboot/admin approval may be required to enable WSL2. Do not disable Windows security features to work around setup problems.

Before registration, the launcher requires a successful Linux-container GPU probe that prints the RTX 4060. If that probe fails, it stops before requesting a runner registration token.

## Start the runner

Authenticate the GitHub CLI once using its browser-based flow (gh auth login). The authenticated account must have repository-admin permission to create a self-hosted runner registration token.

Then run this **single project command** in PowerShell from the repository root:

    .\scripts\start_cutile_docker_runner.ps1

The launcher:
1. verifies Docker Engine, Compose v2, GitHub CLI authentication, and RTX 4060 GPU passthrough;
2. requests a short-lived repository runner registration token just in time;
3. builds the local runner image and starts an **ephemeral one-job runner**;
4. waits until that exact runner is online, dispatches the workflow on `main`, and watches the result;
5. removes the token from its process environment before executing jobs, tears down the container, and cleans up a leftover runner registration if dispatch fails.

Do not paste the registration token or any GitHub credential into chat, files, or workflow logs.

## Isolation design

The runner container is non-root, drops Linux capabilities, sets no-new-privileges, has process/memory limits, and has no host bind mounts, no Docker socket, no privileged mode, no exposed ports, and no persistent workspace. It is registered as ephemeral and intended to execute one trusted job only. Workflow permissions are read-only for repository contents and the workflow is manual-dispatch-only with an explicit main-branch guard.

This materially reduces host exposure, but **a container is not a perfect security boundary**: Docker Desktop/WSL2 and the host kernel remain part of the trust boundary. Run only code you trust from main; do not dispatch untrusted PR/fork code to this runner. Do not add repository secrets to the workflow. The job's GitHub token is limited by the workflow's contents: read permission.

## Workflow

The launcher automatically dispatches the RTX 4060 Docker cuTile validation workflow on `main` and waits for its result; no separate Actions UI step is needed. The job checks GPU visibility, installs project test/cuTile dependencies inside the disposable runner, executes the genuine cuTile smoke test, then runs the actual attention correctness matrix with `REQUIRE_CUTILE=1`. It uploads the smoke log and test cache even on failure.

A successful smoke test is not enough to declare attention complete. The correctness job must pass. Benchmarking and at least two real cuTile profiler captures remain separate completion gates.

## Files

- docker/cutile-runner/Dockerfile — non-root runner image.
- docker/cutile-runner/entrypoint.sh — short-lived registration and one-job execution.
- docker/cutile-runner/compose.yml — GPU access and container restrictions.
- scripts/start_cutile_docker_runner.ps1 — preflight, just-in-time token, lifecycle.
- .github/workflows/cutile-rtx4060-docker.yml — manual GPU smoke/correctness workflow.
