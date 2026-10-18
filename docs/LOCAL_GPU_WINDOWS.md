# Local GPU Bring-Up — Windows RTX 4060

**Inventory received:** 2026-10-18  
**Purpose:** actual cuTile bring-up on the user's local GPU. This is separate from the existing A100 ordinary-CUDA surrogate evidence.

## 1. Observed machine inventory

| Item | Observed value | Interpretation |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 | Actual local-device candidate |
| Compute capability | 8.9 | Inside NVIDIA cuTile Python's documented 8.x–12.x range |
| Driver | 616.92 | Newer than the cuTile runtime minimum R580 |
| Driver-reported CUDA UMD level | 13.4 | Driver support level; **not** proof that the CUDA Toolkit/NVCC is installed |
| VRAM | 8,188 MiB total, 7,925 MiB free at inventory | Enough to begin smoke/correctness tests; benchmark each case and watch memory use |
| Driver model | WDDM | Native Windows GPU path |
| OS | Windows 10 Pro, x64, build 26300 | Windows x86_64 is on NVIDIA's current cuTile support list |
| NVCC / system Toolkit | `nvcc` not found | Not a blocker if the isolated Python compiler packages install correctly |
| Python | Not installed/available via `python`; Microsoft Store alias responded | Install 64-bit CPython 3.12 for current user |
| WSL2 | Not installed | Not needed for the first attempt |
| Docker | Not installed | Not needed for the first attempt |
| Nsight Compute CLI | Not checked yet | Needed later for profiler runs; verify separately |

## 2. Execution decision

**Proceed with a native Windows x86_64 virtual environment as the first bring-up path.** This is not merely a habitual choice: NVIDIA's current [cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) explicitly supports Windows x86_64, compute capability 8.x–12.x, drivers R580+, and Python 3.10–3.14. The observed GPU/driver/OS satisfy those documented hardware/driver/OS gates. The driver also exceeds the Quickstart's Windows R582.16+ requirement for detailed cuTile profiling.

The remaining gate is software/runtime validation. There is no installed Python or system NVCC reported, and an official NVIDIA issue documents an earlier Windows cuTile + CuPy installation failing in NVRTC with `nvrtc-builtins64_131.dll`; the issue is marked closed/waiting-for-feedback, not as a universal fix. See [NVIDIA cuTile Python issue #57](https://github.com/NVIDIA/cutile-python/issues/57). For that reason, **a real kernel compile + launch + result check is mandatory before running the attention kernel**.

The setup script deliberately avoids system-wide CUDA Toolkit and driver changes. It installs the TileIR compiler through `cuda-tile[tileiras]` and asks current CuPy to install matched CUDA component wheels through `cupy-cuda13x[ctk]`. CuPy documents this [component-wheel installation path](https://docs.cupy.dev/en/stable/install.html); NVIDIA requires the `nvidia-cuda-tileiras`, `nvidia-cuda-nvcc` and `nvidia-nvvm` versions to match major.minor when using the cuTile extra.

## 3. Bootstrap sequence

1. Install **64-bit CPython 3.12** for the current Windows user from [Python for Windows](https://www.python.org/downloads/windows/). Enable the Python Launcher, then open a new PowerShell and check `py -3.12 --version`. Use a supported 3.10–3.14 CPython build; 3.12 aligns with the repository's existing CI version.
2. Check out the private `xemptynothing/gpu-nvidia` repository into a project-only working directory. If Git for Windows is available, run `git clone https://github.com/xemptynothing/gpu-nvidia.git` and then `cd gpu-nvidia`; otherwise use GitHub Desktop or install Git for Windows. Authenticate through your normal browser/Git Credential Manager flow; do not paste a PAT or any credential into chat. Do not put personal documents or credentials in the project directory.
3. From the repository root, run:
   
       .\scripts\setup_cutile_windows.ps1

   This creates `.venv-cutile-windows`, installs CPU/test requirements plus `cupy-cuda13x[ctk]` and `cuda-tile[tileiras]`, and runs `scripts/cutile_smoke.py`. It does not install drivers, system Toolkit, WSL2, Docker, or a GitHub runner. The smoke log is stored under `%LOCALAPPDATA%\GPU-NVIDIA\logs`.
4. Require the log to show both `OUTPUT_CHECK=PASS` and `CUTILE_SMOKE=PASS`. The smoke launches a real `cuda.tile` vector-add kernel, synchronizes the GPU, and compares all 4,096 values with a known NumPy result. An import-only pass does not qualify.
5. Only after smoke succeeds, run the actual attention correctness suite without allowing missing dependencies to become skips:

       $env:REQUIRE_CUTILE = "1"
       .\.venv-cutile-windows\Scripts\python.exe -m pytest -q tests/test_gpu_correctness.py tests/test_gpu_edge_cases.py
       $testExit = $LASTEXITCODE
       Remove-Item Env:REQUIRE_CUTILE
       if ($testExit -ne 0) { throw "Required cuTile correctness tests failed" }

   The new `REQUIRE_CUTILE=1` mode makes missing CuPy/cuTile an error while allowing an RTX 4060 (not an A100) to run the suite. Save the console output and exact `git rev-parse HEAD` with the result.
6. If correctness passes, exercise the required CLI with a small case. These commands keep generated inputs/outputs under the local logs folder rather than the repository:
   
       $py = ".\\.venv-cutile-windows\\Scripts\\python.exe"
       $work = Join-Path $env:LOCALAPPDATA "GPU-NVIDIA\\logs"
       $input = Join-Path $work "attention_cli_input.npz"
       $output = Join-Path $work "attention_cli_output.npy"
       & $py scripts\\generate_inputs.py --output $input --batch 1 --heads 1 --seq 64 --head-dim 32
       if ($LASTEXITCODE -ne 0) { throw "Input generation failed" }
       & $py run_attention.py --input $input --output $output --backend cutile --tile-q 64 --tile-k 64
       if ($LASTEXITCODE -ne 0) { throw "Attention CLI failed" }
       $env:ATTENTION_INPUT = $input
       $env:ATTENTION_OUTPUT = $output
       $env:PYTHONPATH = Join-Path (Get-Location) "src"
       & $py -c "import os,numpy as np; from attention_io import load_attention_npz; from reference import causal_attention_fp32; q,k,v=load_attention_npz(os.environ['ATTENTION_INPUT']); expected=causal_attention_fp32(q,k,v,output_dtype=np.float16); actual=np.load(os.environ['ATTENTION_OUTPUT']); assert actual.shape==expected.shape and actual.dtype==np.float16 and np.isfinite(actual).all(); np.testing.assert_allclose(actual,expected,rtol=2e-2,atol=2e-2); print('CLI_OUTPUT_CHECK=PASS')"
       $checkExit = $LASTEXITCODE
       Remove-Item Env:ATTENTION_INPUT,Env:ATTENTION_OUTPUT,Env:PYTHONPATH -ErrorAction SilentlyContinue
       if ($checkExit -ne 0) { throw "CLI output/reference check failed" }

   Do not benchmark or profile a kernel that has not passed correctness. Use at least 10 warmups, at least 30 CUDA-event samples, cold/JIT separate, median/P90, throughput, and raw results.
7. Before detailed profiling, install/locate Nsight Compute and verify non-admin profiling access. The driver meets the documented version threshold, but the profiler binary and counter permission have not yet been checked.

## 4. Security boundary and runner policy

**Do not register a self-hosted GitHub Actions runner on the personal Windows host yet.** A runner executes repository workflow commands with the permissions of its Windows account; it is not a sandbox. First prove native cuTile smoke and correctness manually. If remote Actions execution is still needed, design and test a dedicated non-admin execution boundary with only the repository workspace and required cache/output paths, no personal profile/credential mounts, no admin privilege, no unrelated drive access, and trusted manual workflows from `main` only. Keep all runner registration/connection credentials local; never paste them into chat or make them available to job code.

No driver upgrade or machine-wide installation should be performed by the project scripts.
