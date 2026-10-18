from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    ".gitignore",
    "README.md",
    "docs/README.md",
    "docs/TODO_CHECKLIST.md",
    "docs/EXPERIMENT_REPORT.md",
    "results/STATUS.md",
    "benchmarks/benchmark_v5.py",
    "scripts/ptxas_tile_report.py",
    ".github/workflows/a100-nsys-timeline.yml",
    ".github/workflows/a100-nsight.yml",
    ".github/workflows/a100-ptxas-report.yml",
    ".github/workflows/a100-v5-experiment.yml",
    "run_attention.py",
    "docs/DESIGN.md",
    "docs/TESTING_CONTRACT.md",
    "docs/BASELINE_LADDER.md",
    "docs/PROGRESS.md",
    "docs/TEST_MATRIX.md",
    "docs/CUDA_VARIANTS.md",
    "docs/ADMIN_CHECKLIST.md",
    "results/README.md",
    "src/reference.py",
    "src/online_reference.py",
    "src/baseline.py",
    "src/naive_cuda_attention.py",
    "src/custom_cuda_attention.py",
    "src/warp_cuda_attention.py",
    "src/shared_tile_cuda_attention.py",
    "scripts/cutile_sanity_check.py",
    "scripts/cutile_smoke.py",
    "scripts/setup_cutile_windows.ps1",
    "scripts/start_cutile_docker_runner.ps1",
    "docker/cutile-runner/Dockerfile",
    "docker/cutile-runner/entrypoint.sh",
    "docker/cutile-runner/compose.yml",
    "docs/CUTILE_DOCKER_RUNNER.md",
    ".github/workflows/cutile-rtx4060-docker.yml",
    "docs/LOCAL_GPU_WINDOWS.md",
    "src/torch_reference.py",
    "src/cutile_attention.py",
    "benchmarks/benchmark_attention.py",
    "benchmarks/profile_fused.py",
    "tests/test_reference_contract.py",
    "tests/test_reference_causal.py",
    "tests/test_reference_edge_cases.py",
    "tests/test_reference_metamorphic.py",
    "tests/test_reference_exhaustive.py",
    "docs/KERNEL_VERSIONS.md",
    "docs/EXHAUSTIVE_TEST_PLAN.md",
    "src/variants/shared_tile_v1.py",
    "src/variants/__init__.py",
    "src/variants/shared_tile_v2.py",
    ".github/workflows/a100-native-full.yml",
    "tests/test_online_reference.py",
    "tests/test_launch_contract.py",
    "tests/test_cutile_config.py",
    "tests/test_gpu_correctness.py",
    "tests/test_gpu_edge_cases.py",
    "tests/test_gpu_baselines.py",
    "tests/test_cases.py",
    ".github/workflows/a100-native-cuda.yml",
    ".github/workflows/a100-ncu-access.yml",
    "scripts/ncu_smoke.cu",
    "scripts/setup_native_env.sh",
    "tests/test_gpu_device_matrix.py",
    "tests/test_gpu_shared_tile_exhaustive.py",
    "tests/test_gpu_kernel_versions.py",
    "tests/test_gpu_kernel_v3.py",
    "tests/test_tooling_compile.py",
    "benchmarks/benchmark_kernel_v3.py",
    "src/variants/shared_tile_v3.py",
    ".github/workflows/a100-kernel-version-benchmark.yml",
    "benchmarks/benchmark_kernel_versions.py",
)

def test_required_project_files_exist():
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    assert not missing, f"Missing required project files: {missing}"

def test_required_benchmark_case_dimensions_are_present():
    text = (ROOT / "tests/cases.py").read_text()
    for dimension in ("32", "64", "128"):
        assert dimension in text
    assert "short_d64" in text
    assert "medium_d128" in text
    assert "long_d128" in text

def test_progress_log_mentions_current_environment_gate():
    progress = (ROOT / "docs/PROGRESS.md").read_text()
    assert "R550.90.07" in progress
    assert "cudaErrorSystemDriverMismatch" in progress


def test_root_readme_links_project_tracking_and_evidence():
    readme = (ROOT / "README.md").read_text()
    for path in (
        "docs/TODO_CHECKLIST.md",
        "docs/README.md",
        "docs/EXPERIMENT_REPORT.md",
        "docs/TESTING_CONTRACT.md",
        "docs/TEST_MATRIX.md",
        "docs/PROGRESS.md",
        "results/STATUS.md",
    ):
        assert path in readme, f"Root README must link or refer to {path}"
    assert "not actual NVIDIA cuTile" in readme or "not actual cuTile" in readme


def test_supported_cutile_profile_workflow_gates_profiles_on_correctness():
    workflow = (ROOT / ".github/workflows/a100-nsight.yml").read_text()
    assert "Verify supported cuTile runtime" in workflow
    assert "Run actual cuTile correctness matrix" in workflow
    assert "tests/test_gpu_correctness.py" in workflow
    assert "tests/test_gpu_edge_cases.py" in workflow
    assert "medium_d128_t64x64" in workflow
    assert "longer_batched_d128_t64x32" in workflow


def test_docker_cutile_runner_is_ephemeral_and_unprivileged():
    compose = (ROOT / "docker/cutile-runner/compose.yml").read_text()
    entrypoint = (ROOT / "docker/cutile-runner/entrypoint.sh").read_text()
    workflow = (ROOT / ".github/workflows/cutile-rtx4060-docker.yml").read_text()
    smoke = (ROOT / "scripts/cutile_smoke.py").read_text()
    assert "gpus: all" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose and "ALL" in compose
    assert "/var/run/docker.sock" not in compose
    assert "privileged: true" not in compose
    assert "--ephemeral" in entrypoint
    assert "unset RUNNER_TOKEN" in entrypoint
    assert "workflow_dispatch:" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "contents: read" in workflow
    assert "global ct" in smoke, "cuTile smoke kernel annotations need module-global ct resolution"
