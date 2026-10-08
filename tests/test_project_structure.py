from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
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
    "src/variants/shared_tile_v1.py",
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
    "tests/test_gpu_device_matrix.py",
    "tests/test_gpu_shared_tile_exhaustive.py",
    "tests/test_gpu_kernel_versions.py",
    "docs/KERNEL_VERSIONS.md",
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
