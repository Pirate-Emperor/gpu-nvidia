import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "scripts" / "benchmark_torch_sdpa.py",
    ROOT / "scripts" / "autotune_cutile_tiles.py",
    ROOT / "benchmarks" / "benchmark_load_latency_hints.py",
    ROOT / "benchmarks" / "profile_fused.py",
    ROOT / "benchmarks" / "benchmark_torch_model_integration.py",
    ROOT / "src" / "torch_integration.py",
    ROOT / "src" / "extended_attention.py",
    ROOT / "scripts" / "capture_ncu_screenshots.py",
)


def test_optional_experiment_python_sources_parse_without_gpu_dependencies():
    for path in SCRIPTS:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


def test_sdpa_benchmark_help_is_cpu_safe():
    script = ROOT / "scripts" / "benchmark_torch_sdpa.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--iterations" in result.stdout
    assert "--output" in result.stdout

    hint_script = ROOT / "benchmarks" / "benchmark_load_latency_hints.py"
    hint_result = subprocess.run(
        [sys.executable, str(hint_script), "--help"],
        cwd=ROOT, text=True, capture_output=True, timeout=20,
    )
    assert hint_result.returncode == 0, hint_result.stderr
    assert "--minimum-gain-pct" in hint_result.stdout
